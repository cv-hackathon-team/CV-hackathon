"""Behaviour tests for corruption, loaders, classical methods and metrics.

Tests marked with ``needs_data`` run against the real datasets and are skipped if absent.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.classical.guided_filter import guided_filter_numpy, guided_filter_ximgproc, guided_upsample
from src.classical.hole_filling import fill_holes
from src.classical.joint_bilateral import joint_bilateral_filter_numpy
from src.datasets.corruption import (
    MODES,
    CorruptedDataset,
    corrupt_depth,
    downsample_area,
    make_rng,
    paired_random_crop_flip,
)
from src.datasets.kitti import KITTIDepthCompletionDataset, open_kitti
from src.datasets.middlebury import Middlebury2014Dataset, disparity_to_depth, parse_calib, read_pfm
from src.datasets.nyu import build_nyu_dataset, make_train_val_split, read_scene_names
from src.evaluation.evaluate import evaluate_dataset, get_methods, summarise
from src.evaluation.metrics import boundary_f_score, mae, psnr, rmse, ssim
import pandas as pd
from src.utils import load_config, resolve_dataset_root

CONFIG = load_config()
PARAMS = CONFIG["corruption"]["levels"]["medium"]
ROOT = resolve_dataset_root(CONFIG)
NYU_MAT = ROOT / CONFIG["datasets"]["nyu"]["mat_file"]
needs_nyu = pytest.mark.skipif(not NYU_MAT.is_file(), reason="NYU not available")


def _smooth_depth(height: int = 64, width: int = 64) -> tuple[np.ndarray, np.ndarray]:
    depth = np.tile(np.linspace(1, 5, width, dtype=np.float32), (height, 1))
    return depth, np.ones_like(depth, dtype=bool)


def test_area_downsample_ignores_invalid_pixels() -> None:
    depth = np.array([[2, 0], [2, 2]], dtype=np.float32)
    lr, valid = downsample_area(depth, depth > 0, 2)
    assert valid.all()
    assert lr[0, 0] == pytest.approx(2.0)


@pytest.mark.parametrize("mode", MODES)
def test_corruption_is_reproducible(mode: str) -> None:
    depth, valid = _smooth_depth()
    a = corrupt_depth(depth, valid, 2, mode, PARAMS, make_rng(7, 3, 2, mode))
    b = corrupt_depth(depth, valid, 2, mode, PARAMS, make_rng(7, 3, 2, mode))
    assert np.array_equal(a.lr_depth, b.lr_depth)
    assert np.array_equal(a.validity_mask, b.validity_mask)


@pytest.mark.parametrize("mode", ["noisy", "holes", "sparse", "mixed"])
def test_corruption_varies_with_seed(mode: str) -> None:
    depth, valid = _smooth_depth()
    a = corrupt_depth(depth, valid, 2, mode, PARAMS, make_rng(1, 0, 2, mode))
    b = corrupt_depth(depth, valid, 2, mode, PARAMS, make_rng(2, 0, 2, mode))
    assert not np.array_equal(a.lr_depth, b.lr_depth)


def test_invalid_pixels_are_zero_in_every_mode() -> None:
    depth, valid = _smooth_depth()
    depth[:, 30:34] += 3
    for mode in MODES:
        result = corrupt_depth(depth, valid, 2, mode, PARAMS, make_rng(1, 0, 2, mode))
        assert (result.lr_depth[~result.validity_mask] == 0).all()
        assert (result.lr_depth[result.validity_mask] > 0).all()


def test_sparse_keeps_requested_percentage() -> None:
    depth, valid = _smooth_depth()
    result = corrupt_depth(depth, valid, 2, "sparse", PARAMS, make_rng(1, 0, 2, "sparse"))
    assert result.validity_mask.mean() == pytest.approx(PARAMS["sparse_keep_percent"] / 100, abs=0.01)


def test_noise_grows_with_depth() -> None:
    depth = np.ones((400, 400), dtype=np.float32) * 2
    depth[:, 200:] = 8
    result = corrupt_depth(depth, depth > 0, 2, "noisy", PARAMS, make_rng(1, 0, 2, "noisy"))
    near, far = result.lr_depth[:, :90], result.lr_depth[:, 110:]
    assert far.std() == pytest.approx(4 * near.std(), rel=0.1)


def test_holes_are_blobs() -> None:
    depth, valid = _smooth_depth(128, 128)
    result = corrupt_depth(depth, valid, 2, "holes", PARAMS, make_rng(1, 0, 2, "holes"))
    holes = (~result.validity_mask).astype(np.uint8)
    assert holes.mean() == pytest.approx(PARAMS["hole_ratio"], abs=0.01)
    assert cv2.connectedComponents(holes)[0] - 1 < holes.sum() / 4


def test_boundary_corruption_only_drops_near_edges() -> None:
    depth = np.full((64, 64), 2.0, dtype=np.float32)
    depth[:, 32:] = 4
    result = corrupt_depth(depth, depth > 0, 2, "boundary", PARAMS, make_rng(1, 0, 2, "boundary"))
    dropped_columns = np.flatnonzero((~result.validity_mask).any(axis=0))
    assert dropped_columns.size and dropped_columns.min() >= 13 and dropped_columns.max() <= 17


def test_crop_and_flip_are_paired() -> None:
    height, width = 60, 80
    index_map = np.arange(height * width, dtype=np.float32).reshape(height, width)
    rgb = np.stack([index_map] * 3, axis=2)
    for seed in range(20):
        out_rgb, out_depth, out_valid = paired_random_crop_flip(
            rgb, index_map, index_map > -1, (32, 32), True, np.random.default_rng(seed)
        )
        assert np.array_equal(out_rgb[:, :, 0], out_depth)
        assert out_valid.shape == out_depth.shape == (32, 32)


def test_corrupted_dataset_is_reproducible_and_changes_by_epoch() -> None:
    depth, valid = _smooth_depth(64, 64)
    sample = {
        "rgb": torch.rand(3, 64, 64),
        "depth": torch.from_numpy(depth)[None],
        "valid": torch.from_numpy(valid)[None],
        "name": "x",
    }
    dataset = CorruptedDataset([sample], 2, "mixed", PARAMS, seed=5, crop_size_hw=(32, 32), flip=True)
    first = dataset[0]
    assert torch.equal(first["lr_depth"], dataset[0]["lr_depth"])
    assert first["lr_depth"].shape == (1, 16, 16)
    dataset.set_epoch(1)
    assert not torch.equal(first["lr_depth"], dataset[0]["lr_depth"])


def test_train_val_split_keeps_scenes_separate() -> None:
    scenes = [f"scene_{i // 5}" for i in range(100)]
    train, val = make_train_val_split(np.arange(100), scenes, 0.1, seed=1)
    assert len(train) + len(val) == 100
    assert len(val) == pytest.approx(10, abs=5)
    assert not {scenes[i] for i in train} & {scenes[i] for i in val}


@needs_nyu
def test_nyu_test_split_is_guarded() -> None:
    with pytest.raises(PermissionError):
        build_nyu_dataset(CONFIG, "test")


@needs_nyu
def test_nyu_splits_and_sample_format() -> None:
    train = build_nyu_dataset(CONFIG, "train")
    val = build_nyu_dataset(CONFIG, "val")
    test = build_nyu_dataset(CONFIG, "test", allow_test=True)
    scenes = read_scene_names(NYU_MAT)
    assert len(train) + len(val) == 795 and len(test) == 654
    assert not {scenes[i] for i in val.indices} & {scenes[i] for i in train.indices}
    assert not set(test.indices) & (set(train.indices) | set(val.indices))
    sample = train[0]
    assert sample["rgb"].shape == (3, 480, 640) and sample["depth"].shape == (1, 480, 640)
    assert 0 <= sample["rgb"].min() and sample["rgb"].max() <= 1
    assert torch.equal(sample["valid"], sample["depth"] > 0)


@needs_nyu
def test_nyu_transpose_aligns_rgb_with_depth() -> None:
    """Depth and luminance edges should line up far better than a flipped depth map does."""
    sample = build_nyu_dataset(CONFIG, "train")[0]
    gray = sample["rgb"].mean(0).numpy()
    depth = sample["depth"][0].numpy()
    def correlation(a: np.ndarray, b: np.ndarray) -> float:
        ax, bx = np.abs(np.diff(a, axis=1))[40:-40, 40:-40], np.abs(np.diff(b, axis=1))[40:-40, 40:-40]
        return float(np.corrcoef(ax.ravel(), bx.ravel())[0, 1])
    assert correlation(gray, depth) > correlation(gray, depth[::-1])
    assert correlation(gray, depth) > correlation(gray, depth[:, ::-1])


def test_read_pfm_flips_rows_and_honours_endianness(tmp_path: Path) -> None:
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    for scale, dtype in ((-1.0, "<f4"), (1.0, ">f4")):
        path = tmp_path / f"{dtype[0] == '<'}.pfm"
        path.write_bytes(b"Pf\n2 3\n" + f"{scale}\n".encode() + data[::-1].astype(dtype).tobytes())
        assert np.array_equal(read_pfm(path), data)


def test_disparity_to_depth_formula_and_invalid_values() -> None:
    calib = {"focal": 2000.0, "baseline": 100.0, "doffs": 100.0}
    disparity = np.array([[100.0, np.inf, 0.0, -1.0, np.nan]], dtype=np.float32)
    depth, valid = disparity_to_depth(disparity, calib)
    assert depth[0, 0] == pytest.approx(1.0)
    assert valid.tolist() == [[True, False, False, False, False]]
    assert (depth[~valid] == 0).all()


def test_parse_calib(tmp_path: Path) -> None:
    path = tmp_path / "calib.txt"
    path.write_text("cam0=[2826.171 0 1292.2; 0 2826.171 965.806; 0 0 1]\ndoffs=123.77\n"
                    "baseline=178.089\nwidth=2820\nheight=1920\n")
    calib = parse_calib(path)
    assert calib["focal"] == 2826.171 and calib["doffs"] == 123.77 and calib["baseline"] == 178.089


def test_middlebury_downscale_is_configurable_and_preserves_depth(tmp_path: Path) -> None:
    scene = tmp_path / "Scene"
    scene.mkdir()
    cv2.imwrite(str(scene / "im0.png"), np.zeros((40, 80, 3), np.uint8))
    disparity = np.full((40, 80), 100, dtype=np.float32)
    disparity[:4, :4] = np.inf
    (scene / "disp0.pfm").write_bytes(b"Pf\n80 40\n-1.0\n" + disparity[::-1].astype("<f4").tobytes())
    (scene / "calib.txt").write_text("cam0=[2000 0 1; 0 2000 1; 0 0 1]\ndoffs=100\nbaseline=100\nwidth=80\nheight=40\n")
    sample = Middlebury2014Dataset(tmp_path, downscale=4)[0]
    assert sample["depth"].shape == (1, 10, 20) and sample["rgb"].shape == (3, 10, 20)
    assert sample["depth"][sample["valid"]].numpy() == pytest.approx(1.0)
    assert not sample["valid"][0, 0, 0]


def test_middlebury_missing_file_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "Scene").mkdir()
    with pytest.raises(FileNotFoundError, match="im0.png"):
        Middlebury2014Dataset(tmp_path)


def test_kitti_missing_folder_skips_with_message(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert open_kitti(tmp_path / "missing") is None
    assert "Skipped" in capsys.readouterr().out


def test_kitti_loader_on_synthetic_files(tmp_path: Path) -> None:
    for folder in ("image", "velodyne_raw", "groundtruth_depth"):
        (tmp_path / folder).mkdir()
    cv2.imwrite(str(tmp_path / "image" / "d_sync_image_0005_image_02.png"), np.zeros((8, 16, 3), np.uint8))
    raw = np.zeros((8, 16), np.uint16)
    raw[0, 0] = 2560
    cv2.imwrite(str(tmp_path / "velodyne_raw" / "d_sync_velodyne_raw_0005_image_02.png"), raw)
    cv2.imwrite(str(tmp_path / "groundtruth_depth" / "d_sync_groundtruth_depth_0005_image_02.png"), raw)
    sample = KITTIDepthCompletionDataset(tmp_path)[0]
    assert sample["depth"][0, 0, 0] == pytest.approx(10.0)
    assert sample["valid"].sum() == 1


def test_fill_holes_uses_nearest_valid_value() -> None:
    depth = np.array([[1, 0, 0, 5]], dtype=np.float32)
    assert fill_holes(depth, depth > 0).tolist() == [[1, 1, 5, 5]]


def test_fill_holes_rejects_empty_mask() -> None:
    with pytest.raises(ValueError):
        fill_holes(np.zeros((2, 2)), np.zeros((2, 2), dtype=bool))


def _real_like_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    guide = cv2.GaussianBlur(rng.random((96, 128, 3), dtype=np.float32), (0, 0), 2)
    guide = (guide - guide.min()) / (guide.max() - guide.min())
    depth = 1 + 3 * guide[..., 0] + (np.arange(128, dtype=np.float32) > 64)
    return depth.astype(np.float32), guide


@pytest.mark.skipif(not hasattr(cv2, "ximgproc"), reason="ximgproc not available")
@pytest.mark.parametrize("epsilon", [1e-4, 1e-3, 1e-2])
def test_guided_filter_backends_agree(epsilon: float) -> None:
    """Regression test: small epsilon used to diverge when ximgproc got a float guide."""
    depth, guide = _real_like_inputs()
    guide = np.round(guide * 255).astype(np.float32) / 255  # 8-bit values, as real images have
    numpy_result = guided_filter_numpy(depth, guide, 8, epsilon)
    assert np.abs(numpy_result - guided_filter_ximgproc(depth, guide, 8, epsilon)).max() < 0.01


@pytest.mark.skipif(not hasattr(cv2, "ximgproc"), reason="ximgproc not available")
def test_numpy_joint_bilateral_matches_ximgproc() -> None:
    depth, guide = _real_like_inputs()
    expected = cv2.ximgproc.jointBilateralFilter(guide, depth, 9, 0.1, 2.0)
    assert np.abs(joint_bilateral_filter_numpy(depth, guide, 4, 2.0, 0.1) - expected)[4:-4, 4:-4].max() < 0.02


def test_guided_upsample_numpy_backend_runs() -> None:
    depth, guide = _real_like_inputs()
    out = guided_upsample(depth[::2, ::2], guide, 2, use_ximgproc=False)
    assert out.shape == depth.shape


def test_metrics_use_valid_pixels_only() -> None:
    target = np.tile(np.linspace(1, 5, 32, dtype=np.float32), (32, 1))
    prediction = target + 0.5
    mask = np.ones_like(target, dtype=bool)
    mask[:, :8] = False
    corrupted = prediction.copy()
    corrupted[:, :8] = 1000
    assert rmse(corrupted, target, mask) == pytest.approx(0.5)
    assert mae(corrupted, target, mask) == pytest.approx(0.5)
    assert psnr(corrupted, target, mask) == pytest.approx(psnr(prediction, target, mask))
    assert ssim(corrupted, target, mask) == pytest.approx(ssim(prediction, target, mask))
    assert psnr(corrupted, target, mask, data_range=1.0) == pytest.approx(10 * np.log10(1 / 0.25))


def test_perfect_prediction_scores() -> None:
    target = np.tile(np.linspace(1, 5, 32, dtype=np.float32), (32, 1))
    mask = np.ones_like(target, dtype=bool)
    assert rmse(target, target, mask) == 0
    assert psnr(target, target, mask) == float("inf")
    assert ssim(target, target, mask) == pytest.approx(1.0)


def test_boundary_f_score_tolerance() -> None:
    target = np.full((40, 40), 2.0, dtype=np.float32)
    target[:, 20:] = 4
    mask = np.ones_like(target, dtype=bool)
    assert boundary_f_score(target, target, mask) == pytest.approx(1.0)
    def edge_at(column: int) -> np.ndarray:
        result = np.full((40, 40), 2.0, dtype=np.float32)
        result[:, column:] = 4
        return result
    assert boundary_f_score(edge_at(22), target, mask, tolerance_px=2) == pytest.approx(1.0)
    assert boundary_f_score(edge_at(23), target, mask, tolerance_px=2) == 0.0
    assert boundary_f_score(np.full((40, 40), 3.0, dtype=np.float32), target, mask) == 0.0
    assert np.isnan(boundary_f_score(target, np.full((40, 40), 2.0, dtype=np.float32), mask))


@needs_nyu
def test_training_samples_do_not_read_raw_depth() -> None:
    assert "measured" not in build_nyu_dataset(CONFIG, "train")[0]


@needs_nyu
def test_measured_mask_is_a_subset_of_valid() -> None:
    sample = build_nyu_dataset(CONFIG, "train", include_measured=True)[0]
    assert sample["measured"].shape == sample["valid"].shape
    assert not (sample["measured"] & ~sample["valid"]).any()
    assert sample["measured"].sum() < sample["valid"].sum()


def _eval_sample(with_measured: bool) -> dict:
    depth = np.tile(np.linspace(1, 5, 64, dtype=np.float32), (64, 1))
    sample = {
        "rgb": torch.rand(3, 64, 64),
        "depth": torch.from_numpy(depth)[None],
        "valid": torch.ones(1, 64, 64, dtype=torch.bool),
        "name": "x",
    }
    if with_measured:
        measured = torch.zeros(1, 64, 64, dtype=torch.bool)
        measured[:, :, :32] = True
        sample["measured"] = measured
    return sample


def _evaluate(samples: list[dict], tmp_path: Path) -> pd.DataFrame:
    config = {**CONFIG, "evaluation": {**CONFIG["evaluation"], "figures_dir": str(tmp_path)}}
    rows = evaluate_dataset("d", "test", samples, get_methods(config, "classical"), ["lr"], None, 0, config)
    return pd.DataFrame(rows)


def test_evaluate_scores_each_prediction_on_both_masks(tmp_path: Path) -> None:
    per_image = _evaluate([_eval_sample(True)], tmp_path)
    summary = summarise(per_image)
    assert set(summary["mask"]) == {"all", "measured"}
    assert len(summary) == 2 * 5 * 3
    assert list(summary.columns[:6]) == ["dataset", "split", "scale", "corruption", "mask", "method"]
    both = per_image.pivot_table(index=["scale", "method"], columns="mask", values="rmse_m")
    assert (both["all"] != both["measured"]).any()


def test_evaluate_without_measured_reports_all_only(tmp_path: Path) -> None:
    assert set(_evaluate([_eval_sample(False)], tmp_path)["mask"]) == {"all"}


def test_guided_upsample_accepts_fractional_radius_per_scale() -> None:
    depth, guide = _real_like_inputs()
    assert guided_upsample(depth[::2, ::2], guide, 2, radius_per_scale=0.5).shape == depth.shape


def test_config_update_keeps_comments_and_rejects_ambiguous_keys() -> None:
    from scripts.tune_classical import replace_config_value

    text = "a:\n  epsilon: 0.001   # note\n  other: 1\n"
    assert replace_config_value(text, "epsilon", 0.01) == "a:\n  epsilon: 0.01   # note\n  other: 1\n"
    with pytest.raises(ValueError):
        replace_config_value(text + "b:\n  epsilon: 2\n", "epsilon", 0.01)


def test_boundary_labels_match_numpy_edges() -> None:
    from src.training.losses import boundary_labels
    from src.utils import depth_edges

    rng = np.random.default_rng(0)
    depth = (1 + rng.random((32, 48))).astype(np.float32)
    mask = rng.random((32, 48)) > 0.1
    labels = boundary_labels(torch.from_numpy(depth)[None, None], torch.from_numpy(mask)[None, None], 0.05)
    assert np.array_equal(labels[0, 0].numpy(), depth_edges(depth, mask, 0.05))


def test_losses_ignore_invalid_pixels_and_log_each_term() -> None:
    from src.models.depthguard_net import DepthGuardOutput
    from src.training.losses import DepthGuardLoss, laplace_nll_loss

    target = torch.full((1, 1, 16, 16), 2.0)
    target[..., 8:] = 4.0
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., :4] = False
    prediction = target + 0.1
    log_variance = torch.ones_like(target)
    corrupted = prediction.clone()
    corrupted[~mask] = 100.0
    assert laplace_nll_loss(corrupted, target, log_variance, mask, 0.02) == pytest.approx(
        laplace_nll_loss(prediction, target, log_variance, mask, 0.02).item()
    )
    output = DepthGuardOutput(corrupted, log_variance, torch.exp(-log_variance), torch.zeros_like(target))
    terms = DepthGuardLoss({"depth": 1.0, "gradient": 0.5, "boundary": 0.2}, 0.02, 0.05)(output, target, mask)
    assert set(terms) == {"depth", "gradient", "boundary", "total"}
    expected = terms["depth"] + 0.5 * terms["gradient"] + 0.2 * terms["boundary"]
    assert terms["total"].item() == pytest.approx(expected.item())


def test_laplace_nll_rewards_matching_uncertainty_to_error() -> None:
    from src.training.losses import laplace_nll_loss

    target = torch.zeros(1, 1, 4, 4)
    prediction = target + 0.2  # error is 10x the reference
    mask = torch.ones_like(target, dtype=torch.bool)
    losses = {s: laplace_nll_loss(prediction, target, torch.full_like(target, s), mask, 0.02).item()
              for s in (0.0, float(np.log(10)), 5.0)}
    assert losses[float(np.log(10))] < losses[0.0] and losses[float(np.log(10))] < losses[5.0]


def test_scale_batch_sampler_is_seeded_and_uses_one_scale_per_batch() -> None:
    from src.training.train import RandomScaleBatchSampler

    sampler = RandomScaleBatchSampler(list(range(40)), 8, [2, 4, 8], seed=1)
    sampler.set_epoch(3)
    first = list(sampler)
    assert first == list(sampler) and len(first) == 5
    assert all(len({item[1] for item in batch}) == 1 for batch in first)
    assert len({batch[0][1] for batch in first}) >= 1
    sampler.set_epoch(4)
    assert first != list(sampler)


def test_network_dataset_crops_are_paired_and_seeded() -> None:
    from src.datasets.corruption import NetworkDataset

    depth = np.tile(np.linspace(1, 5, 128, dtype=np.float32), (128, 1))
    base = [{"rgb": torch.from_numpy(np.stack([depth / 5] * 3)), "depth": torch.from_numpy(depth)[None],
             "valid": torch.ones(1, 128, 128, dtype=torch.bool), "name": "x"}]
    data = NetworkDataset(base, list(MODES), PARAMS, seed=3, crop_size_hw=(64, 64), flip=True)
    a, b = data[(0, 4, 1, None)], data[(0, 4, 1, None)]
    assert torch.equal(a["depth_input"], b["depth_input"]) and a["label"] == b["label"]
    assert a["depth_input"].shape == a["target"].shape == (1, 64, 64)
    assert torch.allclose(a["rgb"][0] * 5, a["target"][0], atol=1e-4)


def test_training_only_builds_train_and_val(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training import train as train_module

    requested = []
    def fake_build(config, split, allow_test=False, include_measured=False):
        requested.append((split, allow_test))
        return [None] * 100
    monkeypatch.setattr(train_module, "build_nyu_dataset", fake_build)
    train_module.build_dataloaders({**CONFIG, "train": {**CONFIG["train"], "num_workers": 0}}, smoke=True)
    assert sorted(requested) == [("train", False), ("val", False)]


def _hybrid_settings():
    from src.hybrid.refine import HybridSettings

    return HybridSettings(boundary_band_px=2, boundary_sigma_spatial=2.0, boundary_sigma_range=10.0)


def test_confidence_fusion_blends_only_low_confidence() -> None:
    from src.hybrid.refine import confidence_fusion

    net, guided = np.full((1, 3), 2.0), np.full((1, 3), 4.0)
    confidence = np.array([[0.25, 0.49, 0.9]])
    result = confidence_fusion(net, guided, confidence, 0.5)
    assert result[0, 0] == pytest.approx(0.25 * 2 + 0.75 * 4)
    assert result[0, 2] == 2.0


def test_keep_measured_requires_high_confidence_and_valid_input() -> None:
    from src.hybrid.refine import keep_measured_depth

    depth, measured = np.full((1, 3), 2.0), np.full((1, 3), 9.0)
    valid = np.array([[True, False, True]])
    confidence = np.array([[0.9, 0.9, 0.1]])
    assert keep_measured_depth(depth, measured, valid, confidence, 0.8).tolist() == [[9.0, 2.0, 2.0]]


def test_boundary_refine_never_averages_across_the_boundary() -> None:
    from src.hybrid.refine import boundary_aware_refine

    depth = np.full((32, 32), 2.0, dtype=np.float32)
    depth[:, 16:] = 6.0
    depth += np.random.default_rng(0).normal(0, 0.01, depth.shape).astype(np.float32)
    boundary = np.zeros_like(depth)
    boundary[:, 15:17] = 0.9
    rgb = np.zeros((32, 32, 3), dtype=np.float32)
    refined = boundary_aware_refine(depth, boundary, rgb, _hybrid_settings())
    # A filter that crossed the edge would pull values toward the middle of 2 and 6.
    assert refined.min() > 1.9 and refined.max() < 6.1
    assert not ((refined > 2.5) & (refined < 5.5)).any()
    # Outside the band nothing changes; inside it, noise is reduced.
    assert np.array_equal(refined[:, :10], depth[:, :10])
    assert refined[:, 13:15].std() < depth[:, 13:15].std()


def test_boundary_refine_without_boundaries_changes_nothing() -> None:
    from src.hybrid.refine import boundary_aware_refine

    depth = np.random.default_rng(0).random((16, 16), dtype=np.float32) + 1
    refined = boundary_aware_refine(depth, np.zeros_like(depth), np.zeros((16, 16, 3), np.float32), _hybrid_settings())
    assert np.array_equal(refined, depth)


def test_final_hybrid_flags_come_from_config_with_fusion_off() -> None:
    from src.hybrid.refine import HybridFlags, HybridSettings

    assert HybridFlags.from_config(CONFIG) == HybridFlags(fusion=False)
    assert HybridSettings.from_config(CONFIG).low_confidence == CONFIG["hybrid"]["low_confidence"]


def test_hybrid_flags_turn_off_each_part() -> None:
    from src.hybrid.refine import HybridFlags, hybrid_refine

    rng = np.random.default_rng(1)
    net = (2 + rng.random((32, 32))).astype(np.float32)
    inp = net + 1
    guided = net + 2
    valid = np.ones_like(net, dtype=bool)
    confidence = np.full_like(net, 0.3)
    confidence[:, 16:] = 0.95
    boundary = np.zeros_like(net)
    boundary[8:10, :] = 0.9
    rgb = np.zeros((32, 32, 3), dtype=np.float32)
    args = (net, inp, valid, guided, confidence, boundary, rgb, _hybrid_settings())
    off = HybridFlags(fusion=False, keep_measured=False, boundary_refine=False)
    assert np.array_equal(hybrid_refine(*args, off), net)
    fusion_only = hybrid_refine(*args, HybridFlags(keep_measured=False, boundary_refine=False))
    assert np.allclose(fusion_only[:, :16], 0.3 * net[:, :16] + 0.7 * guided[:, :16])
    keep_only = hybrid_refine(*args, HybridFlags(fusion=False, boundary_refine=False))
    assert np.array_equal(keep_only[:, 16:], inp[:, 16:]) and np.array_equal(keep_only[:, :16], net[:, :16])
    full = hybrid_refine(*args)
    without_boundary = hybrid_refine(*args, HybridFlags(boundary_refine=False))
    assert not np.array_equal(full, without_boundary)


def test_evaluate_runs_network_methods_from_a_checkpoint(tmp_path: Path) -> None:
    from src.evaluation.evaluate import NetworkRunner, evaluate_dataset, get_methods
    from src.models.depthguard_net import DepthGuardNet

    path = tmp_path / "best.pt"
    torch.save({"model": DepthGuardNet().state_dict(), "epoch": 1, "config": CONFIG}, path)
    methods = get_methods(CONFIG, "ablation", path, device=torch.device("cpu"))
    assert list(methods) == ["dl", "hybrid", "hybrid_with_fusion", "hybrid_no_keep", "hybrid_no_boundary"]
    rows = evaluate_dataset("d", "val", [_eval_sample(True)], methods, ["lr"], None, 0, CONFIG, tmp_path)
    frame = pd.DataFrame(rows)
    assert len(frame) == 5 * 3 * 2 and frame["rmse_m"].notna().all()
    # An untrained network returns its bicubic input, so dl must match bicubic exactly.
    bicubic = get_methods(CONFIG, "classical")["bicubic"]
    depth = _eval_sample(False)["depth"][0].numpy()
    lr, valid = downsample_area(depth, depth > 0, 4)
    runner = NetworkRunner(path, torch.device("cpu"))
    output = runner.predict(lr, valid, np.zeros((64, 64, 3), np.float32), 4)
    assert np.allclose(output["depth"], bicubic(lr, valid, None, 4), atol=1e-5)


def test_missing_checkpoint_fails_clearly(tmp_path: Path) -> None:
    from src.evaluation.evaluate import get_methods

    with pytest.raises(FileNotFoundError, match="Finish training"):
        get_methods(CONFIG, "dl", tmp_path / "none.pt")


def test_append_results_keeps_existing_rows_and_refuses_duplicates(tmp_path: Path) -> None:
    from src.evaluation.evaluate import append_results

    path = tmp_path / "results.csv"
    path.write_text("method,mode,value\nbicubic,lr,0.123456789\n")
    append_results(pd.DataFrame({"method": ["dl"], "mode": ["noisy"], "value": [0.5]}), path, ["method", "mode"])
    assert path.read_text() == "method,mode,value\nbicubic,lr,0.123456789\ndl,noisy,0.50000\n"
    with pytest.raises(ValueError, match="already has"):
        append_results(pd.DataFrame({"method": ["dl"], "mode": ["noisy"], "value": [0.6]}), path, ["method", "mode"])
    with pytest.raises(ValueError, match="columns"):
        append_results(pd.DataFrame({"method": ["x"], "other": ["y"], "value": [1.0]}), path, ["method"])


def test_output_tag_names_the_files() -> None:
    from src.evaluation.evaluate import output_paths

    assert output_paths(CONFIG, "ablation", "test", None, "ablation_test_final")["results_csv"].name == "results_ablation_test_final.csv"


def test_only_full_classical_test_run_writes_primary_results() -> None:
    from src.evaluation.evaluate import output_paths

    assert output_paths(CONFIG, "classical", "test", None)["results_csv"].name == "results.csv"
    for args in (("classical", "test", 20), ("classical", "val", None), ("all", "test", None), ("ablation", "val", None)):
        assert output_paths(CONFIG, *args)["results_csv"].name != "results.csv"
