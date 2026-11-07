"""Run baselines at every scale factor on NYU and Middlebury, and write CSVs and figures.

Usage, from the project root:

    python src/evaluation/evaluate.py --methods classical --split test [--limit 20]

``--methods`` is one of ``classical``, ``dl``, ``hybrid``, ``final``, ``ablation`` or ``all`` (classical,
dl and hybrid). The network methods need a checkpoint, by default ``checkpoints/best.pt``.
Only a full classical test run writes ``outputs/results.csv``. Other runs add their
methods, split and limit to the file names, so they cannot overwrite it.

``--limit`` caps the images per dataset, chosen evenly across it. Middlebury has no
splits, so it only runs with ``--split test``; this keeps it out of any tuning on val.

The ``mask`` column says which pixels were scored against the same ground truth:

    all       every valid ground truth pixel. Primary, and comparable to the literature.
    measured  NYU only: pixels the sensor measured (rawDepths > 0), excluding inpainted ones.

Middlebury has no inpainted depth, so it only reports ``all``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classical.guided_filter import guided_upsample  # noqa: E402
from src.classical.interpolation import (  # noqa: E402
    prepare_network_input,
    upsample_bicubic,
    upsample_bilinear,
    upsample_nearest,
)
from src.classical.joint_bilateral import HAS_XIMGPROC, joint_bilateral_upsample  # noqa: E402
from src.datasets.corruption import MODES, corrupt_depth, get_corruption_params, make_rng  # noqa: E402
from src.datasets.middlebury import build_middlebury_dataset  # noqa: E402
from src.datasets.nyu import build_nyu_dataset  # noqa: E402
from src.evaluation.metrics import boundary_f_score, mae, psnr, rmse, ssim  # noqa: E402
from src.hybrid.refine import HybridFlags, HybridSettings, hybrid_refine  # noqa: E402
from src.models.depthguard_net import DepthGuardNet  # noqa: E402
from src.utils import get_device, load_config, mod_crop, resolve_project_path  # noqa: E402

METRIC_COLUMNS = ["rmse_m", "mae_m", "psnr_db", "ssim", "boundary_f", "runtime_ms"]
TILE_WIDTH = 360

# A method takes (lr_depth, lr_valid, rgb_hr, scale) and returns depth at the RGB resolution.
Method = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]
RESULT_KEYS = ["dataset", "split", "scale", "corruption", "mask", "method"]


class NetworkRunner:
    """Load a checkpoint and run the network on one image at a time."""

    def __init__(self, checkpoint_path: str | Path, device: torch.device | None = None) -> None:
        """Load the weights and model settings saved in the checkpoint."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}. Finish training first.")
        self.device = device or get_device()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        settings = checkpoint["config"]["model"]
        self.model = DepthGuardNet(settings["channels"], settings["blocks_per_level"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.epoch = checkpoint["epoch"]

    @torch.no_grad()
    def predict(self, lr_depth: np.ndarray, lr_valid: np.ndarray, rgb: np.ndarray, scale: int) -> dict[str, np.ndarray]:
        """Return network depth, confidence and boundary probability, plus the network input."""
        input_depth, input_valid = prepare_network_input(lr_depth, lr_valid, scale)
        output = self.model(
            torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(self.device),
            torch.from_numpy(input_depth)[None, None].to(self.device),
            torch.from_numpy(input_valid)[None, None].to(self.device),
        )
        return {
            "depth": output.depth[0, 0].cpu().numpy(),
            "confidence": output.confidence[0, 0].cpu().numpy(),
            "boundary_probability": torch.sigmoid(output.boundary_logits)[0, 0].cpu().numpy(),
            "input_depth": input_depth,
            "input_valid": input_valid,
        }


def get_methods(
    config: dict[str, Any],
    group: str,
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Method]:
    """Return the methods for a group, keyed by name. Network methods load the checkpoint."""
    groups = config["evaluation"]["methods"]
    names = (
        [*groups["classical"], *groups["dl"], *groups["hybrid"]] if group == "all" else groups[group]
    )
    jbf = config["classical"]["joint_bilateral"]
    guided = config["classical"]["guided_filter"]

    def guided_depth(d: np.ndarray, v: np.ndarray, g: np.ndarray, s: int) -> np.ndarray:
        return guided_upsample(d, g, s, guided["radius_per_scale"], guided["epsilon"], v)

    available: dict[str, Method] = {
        "nearest": lambda d, v, g, s: upsample_nearest(d, s, v),
        "bilinear": lambda d, v, g, s: upsample_bilinear(d, s, v),
        "bicubic": lambda d, v, g, s: upsample_bicubic(d, s, v),
        "joint_bilateral": lambda d, v, g, s: joint_bilateral_upsample(
            d, g, s, jbf["sigma_spatial_per_scale"], jbf["sigma_range"], v
        ),
        "guided_filter": guided_depth,
    }
    if any(name not in available for name in names):
        runner = NetworkRunner(checkpoint_path or resolve_project_path(config["paths"]["checkpoints_dir"]) / "best.pt", device)
        print(f"Loaded checkpoint from epoch {runner.epoch}")
        settings = HybridSettings.from_config(config)
        final_flags = HybridFlags.from_config(config)

        def hybrid(flags: HybridFlags) -> Method:
            def run(d: np.ndarray, v: np.ndarray, g: np.ndarray, s: int) -> np.ndarray:
                net = runner.predict(d, v, g, s)
                return hybrid_refine(
                    net["depth"], net["input_depth"], net["input_valid"],
                    guided_depth(d, v, g, s) if flags.fusion else None,
                    net["confidence"], net["boundary_probability"], g, settings, flags,
                )
            return run

        available.update({
            "dl": lambda d, v, g, s: runner.predict(d, v, g, s)["depth"],
            "hybrid": hybrid(final_flags),
            "hybrid_with_fusion": hybrid(replace(final_flags, fusion=True)),
            "hybrid_no_keep": hybrid(replace(final_flags, keep_measured=False)),
            "hybrid_no_boundary": hybrid(replace(final_flags, boundary_refine=False)),
        })
    return {name: available[name] for name in names}


def output_paths(
    config: dict[str, Any], methods: str, split: str, limit: int | None, tag: str | None = None
) -> dict[str, Path]:
    """Return output locations. Only a full classical test run uses the configured names."""
    settings = config["evaluation"]
    if methods == "classical" and split == "test" and limit is None and tag is None:
        return {key: resolve_project_path(settings[key]) for key in ("results_csv", "per_image_csv", "figures_dir")}
    tag = tag or f"{methods}_{split}" + (f"_limit{limit}" if limit is not None else "")
    outputs = resolve_project_path(config["paths"]["outputs_dir"])
    return {
        "results_csv": outputs / f"results_{tag}.csv",
        "per_image_csv": outputs / f"results_{tag}_per_image.csv",
        "figures_dir": outputs / f"figures_{tag}",
    }


def select_indices(length: int, limit: int | None) -> list[int]:
    """Return all indices, or ``limit`` evenly spaced ones."""
    if limit is None or limit >= length:
        return list(range(length))
    return np.linspace(0, length - 1, limit).round().astype(int).tolist()


def score(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, config: dict[str, Any]
) -> dict[str, float]:
    """Return all quality metrics for one prediction."""
    settings = config["evaluation"]
    return {
        "rmse_m": rmse(prediction, target, mask),
        "mae_m": mae(prediction, target, mask),
        "psnr_db": psnr(prediction, target, mask),
        "ssim": ssim(prediction, target, mask),
        "boundary_f": boundary_f_score(
            prediction, target, mask, settings["boundary_tolerance_px"], settings["boundary_edge_threshold"]
        ),
    }


def colourise(depth: np.ndarray, valid: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map depth to a BGR colour image over a fixed range, with invalid pixels in black."""
    scaled = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    colour = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    return colour


def make_tile(image: np.ndarray, caption: str) -> np.ndarray:
    """Resize a BGR image to the tile width and add a caption bar above it."""
    height = round(image.shape[0] * TILE_WIDTH / image.shape[1])
    tile = cv2.resize(image, (TILE_WIDTH, height), interpolation=cv2.INTER_NEAREST)
    bar = np.full((24, TILE_WIDTH, 3), 255, dtype=np.uint8)
    cv2.putText(bar, caption, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def save_figure(
    path: Path,
    rgb: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    lr_depth: np.ndarray,
    lr_valid: np.ndarray,
    predictions: dict[str, np.ndarray],
    rmses: dict[str, float],
) -> None:
    """Save a side-by-side PNG, drawing all depth maps on the ground truth's colour range."""
    low, high = float(target[mask].min()), float(target[mask].max())
    full_valid = np.ones_like(mask)
    tiles = [
        make_tile((rgb[:, :, ::-1] * 255).astype(np.uint8), "RGB"),
        make_tile(colourise(lr_depth, lr_valid, low, high), f"Input {lr_depth.shape[1]}x{lr_depth.shape[0]}"),
        make_tile(colourise(target, mask, low, high), f"Ground truth ({low:.2f}-{high:.2f} m)"),
    ]
    tiles += [
        make_tile(colourise(pred, full_valid, low, high), f"{name}  RMSE {rmses[name]:.3f} m")
        for name, pred in predictions.items()
    ]
    columns = 4
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.vstack(rows)):
        raise OSError(f"Could not write {path}")


def warn_empty(mask_name: str, image_name: str) -> bool:
    """Report that an image has no pixels in a mask, and return False to drop that mask."""
    print(f"  WARNING: {image_name} has no pixels in mask {mask_name!r}; not scored on it.", flush=True)
    return False


def evaluate_dataset(
    dataset_name: str,
    split: str,
    dataset: Any,
    methods: dict[str, Method],
    corruptions: list[str],
    limit: int | None,
    num_figures: int,
    config: dict[str, Any],
    figures_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all methods, scales and corruptions, and return one row per run.

    Images are mod-cropped to the largest scale, so every scale is scored on the same pixels.
    Each prediction is scored once per mask available for the sample.
    """
    scales = config["scale_factors"]
    largest = max(scales)
    params = get_corruption_params(config)
    seed = config["corruption"]["seed"]
    figures_dir = figures_dir or resolve_project_path(config["evaluation"]["figures_dir"])
    indices = select_indices(len(dataset), limit)
    figure_indices = set(indices[:: max(1, len(indices) // num_figures)][:num_figures]) if num_figures else set()
    rows = []
    for position, index in enumerate(indices, 1):
        sample = dataset[index]
        rgb = mod_crop(sample["rgb"].numpy().transpose(1, 2, 0), largest).copy()
        target = mod_crop(sample["depth"][0].numpy(), largest)
        mask = mod_crop(sample["valid"][0].numpy(), largest)
        masks = {"all": mask}
        if "measured" in sample:
            masks["measured"] = mod_crop(sample["measured"][0].numpy(), largest)
        masks = {key: value for key, value in masks.items() if value.any() or warn_empty(key, sample["name"])}
        for scale in scales:
            for mode in corruptions:
                rng = make_rng(seed, index, scale, mode)
                corrupted = corrupt_depth(target, mask, scale, mode, params, rng)
                predictions, rmses = {}, {}
                for name, method in methods.items():
                    start = time.perf_counter()
                    prediction = method(corrupted.lr_depth, corrupted.validity_mask, rgb, scale)
                    runtime_ms = (time.perf_counter() - start) * 1000
                    if prediction.shape != target.shape:
                        raise ValueError(f"{name}: shape {prediction.shape}, expected {target.shape}")
                    if not np.isfinite(prediction).all():
                        raise ValueError(f"{name}: non-finite output on {sample['name']}")
                    predictions[name] = prediction
                    for mask_name, scored_mask in masks.items():
                        metrics = score(prediction, target, scored_mask, config)
                        if mask_name == "all":
                            rmses[name] = metrics["rmse_m"]
                        rows.append({
                            "dataset": dataset_name, "split": split, "image": sample["name"],
                            "scale": scale, "corruption": corrupted.label, "mask": mask_name, "method": name,
                            **metrics, "runtime_ms": runtime_ms,
                        })
                if index in figure_indices:
                    save_figure(
                        figures_dir / f"{dataset_name}_{sample['name']}_x{scale}_{mode}.png",
                        rgb, target, mask, corrupted.lr_depth, corrupted.validity_mask, predictions, rmses,
                    )
        print(f"  [{dataset_name}] {position}/{len(indices)} {sample['name']}", flush=True)
    return rows


def summarise(per_image: pd.DataFrame) -> pd.DataFrame:
    """Average per-image metrics for each dataset, scale, corruption, mask and method."""
    grouped = per_image.groupby(RESULT_KEYS, sort=False)
    summary = grouped[METRIC_COLUMNS].mean()
    summary.insert(0, "n_images", grouped["image"].nunique())
    return summary.reset_index().sort_values(["dataset", "corruption", "mask", "scale"], kind="stable")


def write_results(results: pd.DataFrame, csv_path: str | Path) -> None:
    """Write results to CSV, creating the parent folder if needed."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False, float_format="%.5f")


def append_results(results: pd.DataFrame, csv_path: str | Path, keys: list[str]) -> None:
    """Append rows to a CSV without rewriting its existing rows. Refuse columns or runs that clash."""
    path = Path(csv_path)
    existing = pd.read_csv(path)
    if list(existing.columns) != list(results.columns):
        raise ValueError(f"{path}: columns {list(existing.columns)} differ from {list(results.columns)}")
    overlap = existing[keys].astype(str).merge(results[keys].astype(str).drop_duplicates(), on=keys)
    if len(overlap):
        raise ValueError(f"{path} already has {len(overlap)} of these rows, for example {overlap.iloc[0].to_dict()}")
    results.to_csv(path, mode="a", header=False, index=False, float_format="%.5f")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", choices=["classical", "dl", "hybrid", "final", "ablation", "all"], default="classical")
    parser.add_argument("--checkpoint", default=None, help="defaults to checkpoints/best.pt")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--limit", type=int, default=None, help="max images per dataset")
    parser.add_argument("--corruptions", nargs="+", choices=MODES, default=["lr"])
    parser.add_argument("--num-figures", type=int, default=3, help="images per dataset to draw")
    parser.add_argument("--tag", default=None, help="name outputs results_<tag>.csv, per-image and figures")
    parser.add_argument("--append", action="store_true", help="append to the main results CSVs; full test runs only")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    if args.append and (args.split != "test" or args.limit is not None or args.methods == "classical"):
        parser.error("--append needs a full test run of network methods")
    return args


def main() -> None:
    """Run the evaluation and print the summary table."""
    args = parse_args()
    config = load_config(args.config)
    methods = get_methods(config, args.methods, args.checkpoint)
    paths = output_paths(config, args.methods, args.split, args.limit, args.tag)
    print(f"ximgproc backend: {HAS_XIMGPROC} | scales: {config['scale_factors']} | "
          f"corruptions: {args.corruptions} | severity: {config['corruption']['severity']} | limit: {args.limit}")

    nyu = build_nyu_dataset(config, args.split, allow_test=args.split == "test", include_measured=True)
    common = (methods, args.corruptions, args.limit, args.num_figures, config, paths["figures_dir"])
    rows = evaluate_dataset("nyu", args.split, nyu, *common)
    if args.split == "test":
        rows += evaluate_dataset("middlebury", "all", build_middlebury_dataset(config), *common)

    per_image = pd.DataFrame(rows)
    summary = summarise(per_image)
    if args.append:
        settings = config["evaluation"]
        paths["results_csv"] = resolve_project_path(settings["results_csv"])
        paths["per_image_csv"] = resolve_project_path(settings["per_image_csv"])
        append_results(summary, paths["results_csv"], RESULT_KEYS)
        append_results(per_image, paths["per_image_csv"], RESULT_KEYS)
    else:
        write_results(per_image, paths["per_image_csv"])
        write_results(summary, paths["results_csv"])
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote {paths['results_csv']}, {paths['per_image_csv']} and figures to {paths['figures_dir']}")


if __name__ == "__main__":
    main()
