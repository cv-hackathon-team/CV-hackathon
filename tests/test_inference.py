"""Tests for src/inference.py, using an untrained network on the CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import src.inference as inference
from src.hybrid.diagnose import Diagnosis, apply_rules, diagnose, DiagnosisSettings, estimate_noise_level
from src.models.depthguard_net import DepthGuardNet
from src.utils import load_config

CONFIG = load_config()


@pytest.fixture
def checkpoint(tmp_path: Path) -> str:
    path = tmp_path / "best.pt"
    torch.manual_seed(0)
    torch.save({"model": DepthGuardNet().state_dict(), "epoch": 1, "config": CONFIG}, path)
    inference.get_predictor.cache_clear()
    yield str(path)
    inference.get_predictor.cache_clear()


def _inputs(height: int, width: int, scale: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    rgb = rng.random((height, width, 3), dtype=np.float32)
    lr = np.full((height // scale, width // scale), 2.0, dtype=np.float32)
    lr[:, lr.shape[1] // 2 :] = 3.0
    return rgb, lr


@pytest.mark.parametrize("scale", [2, 4, 8])
@pytest.mark.parametrize("size", [(64, 96), (120, 200), (136, 248), (488, 648)])
def test_predict_returns_rgb_size_for_sizes_needing_padding(checkpoint: str, size: tuple[int, int], scale: int) -> None:
    rgb, lr = _inputs(*size, scale)
    output = inference.predict(rgb, lr, scale, checkpoint, "cpu")
    assert set(output) == {"depth", "confidence", "boundary", "diagnosis", "timings"}
    for key in ("depth", "confidence", "boundary"):
        assert output[key].shape == size and output[key].dtype == np.float32 and np.isfinite(output[key]).all()
    assert output["confidence"].max() <= 1 and output["confidence"].min() > 0
    assert 0 <= output["boundary"].min() and output["boundary"].max() <= 1
    assert output["diagnosis"]["label"] == "clean"
    assert output["timings"]["total_ms"] >= output["timings"]["network_ms"] > 0


def test_predict_handles_rgb_not_a_multiple_of_scale(checkpoint: str) -> None:
    rgb, lr = _inputs(483, 645, 4)
    assert lr.shape == (120, 161)
    output = inference.predict(rgb, lr, 4, checkpoint, "cpu")
    assert output["depth"].shape == (483, 645)
    uint8 = inference.predict((rgb * 255).astype(np.uint8), lr, 4, checkpoint, "cpu")
    assert uint8["depth"].shape == (483, 645)


def test_predict_treats_nan_and_zero_as_missing(checkpoint: str) -> None:
    rgb, lr = _inputs(64, 96, 4)
    lr[2:6, 2:6] = np.nan
    lr[0, 0] = 0
    output = inference.predict(rgb, lr, 4, checkpoint, "cpu")
    assert np.isfinite(output["depth"]).all() and output["diagnosis"]["sparsity"] > 0


def test_predict_rejects_bad_input(checkpoint: str) -> None:
    rgb, lr = _inputs(64, 96, 4)
    with pytest.raises(ValueError, match="scale"):
        inference.predict(rgb, lr, 3, checkpoint, "cpu")
    with pytest.raises(ValueError, match="does not match"):
        inference.predict(rgb, lr, 2, checkpoint, "cpu")
    with pytest.raises(ValueError, match="no valid"):
        inference.predict(rgb, np.zeros_like(lr), 4, checkpoint, "cpu")


def test_checkpoint_is_loaded_once(checkpoint: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real_load = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: calls.append(a) or real_load(*a, **k))
    rgb, lr = _inputs(64, 96, 4)
    inference.predict(rgb, lr, 4, checkpoint, "cpu")
    inference.predict(rgb, lr, 4, checkpoint, "cpu")
    assert len(calls) == 1


def test_accelerator_failure_falls_back_to_cpu(checkpoint: str, monkeypatch: pytest.MonkeyPatch) -> None:
    predictor = inference.get_predictor(checkpoint, "cpu")
    real_forward = predictor.runner.forward
    predictor.runner.device = torch.device("mps")
    predictor.runner.to_cpu = lambda: setattr(predictor.runner, "device", torch.device("cpu"))

    def forward(*args: object) -> dict[str, np.ndarray]:
        if predictor.runner.device.type != "cpu":
            raise RuntimeError("out of memory")
        return real_forward(*args)

    monkeypatch.setattr(predictor.runner, "forward", forward)
    rgb, lr = _inputs(64, 96, 4)
    assert predictor.predict(rgb, lr, 4)["depth"].shape == (64, 96)
    assert predictor.runner.device.type == "cpu"


def test_device_helper_chooses_cpu_without_accelerators(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.utils import get_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_device().type == "cpu"


def test_multiple_of_16_input_matches_evaluation_method(checkpoint: str) -> None:
    from src.evaluation.evaluate import get_methods

    rgb, lr = _inputs(64, 96, 4)
    method = get_methods(CONFIG, "hybrid", checkpoint, device=torch.device("cpu"))["hybrid"]
    expected = method(lr, lr > 0, rgb, 4)
    assert np.allclose(inference.predict(rgb, lr, 4, checkpoint, "cpu")["depth"], expected, atol=1e-6)


def test_diagnosis_labels_follow_the_rules() -> None:
    settings = DiagnosisSettings()
    assert apply_rules(0, 0, 0, settings) == "clean"
    assert apply_rules(0, 0.02, 0, settings) == "noisy"
    assert apply_rules(9, 0, 10, settings) == "holes"
    assert apply_rules(1, 0, 10, settings) == "boundary"
    assert apply_rules(60, 0, 75, settings) == "sparse"
    assert apply_rules(60, 0.02, 75, settings) == "mixed"


def test_noise_estimate_recovers_known_noise_and_ignores_an_edge() -> None:
    rng = np.random.default_rng(0)
    depth = np.full((200, 200), 2.0, dtype=np.float32)
    depth[:, 100:] = 4.0
    valid = np.ones_like(depth, dtype=bool)
    assert estimate_noise_level(depth, valid) == 0
    noisy = depth * (1 + rng.normal(0, 0.01, depth.shape)).astype(np.float32)
    assert estimate_noise_level(noisy, valid) == pytest.approx(0.01, rel=0.15)


def test_diagnose_with_no_measurable_pixels_is_safe() -> None:
    result = diagnose(np.zeros((8, 8), np.float32), np.zeros((8, 8), bool))
    assert isinstance(result, Diagnosis) and result.sparsity == 100 and result.noise_level == 0
