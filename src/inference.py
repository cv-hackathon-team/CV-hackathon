"""Single-image inference: depth, confidence, boundaries, diagnosis and timings."""

from __future__ import annotations

import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.classical.guided_filter import guided_upsample
from src.classical.interpolation import prepare_network_input
from src.hybrid.diagnose import DiagnosisSettings, diagnose
from src.hybrid.refine import HybridFlags, HybridSettings, hybrid_refine
from src.models.depthguard_net import DepthGuardNet
from src.utils import get_device, load_config, resolve_project_path

PAD_MULTIPLE = 16


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

    def to_cpu(self) -> None:
        """Move the network to the CPU."""
        self.device = torch.device("cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def forward(self, rgb: np.ndarray, input_depth: np.ndarray, input_valid: np.ndarray) -> dict[str, np.ndarray]:
        """Run the network on arrays whose size it accepts, and return arrays on the CPU."""
        output = self.model(
            torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(self.device),
            torch.from_numpy(input_depth)[None, None].to(self.device),
            torch.from_numpy(input_valid)[None, None].to(self.device),
        )
        return {
            "depth": output.depth[0, 0].cpu().numpy(),
            "confidence": output.confidence[0, 0].cpu().numpy(),
            "boundary_probability": torch.sigmoid(output.boundary_logits)[0, 0].cpu().numpy(),
        }

    def predict(self, lr_depth: np.ndarray, lr_valid: np.ndarray, rgb: np.ndarray, scale: int) -> dict[str, np.ndarray]:
        """Return network depth, confidence and boundary probability, plus the network input."""
        input_depth, input_valid = prepare_network_input(lr_depth, lr_valid, scale)
        return {**self.forward(rgb, input_depth, input_valid), "input_depth": input_depth, "input_valid": input_valid}


def _pad(array: np.ndarray, multiple: int = PAD_MULTIPLE) -> np.ndarray:
    """Repeat the bottom and right edges until height and width are divisible by ``multiple``."""
    height, width = array.shape[:2]
    pad = [(0, -height % multiple), (0, -width % multiple)] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, pad, mode="edge")


def _to_float_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return an RGB image as float32 in [0, 1], from uint8 or float input."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255
    return np.clip(rgb.astype(np.float32), 0, 1)


class Predictor:
    """The final hybrid pipeline, holding one loaded checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: torch.device | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Load the config and the checkpoint. The checkpoint defaults to ``checkpoints/best.pt``."""
        self.config = config or load_config()
        path = checkpoint_path or resolve_project_path(self.config["paths"]["checkpoints_dir"]) / "best.pt"
        self.runner = NetworkRunner(path, device)
        self.flags = HybridFlags.from_config(self.config)
        self.settings = HybridSettings.from_config(self.config)
        self.diagnosis_settings = DiagnosisSettings.from_config(self.config)

    def _run_network(self, rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
        """Run the network, and retry on the CPU if the accelerator fails."""
        try:
            return self.runner.forward(rgb, depth, valid)
        except (RuntimeError, NotImplementedError) as error:
            if self.runner.device.type == "cpu":
                raise
            print(f"WARNING: {self.runner.device} failed ({error}). Using the CPU.", flush=True)
            self.runner.to_cpu()
            return self.runner.forward(rgb, depth, valid)

    def predict(self, rgb: np.ndarray, lr_depth: np.ndarray, scale: int) -> dict[str, Any]:
        """Upsample ``lr_depth`` to the size of ``rgb``.

        Args:
            rgb: (H, W, 3) image, uint8 or float in [0, 1].
            lr_depth: (h, w) depth in metres. Zero, negative and non-finite pixels are missing.
            scale: one of the trained scale factors. (h * scale, w * scale) must be within one
                ``scale`` of (H, W); if it differs, processing runs at that size and the
                outputs are resized to (H, W).

        Returns:
            ``depth`` in metres, ``confidence`` in (0, 1], ``boundary`` probability in [0, 1],
            all (H, W) float32; ``diagnosis``, a dict of input statistics and a label;
            ``timings`` in milliseconds.
        """
        start = time.perf_counter()
        if scale not in self.config["scale_factors"]:
            raise ValueError(f"scale must be one of {self.config['scale_factors']}, got {scale}")
        if lr_depth.ndim != 2:
            raise ValueError(f"lr_depth must have shape (h, w), got {lr_depth.shape}")
        image = _to_float_rgb(rgb)
        output_size = image.shape[:2]
        working_size = (lr_depth.shape[0] * scale, lr_depth.shape[1] * scale)
        if any(abs(a - b) >= scale for a, b in zip(output_size, working_size)):
            raise ValueError(f"rgb size {output_size} does not match lr_depth {lr_depth.shape} at x{scale}")
        lr_valid = np.isfinite(lr_depth) & (lr_depth > 0)
        if not lr_valid.any():
            raise ValueError("lr_depth has no valid pixels")
        depth_lr = np.where(lr_valid, lr_depth, 0).astype(np.float32)
        if output_size != working_size:
            image = cv2.resize(image, working_size[::-1], interpolation=cv2.INTER_LINEAR)

        marks = {"start": start, "preprocess": time.perf_counter()}
        diagnosis = diagnose(depth_lr, lr_valid, self.diagnosis_settings)
        marks["diagnosis"] = time.perf_counter()

        input_depth, input_valid = prepare_network_input(depth_lr, lr_valid, scale)
        net = self._run_network(_pad(image), _pad(input_depth), _pad(input_valid))
        height, width = working_size
        network_depth, confidence, boundary = (
            net[key][:height, :width] for key in ("depth", "confidence", "boundary_probability")
        )
        marks["network"] = time.perf_counter()

        guided = None
        if self.flags.fusion:
            settings = self.config["classical"]["guided_filter"]
            guided = guided_upsample(depth_lr, image, scale, settings["radius_per_scale"], settings["epsilon"], lr_valid)
        depth = hybrid_refine(
            network_depth, input_depth, input_valid, guided, confidence, boundary, image, self.settings, self.flags
        )
        if output_size != working_size:
            size = output_size[::-1]
            depth = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)
            confidence = cv2.resize(confidence, size, interpolation=cv2.INTER_LINEAR)
            boundary = cv2.resize(boundary, size, interpolation=cv2.INTER_LINEAR)
        marks["refine"] = time.perf_counter()

        steps = list(marks)
        timings = {f"{b}_ms": (marks[b] - marks[a]) * 1000 for a, b in zip(steps, steps[1:])}
        timings["total_ms"] = (marks["refine"] - start) * 1000
        return {
            "depth": depth.astype(np.float32),
            "confidence": confidence.astype(np.float32),
            "boundary": boundary.astype(np.float32),
            "diagnosis": asdict(diagnosis),
            "timings": timings,
        }


@lru_cache(maxsize=2)
def get_predictor(checkpoint_path: str | None = None, device: str | None = None) -> Predictor:
    """Return the shared predictor, loading its checkpoint only on first use."""
    return Predictor(checkpoint_path, torch.device(device) if device else None)


def predict(
    rgb: np.ndarray, lr_depth: np.ndarray, scale: int, checkpoint_path: str | None = None, device: str | None = None
) -> dict[str, Any]:
    """Run the final hybrid pipeline. See :meth:`Predictor.predict`."""
    return get_predictor(checkpoint_path, device).predict(rgb, lr_depth, scale)
