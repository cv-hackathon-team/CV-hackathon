"""Create degraded low-resolution depth inputs and validity masks from ground truth.

Every input is first downsampled by a factor ``s`` with area interpolation. The mode
then optionally adds further damage to the low-resolution map:

    lr        no further damage
    noisy     Gaussian noise with std proportional to depth
    holes     random blob-shaped holes
    sparse    keep p% of the valid pixels
    boundary  dilate depth edges and drop pixels in that band
    mixed     boundary, holes, sparse and noise combined

Severity settings live under ``corruption.levels`` in the config. Randomness comes
only from the generator passed in, and ``make_rng`` derives one per sample, so results
do not depend on iteration order or DataLoader worker count.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.classical.interpolation import prepare_network_input
from src.utils import depth_edges, masked_area_resize, mod_crop

MODES = ("lr", "noisy", "holes", "sparse", "boundary", "mixed")


@dataclass
class CorruptedDepth:
    """A low-resolution input: depth in metres (0 where invalid), mask and label."""

    lr_depth: np.ndarray
    validity_mask: np.ndarray
    label: str


def make_rng(seed: int, index: int, scale: int, mode: str, epoch: int = 0) -> np.random.Generator:
    """Return a generator determined entirely by the seed, sample, scale, mode and epoch."""
    return np.random.default_rng([seed, epoch, index, scale, zlib.crc32(mode.encode())])


def get_corruption_params(config: dict[str, Any], severity: str | None = None) -> dict[str, float]:
    """Return the settings for a severity level, defaulting to the one in the config."""
    settings = config["corruption"]
    return settings["levels"][severity or settings["severity"]]


def downsample_area(
    depth: np.ndarray, valid: np.ndarray, scale: int
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample by ``scale`` with area interpolation over valid pixels only."""
    height, width = depth.shape
    if height % scale or width % scale:
        raise ValueError(f"Size {(height, width)} is not divisible by scale {scale}; use mod_crop.")
    return masked_area_resize(depth, valid, (height // scale, width // scale))


def add_depth_noise(
    depth: np.ndarray, valid: np.ndarray, rel_sigma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise with std ``rel_sigma * depth`` to valid pixels."""
    noise = rng.standard_normal(depth.shape, dtype=np.float32)
    noisy = np.where(valid, np.maximum(depth * (1 + rel_sigma * noise), 1e-3), 0).astype(np.float32)
    return noisy, valid.copy()


def add_holes(
    depth: np.ndarray,
    valid: np.ndarray,
    hole_ratio: float,
    blob_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove about ``hole_ratio`` of the image as blobs, by thresholding smoothed noise."""
    sigma = max(1.0, blob_scale * min(depth.shape))
    field = cv2.GaussianBlur(rng.standard_normal(depth.shape, dtype=np.float32), (0, 0), sigma)
    holes = field >= np.quantile(field, 1 - hole_ratio)
    kept = valid & ~holes
    return np.where(kept, depth, 0).astype(np.float32), kept


def sparsify(
    depth: np.ndarray, valid: np.ndarray, keep_percent: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a random ``keep_percent`` of the valid pixels, and at least one."""
    candidates = np.flatnonzero(valid)
    count = max(1, round(candidates.size * keep_percent / 100)) if candidates.size else 0
    kept = np.zeros(depth.size, dtype=bool)
    kept[rng.choice(candidates, size=count, replace=False)] = True
    kept = kept.reshape(depth.shape)
    return np.where(kept, depth, 0).astype(np.float32), kept


def corrupt_boundaries(
    depth: np.ndarray,
    valid: np.ndarray,
    band_px: int,
    drop_prob: float,
    edge_threshold: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Dilate depth edges by ``band_px`` and drop band pixels with probability ``drop_prob``."""
    edges = depth_edges(depth, valid, edge_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    band = cv2.dilate(edges, kernel).astype(bool)
    kept = valid & ~(band & (rng.random(depth.shape) < drop_prob))
    return np.where(kept, depth, 0).astype(np.float32), kept


def corrupt_depth(
    gt_depth: np.ndarray,
    gt_valid: np.ndarray,
    scale: int,
    mode: str,
    params: dict[str, float],
    rng: np.random.Generator,
) -> CorruptedDepth:
    """Downsample ground truth by ``scale`` and apply the corruption named by ``mode``."""
    if mode not in MODES:
        raise ValueError(f"Unknown corruption mode {mode!r}; expected one of {MODES}")
    depth, valid = downsample_area(gt_depth, gt_valid, scale)
    if mode in ("boundary", "mixed"):
        depth, valid = corrupt_boundaries(
            depth, valid, int(params["boundary_band_px"]), params["boundary_drop_prob"],
            params["boundary_edge_threshold"], rng,
        )
    if mode in ("holes", "mixed"):
        depth, valid = add_holes(depth, valid, params["hole_ratio"], params["hole_blob_scale"], rng)
    if mode in ("sparse", "mixed"):
        depth, valid = sparsify(depth, valid, params["sparse_keep_percent"], rng)
    if mode in ("noisy", "mixed"):
        depth, valid = add_depth_noise(depth, valid, params["noise_rel_sigma"], rng)
    return CorruptedDepth(depth, valid, mode)


def paired_random_crop_flip(
    rgb: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    crop_size_hw: tuple[int, int] | None,
    flip: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one random crop and horizontal flip to rgb (H,W,3), depth and valid (H,W)."""
    if crop_size_hw is not None:
        crop_h, crop_w = crop_size_hw
        height, width = depth.shape
        if crop_h > height or crop_w > width:
            raise ValueError(f"Crop {crop_size_hw} is larger than image {(height, width)}")
        top = int(rng.integers(0, height - crop_h + 1))
        left = int(rng.integers(0, width - crop_w + 1))
        window = (slice(top, top + crop_h), slice(left, left + crop_w))
        rgb, depth, valid = rgb[window], depth[window], valid[window]
    if flip and rng.random() < 0.5:
        rgb, depth, valid = rgb[:, ::-1], depth[:, ::-1], valid[:, ::-1]
    return rgb.copy(), depth.copy(), valid.copy()


class CorruptedDataset(Dataset):
    """Wrap a clean dataset with paired augmentation and seeded corruption."""

    def __init__(
        self,
        base: Dataset,
        scale: int,
        mode: str,
        params: dict[str, float],
        seed: int,
        crop_size_hw: tuple[int, int] | None = None,
        flip: bool = False,
    ) -> None:
        """Store settings. A crop must be divisible by ``scale``; otherwise images are mod-cropped."""
        if crop_size_hw is not None and (crop_size_hw[0] % scale or crop_size_hw[1] % scale):
            raise ValueError(f"Crop {crop_size_hw} is not divisible by scale {scale}")
        self.base, self.scale, self.mode, self.params = base, scale, mode, params
        self.seed, self.crop_size_hw, self.flip = seed, crop_size_hw, flip
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the random draws for a new epoch, while keeping each epoch reproducible."""
        self.epoch = epoch

    def __len__(self) -> int:
        """Return the size of the wrapped dataset."""
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the clean sample with lr_depth, lr_valid, label and scale added."""
        sample = self.base[index]
        rng = make_rng(self.seed, index, self.scale, self.mode, self.epoch)
        rgb = sample["rgb"].numpy().transpose(1, 2, 0)
        depth, valid = sample["depth"][0].numpy(), sample["valid"][0].numpy()
        if self.crop_size_hw is None:
            rgb, depth, valid = (mod_crop(a, self.scale) for a in (rgb, depth, valid))
        rgb, depth, valid = paired_random_crop_flip(rgb, depth, valid, self.crop_size_hw, self.flip, rng)
        corrupted = corrupt_depth(depth, valid, self.scale, self.mode, self.params, rng)
        return {
            "rgb": torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
            "depth": torch.from_numpy(depth)[None],
            "valid": torch.from_numpy(valid)[None],
            "lr_depth": torch.from_numpy(corrupted.lr_depth)[None],
            "lr_valid": torch.from_numpy(corrupted.validity_mask)[None],
            "label": corrupted.label,
            "scale": self.scale,
            "name": sample["name"],
        }


class NetworkDataset(Dataset):
    """Serve network inputs from a clean dataset, seeded per sample.

    Items are requested as ``(index, scale, epoch, mode)`` tuples, so a batch sampler
    controls the scale and results do not depend on worker count. When ``mode`` is
    ``None`` a corruption is drawn at random. A crop gives training samples; without
    one, whole images are mod-cropped to ``mod`` for validation.
    """

    def __init__(
        self,
        base: Dataset,
        modes: list[str],
        params: dict[str, float],
        seed: int,
        crop_size_hw: tuple[int, int] | None = None,
        flip: bool = False,
        mod: int = 8,
    ) -> None:
        """Store settings. Crops must be divisible by every scale and by ``mod``."""
        self.base, self.modes, self.params, self.seed = base, list(modes), params, seed
        self.crop_size_hw, self.flip, self.mod = crop_size_hw, flip, mod

    def __len__(self) -> int:
        """Return the size of the wrapped dataset."""
        return len(self.base)

    def __getitem__(self, item: tuple[int, int, int, str | None]) -> dict[str, Any]:
        """Return tensors for the network, the target and the mask used for scoring."""
        index, scale, epoch, mode = item
        rng = make_rng(self.seed, index, scale, "network", epoch)
        if mode is None:
            mode = str(rng.choice(self.modes))
        sample = self.base[index]
        rgb = sample["rgb"].numpy().transpose(1, 2, 0)
        depth, valid = sample["depth"][0].numpy(), sample["valid"][0].numpy()
        if self.crop_size_hw is None:
            rgb, depth, valid = (mod_crop(a, self.mod) for a in (rgb, depth, valid))
        rgb, depth, valid = paired_random_crop_flip(rgb, depth, valid, self.crop_size_hw, self.flip, rng)
        corrupted = corrupt_depth(depth, valid, scale, mode, self.params, rng)
        depth_input, input_valid = prepare_network_input(corrupted.lr_depth, corrupted.validity_mask, scale)
        return {
            "rgb": torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
            "depth_input": torch.from_numpy(depth_input)[None],
            "input_valid": torch.from_numpy(input_valid)[None],
            "target": torch.from_numpy(depth)[None],
            "target_valid": torch.from_numpy(valid)[None],
            "label": corrupted.label,
            "scale": scale,
            "name": sample["name"],
        }
