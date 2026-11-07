"""Interpolation baselines: nearest, bilinear and bicubic upsampling with cv2."""

from __future__ import annotations

import cv2
import numpy as np

from src.classical.hole_filling import fill_holes


def _upsample(
    depth: np.ndarray, scale_factor: int, interpolation: int, validity_mask: np.ndarray | None
) -> np.ndarray:
    """Fill holes if a mask is given, then resize by ``scale_factor``."""
    filled = depth if validity_mask is None else fill_holes(depth, validity_mask)
    height, width = filled.shape
    return cv2.resize(filled, (width * scale_factor, height * scale_factor), interpolation=interpolation)


def prepare_network_input(
    lr_depth: np.ndarray, lr_valid: np.ndarray, scale_factor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return hole-filled, bicubic-upsampled depth and a nearest-upsampled validity mask."""
    height, width = lr_valid.shape
    mask = cv2.resize(
        lr_valid.astype(np.uint8), (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    return upsample_bicubic(lr_depth, scale_factor, lr_valid), mask


def upsample_nearest(
    depth: np.ndarray, scale_factor: int, validity_mask: np.ndarray | None = None
) -> np.ndarray:
    """Upsample depth with nearest-neighbour interpolation."""
    return _upsample(depth, scale_factor, cv2.INTER_NEAREST, validity_mask)


def upsample_bilinear(
    depth: np.ndarray, scale_factor: int, validity_mask: np.ndarray | None = None
) -> np.ndarray:
    """Upsample depth with bilinear interpolation."""
    return _upsample(depth, scale_factor, cv2.INTER_LINEAR, validity_mask)


def upsample_bicubic(
    depth: np.ndarray, scale_factor: int, validity_mask: np.ndarray | None = None
) -> np.ndarray:
    """Upsample depth with bicubic interpolation, clamped to the input's depth range.

    The clamp removes the overshoot bicubic produces at depth edges, which could
    otherwise create depths outside the measured range, including negative ones.
    """
    filled = depth if validity_mask is None else fill_holes(depth, validity_mask)
    result = _upsample(filled, scale_factor, cv2.INTER_CUBIC, None)
    return np.clip(result, filled.min(), filled.max())
