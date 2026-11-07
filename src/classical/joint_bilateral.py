"""Joint bilateral upsampling of depth, guided by a high-resolution RGB image.

Depth is first upsampled bilinearly, then filtered at full resolution with weights
from pixel distance and RGB similarity. Uses ``cv2.ximgproc`` when available, and an
equivalent numpy implementation otherwise.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.classical.hole_filling import fill_holes
from src.classical.interpolation import upsample_bilinear

HAS_XIMGPROC = hasattr(cv2, "ximgproc")


def joint_bilateral_filter_numpy(
    depth: np.ndarray,
    guide: np.ndarray,
    radius: int,
    sigma_spatial: float,
    sigma_range: float,
) -> np.ndarray:
    """Apply a joint bilateral filter over a circular window.

    Colour distance is the L1 norm across channels, matching OpenCV.
    """
    pad = ((radius, radius), (radius, radius))
    padded_depth = np.pad(depth, pad, mode="reflect")
    padded_guide = np.pad(guide, pad + ((0, 0),), mode="reflect")
    height, width = depth.shape
    total = np.zeros_like(depth, dtype=np.float64)
    weights = np.zeros_like(depth, dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance_sq = dx * dx + dy * dy
            if distance_sq > radius * radius:
                continue
            rows = slice(radius + dy, radius + dy + height)
            cols = slice(radius + dx, radius + dx + width)
            colour = np.abs(padded_guide[rows, cols] - guide).sum(axis=2)
            weight = np.exp(
                -distance_sq / (2 * sigma_spatial**2) - colour**2 / (2 * sigma_range**2)
            )
            total += weight * padded_depth[rows, cols]
            weights += weight
    return (total / weights).astype(np.float32)


def joint_bilateral_upsample(
    depth_lr: np.ndarray,
    guide_rgb: np.ndarray,
    scale_factor: int,
    sigma_spatial_per_scale: float = 1.0,
    sigma_range: float = 0.1,
    validity_mask: np.ndarray | None = None,
    use_ximgproc: bool | None = None,
) -> np.ndarray:
    """Upsample depth using a float32 RGB guide in [0,1] at the target resolution.

    The spatial sigma is ``sigma_spatial_per_scale * scale_factor`` and the window
    radius is twice that. ``use_ximgproc=None`` picks the backend automatically.
    """
    filled = depth_lr if validity_mask is None else fill_holes(depth_lr, validity_mask)
    initial = upsample_bilinear(filled, scale_factor)
    sigma_spatial = sigma_spatial_per_scale * scale_factor
    radius = math.ceil(2 * sigma_spatial)
    if HAS_XIMGPROC if use_ximgproc is None else use_ximgproc:
        return cv2.ximgproc.jointBilateralFilter(
            guide_rgb, initial, 2 * radius + 1, sigma_range, sigma_spatial
        )
    return joint_bilateral_filter_numpy(initial, guide_rgb, radius, sigma_spatial, sigma_range)
