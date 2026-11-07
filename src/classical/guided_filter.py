"""Guided filter upsampling of depth, using a high-resolution RGB image as the guide.

Depth is first upsampled bilinearly, then filtered at full resolution. Uses
``cv2.ximgproc`` when available, and the colour guided filter of He et al. in numpy
otherwise.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.classical.hole_filling import fill_holes
from src.classical.interpolation import upsample_bilinear

HAS_XIMGPROC = hasattr(cv2, "ximgproc")


def box_filter(image: np.ndarray, radius: int) -> np.ndarray:
    """Return the mean over a square window of the given radius."""
    size = 2 * radius + 1
    return cv2.blur(image, (size, size), borderType=cv2.BORDER_REFLECT)


def guided_filter_numpy(
    depth: np.ndarray, guide: np.ndarray, radius: int, epsilon: float
) -> np.ndarray:
    """Filter depth with a colour guide by solving a 3x3 linear model at each pixel."""
    guide = guide.astype(np.float64)
    depth = depth.astype(np.float64)
    mean_guide = box_filter(guide, radius)
    mean_depth = box_filter(depth, radius)
    mean_guide_depth = box_filter(guide * depth[..., None], radius)
    cov_guide_depth = mean_guide_depth - mean_guide * mean_depth[..., None]

    height, width = depth.shape
    covariance = np.empty((height, width, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i, 3):
            value = box_filter(guide[..., i] * guide[..., j], radius) - mean_guide[..., i] * mean_guide[..., j]
            covariance[..., i, j] = covariance[..., j, i] = value
    covariance += epsilon * np.eye(3)

    a = np.linalg.solve(covariance, cov_guide_depth[..., None])[..., 0]
    b = mean_depth - (a * mean_guide).sum(axis=2)
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    return ((mean_a * guide).sum(axis=2) + mean_b).astype(np.float32)


def guided_filter_ximgproc(
    depth: np.ndarray, guide: np.ndarray, radius: int, epsilon: float
) -> np.ndarray:
    """Filter depth with ``cv2.ximgproc`` using an 8-bit guide.

    With a float guide in [0,1], OpenCV 5.0 departs from the guided filter definition
    when epsilon is small (1e-3 gave errors near 0.8 m on NYU; 1e-2 did not). With an
    8-bit guide and epsilon scaled by 255^2, it matches the definition.
    """
    guide_8bit = np.round(guide * 255).astype(np.uint8)
    return cv2.ximgproc.guidedFilter(guide_8bit, depth, radius, epsilon * 255**2)


def guided_upsample(
    depth_lr: np.ndarray,
    guide_rgb: np.ndarray,
    scale_factor: int,
    radius_per_scale: float = 2,
    epsilon: float = 1e-3,
    validity_mask: np.ndarray | None = None,
    use_ximgproc: bool | None = None,
) -> np.ndarray:
    """Upsample depth using a float32 RGB guide in [0,1] at the target resolution.

    The window radius is ``radius_per_scale * scale_factor``, rounded and at least 1. ``use_ximgproc=None``
    picks the backend automatically.
    """
    filled = depth_lr if validity_mask is None else fill_holes(depth_lr, validity_mask)
    initial = upsample_bilinear(filled, scale_factor)
    radius = max(1, round(radius_per_scale * scale_factor))
    if HAS_XIMGPROC if use_ximgproc is None else use_ximgproc:
        return guided_filter_ximgproc(initial, guide_rgb, radius, epsilon)
    return guided_filter_numpy(initial, guide_rgb, radius, epsilon)
