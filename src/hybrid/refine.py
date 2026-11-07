"""Hybrid refinement of network depth, using its confidence and boundary outputs.

Three steps, applied in order, each with an ablation flag. The final ``hybrid`` method takes
its flags from the config, which turns fusion off:

1. Fusion: below ``low_confidence``, final = conf * network + (1 - conf) * guided filter.
2. Keep measured: at or above ``high_confidence``, valid input pixels keep the input depth.
3. Boundary refinement: near predicted boundaries, run a joint bilateral pass whose
   window only averages pixels on the same side of the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class HybridFlags:
    """Which refinement steps to run."""

    fusion: bool = True
    keep_measured: bool = True
    boundary_refine: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HybridFlags":
        """Build the flags of the final hybrid method from a full config."""
        return cls(**config["hybrid"]["flags"])


@dataclass(frozen=True)
class HybridSettings:
    """Thresholds and filter settings, from the ``hybrid`` section of the config."""

    low_confidence: float = 0.5
    high_confidence: float = 0.8
    boundary_probability: float = 0.5
    boundary_band_px: int = 2
    boundary_sigma_spatial: float = 2.0
    boundary_sigma_range: float = 0.1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HybridSettings":
        """Build the settings from a full config."""
        return cls(**{key: value for key, value in config["hybrid"].items() if key != "flags"})


def confidence_fusion(
    network_depth: np.ndarray,
    guided_depth: np.ndarray,
    confidence: np.ndarray,
    low_confidence: float,
) -> np.ndarray:
    """Blend in guided filter depth where confidence is low, and keep the network elsewhere."""
    blended = confidence * network_depth + (1 - confidence) * guided_depth
    return np.where(confidence < low_confidence, blended, network_depth)


def keep_measured_depth(
    depth: np.ndarray,
    input_depth: np.ndarray,
    input_valid: np.ndarray,
    confidence: np.ndarray,
    high_confidence: float,
) -> np.ndarray:
    """Restore the input depth at valid pixels where confidence is high."""
    return np.where(input_valid & (confidence >= high_confidence), input_depth, depth)


def boundary_aware_refine(
    depth: np.ndarray,
    boundary_probability: np.ndarray,
    rgb: np.ndarray,
    settings: HybridSettings,
) -> np.ndarray:
    """Filter pixels near predicted boundaries, without averaging across a boundary.

    The boundary splits the image into connected regions, and each boundary pixel joins
    its nearest region. A window only uses pixels from the centre pixel's region. Pixels
    outside the boundary band are returned unchanged.
    """
    boundary = boundary_probability >= settings.boundary_probability
    if not boundary.any() or boundary.all():
        return depth.copy()
    _, labels = cv2.connectedComponents((~boundary).astype(np.uint8), connectivity=4)
    nearest = ndimage.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
    labels = labels[tuple(nearest)]
    size = 2 * settings.boundary_band_px + 1
    band = cv2.dilate(boundary.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0

    sigma_spatial, sigma_range = settings.boundary_sigma_spatial, settings.boundary_sigma_range
    radius = math.ceil(2 * sigma_spatial)
    pad = ((radius, radius), (radius, radius))
    padded_depth = np.pad(depth, pad, mode="reflect")
    padded_rgb = np.pad(rgb, pad + ((0, 0),), mode="reflect")
    padded_labels = np.pad(labels, pad, mode="reflect")
    rows, cols = np.nonzero(band)
    centre_rgb = rgb[rows, cols]
    centre_labels = labels[rows, cols]
    total = np.zeros(rows.size, dtype=np.float64)
    weights = np.zeros(rows.size, dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance_sq = dx * dx + dy * dy
            if distance_sq > radius * radius:
                continue
            y, x = rows + radius + dy, cols + radius + dx
            colour = np.abs(padded_rgb[y, x] - centre_rgb).sum(axis=1)
            weight = np.exp(-distance_sq / (2 * sigma_spatial**2) - colour**2 / (2 * sigma_range**2))
            weight = weight * (padded_labels[y, x] == centre_labels)
            total += weight * padded_depth[y, x]
            weights += weight
    refined = depth.copy()
    refined[rows, cols] = total / weights
    return refined


def hybrid_refine(
    network_depth: np.ndarray,
    input_depth: np.ndarray,
    input_valid: np.ndarray,
    guided_depth: np.ndarray | None,
    confidence: np.ndarray,
    boundary_probability: np.ndarray,
    rgb: np.ndarray,
    settings: HybridSettings,
    flags: HybridFlags = HybridFlags(),
) -> np.ndarray:
    """Run the enabled refinement steps. With all of them off, this returns network depth.

    ``guided_depth`` is only read by fusion, and can be None when fusion is off.
    """
    depth = network_depth
    if flags.fusion:
        if guided_depth is None:
            raise ValueError("Fusion needs guided_depth")
        depth = confidence_fusion(depth, guided_depth, confidence, settings.low_confidence)
    if flags.keep_measured:
        depth = keep_measured_depth(depth, input_depth, input_valid, confidence, settings.high_confidence)
    if flags.boundary_refine:
        depth = boundary_aware_refine(depth, boundary_probability, rgb, settings)
    return depth.astype(np.float32)
