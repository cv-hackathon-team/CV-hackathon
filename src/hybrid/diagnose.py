"""Rule-based diagnosis of corruption in a low-resolution depth map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

LABELS = ("clean", "noisy", "holes", "sparse", "boundary", "mixed")
_MEDIAN_ABS_TO_SIGMA = 1.4826
_MIN_PAIRS = 20


@dataclass(frozen=True)
class DiagnosisSettings:
    """Thresholds from the ``diagnosis`` section of the config."""

    clean_missing_percent: float = 1.0
    sparse_missing_percent: float = 50.0
    noise_level: float = 0.005
    hole_blob_ratio: float = 0.5

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DiagnosisSettings":
        """Build the settings from a full config."""
        return cls(**config["diagnosis"])


@dataclass
class Diagnosis:
    """Measured corruption statistics and the resulting label."""

    hole_percent: float
    noise_level: float
    sparsity: float
    label: str


def estimate_hole_percent(validity_mask: np.ndarray) -> float:
    """Return the percentage of pixels in contiguous invalid regions at least 2x2 in size."""
    blobs = ndimage.binary_opening(~validity_mask, structure=np.ones((2, 2), bool))
    return float(blobs.mean() * 100)


def estimate_noise_level(depth: np.ndarray, validity_mask: np.ndarray) -> float:
    """Estimate relative noise from neighbouring valid pixels, using medians to ignore real edges.

    Uses the second difference where three pixels in a row are valid, since that cancels
    smooth slopes, and the first difference on sparse maps where triples are too rare.
    Returns 0 when too few pixels are available to measure.
    """
    relative = {"second": [], "first": []}
    for axis in (0, 1):
        a = np.moveaxis(depth, axis, 0)
        v = np.moveaxis(validity_mask, axis, 0)
        triple = v[:-2] & v[1:-1] & v[2:]
        relative["second"].append(((a[:-2] - 2 * a[1:-1] + a[2:]) / np.maximum(a[1:-1], 1e-6))[triple])
        pair = v[:-1] & v[1:]
        relative["first"].append(((a[1:] - a[:-1]) / np.maximum(a[:-1], 1e-6))[pair])
    second = np.concatenate(relative["second"])
    if second.size >= _MIN_PAIRS:
        return float(np.median(np.abs(second)) * _MEDIAN_ABS_TO_SIGMA / np.sqrt(6))
    first = np.concatenate(relative["first"])
    if first.size >= _MIN_PAIRS:
        return float(np.median(np.abs(first)) * _MEDIAN_ABS_TO_SIGMA / np.sqrt(2))
    return 0.0


def estimate_sparsity(validity_mask: np.ndarray) -> float:
    """Return the percentage of pixels without a measurement."""
    return float((~validity_mask).mean() * 100)


def apply_rules(hole_percent: float, noise_level: float, sparsity: float, settings: DiagnosisSettings) -> str:
    """Choose a label from the three statistics."""
    noisy = noise_level > settings.noise_level
    if sparsity > settings.sparse_missing_percent:
        return "mixed" if noisy else "sparse"
    if sparsity <= settings.clean_missing_percent:
        return "noisy" if noisy else "clean"
    if noisy:
        return "mixed"
    return "holes" if hole_percent / sparsity >= settings.hole_blob_ratio else "boundary"


def diagnose(
    depth: np.ndarray, validity_mask: np.ndarray, settings: DiagnosisSettings = DiagnosisSettings()
) -> Diagnosis:
    """Compute the statistics of a low-resolution depth map and label it."""
    hole_percent = estimate_hole_percent(validity_mask)
    noise_level = estimate_noise_level(depth, validity_mask)
    sparsity = estimate_sparsity(validity_mask)
    return Diagnosis(hole_percent, noise_level, sparsity, apply_rules(hole_percent, noise_level, sparsity, settings))
