"""Evaluation metrics, computed only over valid ground truth pixels."""

from __future__ import annotations

import resource
import sys
import time
from typing import Callable

import numpy as np
import torch
from scipy import ndimage
from skimage.metrics import structural_similarity

from src.classical.hole_filling import fill_holes
from src.utils import depth_edges


SSIM_WINDOW = 7


def rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Return root mean squared error in metres over valid pixels."""
    return float(np.sqrt(np.mean((prediction[mask].astype(np.float64) - target[mask]) ** 2)))


def mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Return mean absolute error in metres over valid pixels."""
    return float(np.mean(np.abs(prediction[mask].astype(np.float64) - target[mask])))


def valid_data_range(target: np.ndarray, mask: np.ndarray) -> float:
    """Return the ground truth depth range over valid pixels, used for PSNR and SSIM."""
    return float(target[mask].max() - target[mask].min())


def psnr(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, data_range: float | None = None
) -> float:
    """Return PSNR in dB over valid pixels, or inf for a perfect prediction.

    ``data_range`` defaults to the image's own ground truth range, so values are
    comparable between methods, but not with papers that fix the range (such as 10 m).
    """
    data_range = valid_data_range(target, mask) if data_range is None else data_range
    error = float(np.mean((prediction[mask].astype(np.float64) - target[mask]) ** 2))
    return float("inf") if error == 0 else float(10 * np.log10(data_range**2 / error))


def ssim(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, data_range: float | None = None
) -> float:
    """Return mean SSIM over valid pixels.

    Invalid pixels are set to the same filled value in both images, so they add no
    error to windows that overlap them. The map is then averaged over valid pixels,
    excluding the image border that SSIM windows cannot fully cover.
    """
    data_range = valid_data_range(target, mask) if data_range is None else data_range
    filled_target = fill_holes(target.astype(np.float64), mask)
    filled_prediction = np.where(mask, prediction.astype(np.float64), filled_target)
    _, ssim_map = structural_similarity(
        filled_prediction, filled_target, win_size=SSIM_WINDOW, data_range=data_range, full=True
    )
    pad = SSIM_WINDOW // 2
    inner = mask[pad:-pad, pad:-pad]
    return float(ssim_map[pad:-pad, pad:-pad][inner].mean())


def boundary_f_score(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    tolerance_px: int = 2,
    edge_threshold: float = 0.05,
) -> float:
    """Return the F-score of predicted depth edges against ground truth edges.

    An edge counts as matched if the other map has an edge within ``tolerance_px``.
    Returns NaN when the ground truth has no edges, since the score is then undefined.
    """
    target_edges = depth_edges(target, mask, edge_threshold)
    predicted_edges = depth_edges(prediction, mask, edge_threshold)
    if not target_edges.any():
        return float("nan")
    if not predicted_edges.any():
        return 0.0
    near_target = ndimage.distance_transform_edt(~target_edges) <= tolerance_px
    near_predicted = ndimage.distance_transform_edt(~predicted_edges) <= tolerance_px
    precision = (predicted_edges & near_target).sum() / predicted_edges.sum()
    recall = (target_edges & near_predicted).sum() / target_edges.sum()
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def measure_latency(fn: Callable[[], object], repeats: int = 10, warmup: int = 3) -> dict[str, float]:
    """Return the mean, median and standard deviation of ``fn`` in milliseconds, after warm-up runs.

    ``fn`` must finish its device work before returning, for example by copying results to CPU.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)), "std_ms": float(np.std(times))}


def measure_memory(fn: Callable[[], object]) -> dict[str, float]:
    """Run ``fn`` and return peak process memory, and device memory where a counter exists, in MB.

    Peak process memory never goes down, so call this in a fresh process for a clean figure.
    Apple GPUs share system memory. Their figure is the driver allocation after ``fn``, not a peak.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fn()
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    result = {"process_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor}
    if torch.cuda.is_available():
        result["device_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    elif torch.backends.mps.is_available():
        result["device_mb"] = torch.mps.driver_allocated_memory() / 1024**2
    return result
