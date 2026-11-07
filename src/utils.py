"""Shared helpers: device selection, configuration access and depth array operations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Must be set before torch is imported: unsupported MPS operations fall back to the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_device() -> torch.device:
    """Return the best available device, in the order cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path = PROJECT_ROOT / "configs" / "default.yaml") -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path against the project root, leaving absolute paths unchanged."""
    return (PROJECT_ROOT / path).resolve()


def resolve_dataset_root(config: dict[str, Any]) -> Path:
    """Return the absolute dataset root from ``config["paths"]["dataset_root"]``.

    Relative values resolve against the project root, never the working directory.
    """
    return resolve_project_path(config["paths"]["dataset_root"])


def masked_area_resize(
    depth: np.ndarray,
    valid: np.ndarray,
    size_hw: tuple[int, int],
    min_coverage: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-resize depth using valid pixels only, so invalid zeros never bias the average.

    An output pixel is valid when at least ``min_coverage`` of its area was valid.
    Invalid output pixels are set to 0.
    """
    dsize = (size_hw[1], size_hw[0])
    mask = valid.astype(np.float32)
    weight = cv2.resize(mask, dsize, interpolation=cv2.INTER_AREA)
    total = cv2.resize(np.where(valid, depth, 0).astype(np.float32), dsize, interpolation=cv2.INTER_AREA)
    out_valid = weight >= min_coverage
    out = np.zeros_like(total)
    out[out_valid] = total[out_valid] / weight[out_valid]
    return out, out_valid


def depth_edges(depth: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    """Mark pixels whose relative depth jump to a valid neighbour exceeds ``threshold``.

    The jump is ``|a - b| / min(a, b)`` over right and down neighbours. Pairs with an
    invalid or non-positive pixel never produce an edge.
    """

    def jump(a: np.ndarray, b: np.ndarray, ok: np.ndarray) -> np.ndarray:
        result = np.zeros(a.shape, dtype=bool)
        result[ok] = np.abs(a[ok] - b[ok]) / np.minimum(a[ok], b[ok]) > threshold
        return result

    usable = valid & (depth > 0)
    edges = np.zeros(depth.shape, dtype=bool)
    edges[:, :-1] |= jump(depth[:, :-1], depth[:, 1:], usable[:, :-1] & usable[:, 1:])
    edges[:-1, :] |= jump(depth[:-1, :], depth[1:, :], usable[:-1, :] & usable[1:, :])
    return edges


def mod_crop(array: np.ndarray, multiple: int) -> np.ndarray:
    """Centre-crop the first two axes of an array so each is divisible by ``multiple``."""
    height, width = array.shape[:2]
    top = (height % multiple) // 2
    left = (width % multiple) // 2
    return array[top : top + height - height % multiple, left : left + width - width % multiple]
