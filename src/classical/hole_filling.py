"""Basic hole filling, applied before any interpolation or filtering."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def fill_holes(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace each invalid pixel with its nearest valid pixel's depth.

    Nearest-neighbour filling invents no in-between depths across object boundaries.
    """
    if not valid.any():
        raise ValueError("Cannot fill holes: the depth map has no valid pixels.")
    if valid.all():
        return depth.copy()
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return depth[tuple(nearest)]
