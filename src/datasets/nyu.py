"""NYU Depth v2 labeled dataset loader.

Reads ``nyu_depth_v2_labeled.mat``, a MATLAB v7.3 file, with h5py. Arrays stored in
this format come back with their axes reversed: images are (N, 3, W, H) and depths
are (N, W, H) on disk. Samples are read one index at a time and transposed here.

``depths`` is the inpainted depth map, so ``depth > 0`` holds for almost every pixel.
Depth values at pixels the sensor missed are estimates rather than measurements.
Training uses ``depths``. With ``include_measured=True``, samples also carry a
``measured`` mask (``rawDepths > 0``), which evaluation uses to score sensor pixels only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from src.utils import resolve_dataset_root


def load_official_splits(splits_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return 0-based train and test indices from ``splits.mat``, which stores 1-based ones."""
    splits = scipy.io.loadmat(str(splits_path))
    train = np.sort(splits["trainNdxs"].ravel().astype(np.int64) - 1)
    test = np.sort(splits["testNdxs"].ravel().astype(np.int64) - 1)
    if np.intersect1d(train, test).size:
        raise ValueError(f"{splits_path}: train and test indices overlap")
    return train, test


def read_scene_names(mat_path: str | Path) -> list[str]:
    """Return the scene name of every image from the ``scenes`` field."""
    with h5py.File(mat_path, "r") as f:
        return ["".join(map(chr, f[ref][()].ravel())) for ref in f["scenes"][0]]


def make_train_val_split(
    train_indices: np.ndarray,
    scenes: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split official TRAIN indices into train and val, keeping each scene on one side.

    Whole scenes are moved to val in a seeded random order until val holds at least
    ``val_fraction`` of the images. The test split is never an input to this function.
    """
    names = np.array([scenes[i] for i in train_indices])
    unique_scenes = np.unique(names)
    order = np.random.default_rng(seed).permutation(unique_scenes)
    target = round(val_fraction * len(train_indices))
    val_scenes: list[str] = []
    count = 0
    for scene in order:
        if count >= target:
            break
        val_scenes.append(str(scene))
        count += int((names == scene).sum())
    is_val = np.isin(names, val_scenes)
    return train_indices[~is_val], train_indices[is_val]


class NYUDepthV2Dataset(Dataset):
    """RGB and metric depth pairs, read from disk per index.

    The test split raises unless ``allow_test=True``, which only evaluation code sets.
    """

    def __init__(
        self,
        mat_path: str | Path,
        splits_path: str | Path,
        split: str = "train",
        val_fraction: float = 0.1,
        split_seed: int = 42,
        allow_test: bool = False,
        include_measured: bool = False,
    ) -> None:
        """Resolve the image indices of a split without loading any image data."""
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split!r}")
        if split == "test" and not allow_test:
            raise PermissionError("The NYU test split is reserved for final evaluation.")
        self.mat_path = Path(mat_path)
        self.split = split
        self.include_measured = include_measured
        self.scenes = read_scene_names(self.mat_path)
        official_train, official_test = load_official_splits(splits_path)
        if split == "test":
            self.indices = official_test
        else:
            train, val = make_train_val_split(official_train, self.scenes, val_fraction, split_seed)
            self.indices = train if split == "train" else val
        self._file: h5py.File | None = None
        self._file_pid: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Drop the open HDF5 handle, which cannot be sent to DataLoader workers."""
        state = self.__dict__.copy()
        state["_file"] = None
        state["_file_pid"] = None
        return state

    def _handle(self) -> h5py.File:
        """Return an HDF5 handle owned by the current process."""
        if self._file is None or self._file_pid != os.getpid():
            self._file = h5py.File(self.mat_path, "r")
            self._file_pid = os.getpid()
        return self._file

    def __len__(self) -> int:
        """Return the number of samples in the split."""
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return rgb (3,H,W) in [0,1], depth (1,H,W) in metres and valid (1,H,W) bool."""
        file_index = int(self.indices[index])
        f = self._handle()
        rgb = f["images"][file_index].transpose(0, 2, 1).astype(np.float32) / 255.0
        depth = f["depths"][file_index].T.astype(np.float32)
        sample = {
            "rgb": torch.from_numpy(rgb),
            "depth": torch.from_numpy(depth)[None],
            "valid": torch.from_numpy(depth > 0)[None],
            "name": f"{self.scenes[file_index]}_{file_index:04d}",
        }
        if self.include_measured:
            sample["measured"] = torch.from_numpy(f["rawDepths"][file_index].T > 0)[None]
        return sample


def build_nyu_dataset(
    config: dict[str, Any], split: str, allow_test: bool = False, include_measured: bool = False
) -> NYUDepthV2Dataset:
    """Build a dataset for a split using the paths and split settings in the config."""
    settings = config["datasets"]["nyu"]
    root = resolve_dataset_root(config)
    return NYUDepthV2Dataset(
        root / settings["mat_file"],
        root / settings["splits_file"],
        split=split,
        val_fraction=settings["val_fraction"],
        split_seed=settings["split_seed"],
        allow_test=allow_test,
        include_measured=include_measured,
    )


def inspect_mat(mat_path: str | Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Return the on-disk shape and dtype of the image and depth arrays."""
    with h5py.File(mat_path, "r") as f:
        return {key: (f[key].shape, str(f[key].dtype)) for key in ("images", "depths", "scenes")}
