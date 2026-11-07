"""KITTI depth completion loader for ``val_selection_cropped``.

Expected subfolders are ``image/``, ``velodyne_raw/`` and ``groundtruth_depth/``.
Depth PNGs are 16-bit: metres = value / 256, and 0 means invalid. Ground truth is
semi-dense, so metrics must only use valid pixels.

Use ``open_kitti`` to get ``None`` and a message when the folder is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import resolve_dataset_root

SUBFOLDERS = ("image", "velodyne_raw", "groundtruth_depth")


def read_depth_png(png_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 16-bit KITTI depth PNG and return depth in metres and a validity mask."""
    raw = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"Could not read {png_path}")
    if raw.dtype != np.uint16:
        raise ValueError(f"{png_path}: expected 16-bit depth, got {raw.dtype}")
    return raw.astype(np.float32) / 256.0, raw > 0


class KITTIDepthCompletionDataset(Dataset):
    """RGB, sparse LiDAR and semi-dense ground truth from KITTI."""

    def __init__(self, root: str | Path) -> None:
        """Match image, LiDAR and ground truth files by name, raising if any are missing."""
        self.root = Path(root)
        self.samples: list[tuple[Path, Path, Path]] = []
        for image in sorted((self.root / "image").glob("*.png")):
            lidar = self.root / "velodyne_raw" / image.name.replace("_image_", "_velodyne_raw_", 1)
            target = self.root / "groundtruth_depth" / image.name.replace("_image_", "_groundtruth_depth_", 1)
            for path in (lidar, target):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing KITTI file: {path}")
            self.samples.append((image, lidar, target))
        if not self.samples:
            raise FileNotFoundError(f"No KITTI images found in {self.root / 'image'}")

    def __len__(self) -> int:
        """Return the number of matched samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return rgb, ground truth depth and mask, and raw LiDAR depth and mask."""
        image_path, lidar_path, target_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read {image_path}")
        rgb = image[:, :, ::-1].astype(np.float32).transpose(2, 0, 1) / 255.0
        depth, valid = read_depth_png(target_path)
        lidar, lidar_valid = read_depth_png(lidar_path)
        return {
            "rgb": torch.from_numpy(rgb.copy()),
            "depth": torch.from_numpy(depth)[None],
            "valid": torch.from_numpy(valid)[None],
            "lidar": torch.from_numpy(lidar)[None],
            "lidar_valid": torch.from_numpy(lidar_valid)[None],
            "name": image_path.stem,
        }


def open_kitti(root: str | Path) -> KITTIDepthCompletionDataset | None:
    """Return the dataset, or print a message and return ``None`` if the folder is missing."""
    root = Path(root)
    if not root.is_dir():
        print(f"[KITTI] Skipped: folder not found: {root}")
        return None
    return KITTIDepthCompletionDataset(root)


def build_kitti_dataset(config: dict[str, Any]) -> KITTIDepthCompletionDataset | None:
    """Open KITTI using the path in the config, or skip it if the folder is missing."""
    return open_kitti(resolve_dataset_root(config) / config["datasets"]["kitti"]["root"])
