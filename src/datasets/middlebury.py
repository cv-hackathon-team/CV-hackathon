"""Middlebury 2014 stereo dataset loader.

Converts disparity to metric depth using each scene's ``calib.txt``:

    depth_mm = baseline_mm * focal_px / (disparity_px + doffs_px)

Invalid disparities are stored as ``inf`` in the PFM files. Infinite, non-positive
and non-finite disparities are all treated as invalid.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import masked_area_resize, resolve_dataset_root

REQUIRED_FILES = ("im0.png", "disp0.pfm", "calib.txt")


def parse_calib(calib_path: str | Path) -> dict[str, float]:
    """Parse ``calib.txt`` into focal length (px), baseline (mm), doffs (px) and image size."""
    text = Path(calib_path).read_text(encoding="utf-8")
    values = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    cam0 = re.findall(r"[-+.\deE]+", values["cam0"])
    return {
        "focal": float(cam0[0]),
        "baseline": float(values["baseline"]),
        "doffs": float(values["doffs"]),
        "width": float(values["width"]),
        "height": float(values["height"]),
    }


def read_pfm(pfm_path: str | Path) -> np.ndarray:
    """Read a single-channel PFM as a top-to-bottom float32 array.

    PFM rows are stored bottom-up, and a negative scale means little-endian.
    """
    with open(pfm_path, "rb") as f:
        header = f.readline().strip()
        if header != b"Pf":
            raise ValueError(f"{pfm_path}: expected single-channel PFM, got header {header!r}")
        width, height = map(int, f.readline().split())
        scale = float(f.readline().strip())
        dtype = "<f4" if scale < 0 else ">f4"
        data = np.fromfile(f, dtype=dtype, count=width * height)
    if data.size != width * height:
        raise ValueError(f"{pfm_path}: truncated, expected {width * height} values, got {data.size}")
    return data.reshape(height, width)[::-1].astype(np.float32)


def disparity_to_depth(
    disparity: np.ndarray, calib: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert disparity (px) to depth (m) and return depth with a validity mask.

    Invalid pixels are set to 0 in the returned depth.
    """
    valid = np.isfinite(disparity) & (disparity > 0)
    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[valid] = calib["baseline"] * calib["focal"] / (disparity[valid] + calib["doffs"]) / 1000.0
    return depth, valid


def find_scenes(root: str | Path) -> list[Path]:
    """Return sorted scene folders, raising if a scene lacks a required file."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Middlebury folder not found: {root}")
    scenes = sorted(p for p in root.iterdir() if p.is_dir())
    missing = [str(s / name) for s in scenes for name in REQUIRED_FILES if not (s / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing Middlebury files:\n  " + "\n  ".join(missing))
    return scenes


class Middlebury2014Dataset(Dataset):
    """RGB and metric depth pairs from Middlebury 2014 scenes."""

    def __init__(self, root: str | Path, downscale: float = 4) -> None:
        """Discover scenes. Images are divided in size by ``downscale`` when loaded."""
        if downscale < 1:
            raise ValueError(f"downscale must be >= 1, got {downscale}")
        self.scenes = find_scenes(root)
        self.downscale = downscale

    def __len__(self) -> int:
        """Return the number of scenes."""
        return len(self.scenes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return rgb (3,H,W) in [0,1], depth (1,H,W) in metres and valid (1,H,W) bool.

        Depth is computed at full resolution and then area-averaged over valid pixels.
        Focal length therefore needs no adjustment for the downscale.
        """
        scene = self.scenes[index]
        image = cv2.imread(str(scene / "im0.png"), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read {scene / 'im0.png'}")
        rgb = image[:, :, ::-1].astype(np.float32) / 255.0
        depth, valid = disparity_to_depth(read_pfm(scene / "disp0.pfm"), parse_calib(scene / "calib.txt"))
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"{scene}: image {rgb.shape[:2]} and disparity {depth.shape} differ")
        if self.downscale != 1:
            size_hw = (round(depth.shape[0] / self.downscale), round(depth.shape[1] / self.downscale))
            rgb = cv2.resize(rgb, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)
            depth, valid = masked_area_resize(depth, valid, size_hw)
        return {
            "rgb": torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
            "depth": torch.from_numpy(depth)[None],
            "valid": torch.from_numpy(valid)[None],
            "name": scene.name,
        }


def build_middlebury_dataset(config: dict[str, Any]) -> Middlebury2014Dataset:
    """Build the dataset using the paths and downscale factor in the config."""
    settings = config["datasets"]["middlebury"]
    return Middlebury2014Dataset(
        resolve_dataset_root(config) / settings["root"], downscale=settings["downscale"]
    )
