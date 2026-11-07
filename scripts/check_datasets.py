"""Verify dataset files exist, and print shapes, dtypes and depth ranges for 2 samples each.

The dataset root comes from ``paths.dataset_root`` in the config. Paths are never
guessed: NYU or Middlebury problems exit with status 1. KITTI is optional and skipped
with a message if its folder is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.datasets.kitti import build_kitti_dataset  # noqa: E402
from src.datasets.middlebury import build_middlebury_dataset, find_scenes  # noqa: E402
from src.datasets.nyu import (  # noqa: E402
    build_nyu_dataset,
    inspect_mat,
    load_official_splits,
    read_scene_names,
)
from src.utils import load_config, resolve_dataset_root  # noqa: E402


def describe_sample(sample: dict[str, Any]) -> None:
    """Print one sample's shapes, dtypes and valid depth range, raising if it looks wrong."""
    rgb, depth, valid = sample["rgb"], sample["depth"], sample["valid"]
    print(f"  {sample['name']}")
    print(f"    rgb   {tuple(rgb.shape)} {rgb.dtype} range [{rgb.min():.3f}, {rgb.max():.3f}]")
    if valid.any():
        print(f"    depth {tuple(depth.shape)} {depth.dtype} valid range "
              f"[{depth[valid].min():.3f}, {depth[valid].max():.3f}] m")
    print(f"    valid {tuple(valid.shape)} {valid.dtype} {valid.float().mean() * 100:.1f}% of pixels")
    if rgb.shape[0] != 3 or rgb.shape[1:] != depth.shape[1:] or depth.shape != valid.shape:
        raise ValueError(f"Inconsistent shapes in {sample['name']}")
    if rgb.min() < 0 or rgb.max() > 1:
        raise ValueError(f"RGB outside [0,1] in {sample['name']}")
    if not valid.any():
        raise ValueError(f"No valid depth in {sample['name']}")
    if not np.isfinite(depth[valid].numpy()).all():
        raise ValueError(f"Non-finite depth in {sample['name']}")


def check_nyu(dataset_root: Path, config: dict[str, Any]) -> None:
    """Check the NYU files and splits, and describe 2 samples."""
    settings = config["datasets"]["nyu"]
    mat_path = dataset_root / settings["mat_file"]
    splits_path = dataset_root / settings["splits_file"]
    for path in (mat_path, splits_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing NYU file: {path}")
        print(f"  found {path}")
    for key, (shape, dtype) in inspect_mat(mat_path).items():
        print(f"  on disk: {key} {shape} {dtype}")

    train_full, test = load_official_splits(splits_path)
    train = build_nyu_dataset(config, "train")
    val = build_nyu_dataset(config, "val")
    scenes = read_scene_names(mat_path)
    train_scenes = {scenes[i] for i in train.indices}
    val_scenes = {scenes[i] for i in val.indices}
    test_scenes = {scenes[i] for i in test}
    print(f"  splits: official train {len(train_full)} -> train {len(train)} ({len(train_scenes)} scenes) "
          f"+ val {len(val)} ({len(val_scenes)} scenes, {len(val) / len(train_full) * 100:.1f}%); "
          f"test {len(test)} ({len(test_scenes)} scenes)")
    overlaps = {
        "train/val images": np.intersect1d(train.indices, val.indices).size,
        "train/val scenes": len(train_scenes & val_scenes),
        "train/test scenes": len(train_scenes & test_scenes),
        "val/test scenes": len(val_scenes & test_scenes),
    }
    print(f"  overlaps: {overlaps}")
    if any(overlaps.values()):
        raise ValueError(f"Split leakage: {overlaps}")
    for index in (0, 1):
        describe_sample(train[index])


def check_middlebury(dataset_root: Path, config: dict[str, Any]) -> None:
    """Check each scene's files and the scene count, and describe 2 samples."""
    settings = config["datasets"]["middlebury"]
    scenes = find_scenes(dataset_root / settings["root"])
    print(f"  found {len(scenes)} scenes: {', '.join(s.name for s in scenes)}")
    if len(scenes) != settings["expected_scenes"]:
        raise ValueError(f"Expected {settings['expected_scenes']} scenes, found {len(scenes)}")
    dataset = build_middlebury_dataset(config)
    print(f"  downscale factor: {dataset.downscale}")
    for index in (0, 1):
        describe_sample(dataset[index])


def check_kitti(config: dict[str, Any]) -> None:
    """Describe 2 KITTI samples, or print a skip message if the folder is missing."""
    dataset = build_kitti_dataset(config)
    if dataset is not None:
        print(f"  found {len(dataset)} samples")
        for index in (0, 1):
            describe_sample(dataset[index])


def main() -> int:
    """Run all checks and return a non-zero exit code if any required dataset fails."""
    config = load_config()
    dataset_root = resolve_dataset_root(config)
    print(f"Dataset root: {dataset_root}")
    failures = []
    for name, check in (
        ("NYU Depth v2", lambda: check_nyu(dataset_root, config)),
        ("Middlebury 2014", lambda: check_middlebury(dataset_root, config)),
        ("KITTI", lambda: check_kitti(config)),
    ):
        print(f"\n[{name}]")
        try:
            check()
        except Exception as error:  # Reported below, so every dataset still gets checked.
            print(f"  FAILED: {error}")
            failures.append(name)
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nAll required datasets OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
