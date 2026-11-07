"""Measure latency and memory of the final pipeline on one real NYU val image.

Each device and scale runs in a fresh process, so peak process memory is not carried over.
Usage, from the project root: python scripts/benchmark.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.corruption import corrupt_depth, get_corruption_params, make_rng  # noqa: E402
from src.datasets.nyu import build_nyu_dataset  # noqa: E402
from src.evaluation.metrics import measure_latency, measure_memory  # noqa: E402
from src.inference import Predictor  # noqa: E402
from src.models.depthguard_net import DepthGuardNet, count_parameters  # noqa: E402
from src.utils import load_config, mod_crop, resolve_project_path  # noqa: E402

REPEATS = 20


def measure(device: str, scale: int) -> dict[str, object]:
    """Return latency and memory for one device and scale, measured on an lr input."""
    config = load_config()
    sample = build_nyu_dataset(config, "val")[0]
    rgb = mod_crop(sample["rgb"].numpy().transpose(1, 2, 0), 8).copy()
    depth = mod_crop(sample["depth"][0].numpy(), 8)
    valid = mod_crop(sample["valid"][0].numpy(), 8)
    corrupted = corrupt_depth(depth, valid, scale, "lr", get_corruption_params(config), make_rng(0, 0, scale, "lr"))
    predictor = Predictor(device=torch.device(device), config=config)
    run = lambda: predictor.predict(rgb, corrupted.lr_depth, scale)  # noqa: E731
    memory = measure_memory(run)
    latency = measure_latency(run, repeats=REPEATS)
    steps = run()["timings"]
    return {
        "device": device, "scale": scale, "height": rgb.shape[0], "width": rgb.shape[1], "repeats": REPEATS,
        **latency, **memory, **{f"last_{key}": value for key, value in steps.items()},
    }


def main() -> None:
    """Run one child process per device and scale, and write outputs/benchmark.csv."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", nargs=2, metavar=("DEVICE", "SCALE"))
    args = parser.parse_args()
    if args.child:
        print("RESULT " + json.dumps(measure(args.child[0], int(args.child[1]))))
        return
    config = load_config()
    devices = (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if torch.backends.mps.is_available() else []) + ["cpu"]
    rows = []
    for device in devices:
        for scale in config["scale_factors"]:
            output = subprocess.run(
                [sys.executable, __file__, "--child", device, str(scale)], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
            ).stdout
            rows.append(json.loads(next(line for line in output.splitlines() if line.startswith("RESULT "))[7:]))
            print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    frame.insert(0, "parameters", count_parameters(DepthGuardNet(config["model"]["channels"], config["model"]["blocks_per_level"])))
    frame.to_csv(resolve_project_path(config["paths"]["outputs_dir"]) / "benchmark.csv", index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
