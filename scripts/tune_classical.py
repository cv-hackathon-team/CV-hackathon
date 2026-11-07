"""Grid-search the guided filter and joint bilateral settings on the NYU val split at x4.

Only the val split is ever loaded; the dataset guard rejects test without ``allow_test``.
Settings are searched in pixels at x4 and saved per scale in ``configs/default.yaml``,
so x2 and x8 use the same settings scaled in proportion. They were not tuned separately.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.classical.guided_filter import guided_upsample  # noqa: E402
from src.classical.interpolation import upsample_bicubic, upsample_bilinear  # noqa: E402
from src.classical.joint_bilateral import joint_bilateral_upsample  # noqa: E402
from src.datasets.corruption import corrupt_depth, get_corruption_params, make_rng  # noqa: E402
from src.datasets.nyu import build_nyu_dataset  # noqa: E402
from src.evaluation.evaluate import score  # noqa: E402
from src.utils import load_config, mod_crop, resolve_project_path  # noqa: E402

SCALE = 4
GUIDED_RADII_AT_X4 = [1, 2, 4, 8, 12, 16]
GUIDED_EPSILONS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
JBF_SIGMA_SPATIAL_AT_X4 = [1, 2, 4, 8]
JBF_SIGMA_RANGE = [0.02, 0.05, 0.1, 0.2]
# Lower is better for errors, higher for the rest.
CRITERIA = {"rmse_m": True, "mae_m": True, "psnr_db": False, "ssim": False, "boundary_f": False}


def candidates() -> list[dict[str, Any]]:
    """Return every grid setting, plus interpolation baselines for context."""
    grid: list[dict[str, Any]] = [
        {"method": "bilinear", "run": lambda d, v, g: upsample_bilinear(d, SCALE, v)},
        {"method": "bicubic", "run": lambda d, v, g: upsample_bicubic(d, SCALE, v)},
    ]
    for radius in GUIDED_RADII_AT_X4:
        for epsilon in GUIDED_EPSILONS:
            grid.append({
                "method": "guided_filter", "radius": radius, "epsilon": epsilon,
                "run": lambda d, v, g, r=radius, e=epsilon: guided_upsample(d, g, SCALE, r / SCALE, e, v),
            })
    for sigma_spatial in JBF_SIGMA_SPATIAL_AT_X4:
        for sigma_range in JBF_SIGMA_RANGE:
            grid.append({
                "method": "joint_bilateral", "sigma_spatial": sigma_spatial, "sigma_range": sigma_range,
                "run": lambda d, v, g, s=sigma_spatial, c=sigma_range: joint_bilateral_upsample(
                    d, g, SCALE, s / SCALE, c, v
                ),
            })
    return grid


def run_grid(config: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """Score every setting on val and return one row per setting, averaged over images."""
    dataset = build_nyu_dataset(config, "val")
    params = get_corruption_params(config)
    seed = config["corruption"]["seed"]
    largest = max(config["scale_factors"])
    grid = candidates()
    totals: list[list[dict[str, float]]] = [[] for _ in grid]
    indices = range(len(dataset) if limit is None else min(limit, len(dataset)))
    for index in indices:
        sample = dataset[index]
        rgb = mod_crop(sample["rgb"].numpy().transpose(1, 2, 0), largest).copy()
        target = mod_crop(sample["depth"][0].numpy(), largest)
        mask = mod_crop(sample["valid"][0].numpy(), largest)
        corrupted = corrupt_depth(target, mask, SCALE, "lr", params, make_rng(seed, index, SCALE, "lr"))
        for setting, scores in zip(grid, totals):
            prediction = setting["run"](corrupted.lr_depth, corrupted.validity_mask, rgb)
            scores.append(score(prediction, target, mask, config))
        print(f"  [val] {index + 1}/{len(indices)} {sample['name']}", flush=True)
    rows = []
    for setting, scores in zip(grid, totals):
        settings = {k: v for k, v in setting.items() if k != "run"}
        rows.append({**settings, "n_images": len(scores), **pd.DataFrame(scores).mean().to_dict()})
    return pd.DataFrame(rows)


def best_setting(results: pd.DataFrame, method: str, criterion: str) -> pd.Series:
    """Return the best row for a method under the given criterion."""
    rows = results[results["method"] == method]
    return rows.sort_values(criterion, ascending=CRITERIA[criterion]).iloc[0]


def replace_config_value(text: str, key: str, value: float) -> str:
    """Replace one ``key: value`` line in YAML text, keeping comments and layout."""
    updated, count = re.subn(rf"(?m)^(\s*{key}:\s*)[^\s#]+", rf"\g<1>{value!r}", text)
    if count != 1:
        raise ValueError(f"Expected exactly one {key!r} in the config, found {count}")
    return updated


def save_best(config_path: Path, guided: pd.Series, jbf: pd.Series, criterion: str) -> None:
    """Write the chosen settings to the config as per-scale values."""
    text = config_path.read_text(encoding="utf-8")
    text = replace_config_value(text, "radius_per_scale", float(guided["radius"]) / SCALE)
    text = replace_config_value(text, "epsilon", float(guided["epsilon"]))
    text = replace_config_value(text, "sigma_spatial_per_scale", float(jbf["sigma_spatial"]) / SCALE)
    text = replace_config_value(text, "sigma_range", float(jbf["sigma_range"]))
    note = f"# Filter sizes grow with the scale factor. Tuned on NYU val at x4 for {criterion}: scripts/tune_classical.py\n"
    text, count = re.subn(r"(?m)^# Filter sizes grow with the scale factor\..*\n", note, text)
    if count != 1:
        raise ValueError("Could not find the classical settings comment in the config")
    config_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--criterion", choices=sorted(CRITERIA), default="rmse_m")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    return parser.parse_args()


def main() -> None:
    """Run the grid search, write all results, and save the best settings to the config."""
    args = parse_args()
    config = load_config(args.config)
    results = run_grid(config)
    output = resolve_project_path(config["paths"]["outputs_dir"]) / "tuning_val_x4.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False, float_format="%.5f")
    guided = best_setting(results, "guided_filter", args.criterion)
    jbf = best_setting(results, "joint_bilateral", args.criterion)
    save_best(Path(args.config), guided, jbf, args.criterion)
    columns = ["method", "radius", "epsilon", "sigma_spatial", "sigma_range", "n_images", *CRITERIA]
    print(results[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nBest by {args.criterion}:\n{pd.DataFrame([guided, jbf])[columns].to_string(index=False)}")
    print(f"Wrote {output} and {args.config}")


if __name__ == "__main__":
    main()
