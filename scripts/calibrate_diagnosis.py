"""Choose diagnosis thresholds on NYU val only. Fit on even images, report on odd images.

Usage, from the project root: python scripts/calibrate_diagnosis.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.corruption import MODES, corrupt_depth, get_corruption_params, make_rng  # noqa: E402
from src.datasets.nyu import build_nyu_dataset  # noqa: E402
from src.hybrid.diagnose import DiagnosisSettings, apply_rules, diagnose  # noqa: E402
from src.utils import load_config, mod_crop, resolve_project_path  # noqa: E402

TRUE_LABEL = {mode: ("clean" if mode == "lr" else mode) for mode in MODES}
GRID = {
    "clean_missing_percent": [0.5, 1, 2, 3, 5],
    "sparse_missing_percent": [30, 40, 50, 60, 70],
    "noise_level": [0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.0065, 0.008],
    "hole_blob_ratio": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}


def accuracy(frame: pd.DataFrame, settings: DiagnosisSettings) -> tuple[float, pd.Series]:
    """Return the mean per-class accuracy and the predicted labels."""
    labels = pd.Series(
        [apply_rules(h, n, s, settings) for h, n, s in zip(frame.hole_percent, frame.noise_level, frame.sparsity)],
        index=frame.index,
    )
    return float((labels == frame.true_label).groupby(frame.true_label).mean().mean()), labels


def main() -> None:
    """Measure statistics on val, grid-search on even images, and report on odd images."""
    config = load_config()
    dataset = build_nyu_dataset(config, "val")
    params = get_corruption_params(config)
    largest = max(config["scale_factors"])
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        depth = mod_crop(sample["depth"][0].numpy(), largest)
        valid = mod_crop(sample["valid"][0].numpy(), largest)
        for scale, mode in itertools.product(config["scale_factors"], MODES):
            corrupted = corrupt_depth(depth, valid, scale, mode, params, make_rng(config["corruption"]["seed"], index, scale, mode))
            result = diagnose(corrupted.lr_depth, corrupted.validity_mask)
            rows.append({"index": index, "scale": scale, "true_label": TRUE_LABEL[mode], "hole_percent": result.hole_percent,
                         "noise_level": result.noise_level, "sparsity": result.sparsity})
    frame = pd.DataFrame(rows)
    frame.to_csv(resolve_project_path(config["paths"]["outputs_dir"]) / "diagnosis_val.csv", index=False)
    print(frame.groupby(["true_label", "scale"])[["hole_percent", "noise_level", "sparsity"]].median().round(4).to_string())
    fit, report = frame[frame["index"] % 2 == 0], frame[frame["index"] % 2 == 1]
    best = max(
        (DiagnosisSettings(*values) for values in itertools.product(*GRID.values())),
        key=lambda settings: accuracy(fit, settings)[0],
    )
    fit_accuracy, _ = accuracy(fit, best)
    report_accuracy, labels = accuracy(report, best)
    print(f"\nBest: {best}\nmean per-class accuracy: fit {fit_accuracy:.3f}, held-out {report_accuracy:.3f}\n")
    print(pd.crosstab(report.true_label, labels).to_string())
    print((labels == report.true_label).groupby(report.scale).mean().round(3).to_string())


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
