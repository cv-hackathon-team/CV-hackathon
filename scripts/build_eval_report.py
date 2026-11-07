"""Write eval_P4_TXX.md from the result CSVs, so that no number is typed by hand.

Usage, from the project root: python scripts/build_eval_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
REPORT = PROJECT_ROOT / "eval_P4_TXX.md"
sys.path.insert(0, str(PROJECT_ROOT))

METHODS = ["nearest", "bilinear", "bicubic", "joint_bilateral", "guided_filter", "dl", "hybrid"]
ABLATION = ["dl", "hybrid", "hybrid_with_fusion", "hybrid_no_keep", "hybrid_no_boundary"]
MODES = ["noisy", "holes", "sparse", "boundary", "mixed"]
ROWS = [("nyu", "all"), ("nyu", "measured"), ("middlebury", "all")]
METRICS = [
    ("RMSE (m)", "rmse_m", 4, min),
    ("MAE (m)", "mae_m", 4, min),
    ("PSNR (dB)", "psnr_db", 2, max),
    ("SSIM", "ssim", 4, max),
    ("Edge / boundary F-score", "boundary_f", 4, max),
]


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Return a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def metric_table(frame: pd.DataFrame, column: str, digits: int, best, methods: list[str]) -> str:
    """Return one row per dataset, mask and scale, with the best method in bold."""
    rows = []
    for dataset, mask in ROWS:
        for scale in sorted(frame.scale.unique()):
            part = frame[(frame.dataset == dataset) & (frame["mask"] == mask) & (frame.scale == scale)].set_index("method")[column]
            top = best(part[methods])
            cells = [f"**{part[m]:.{digits}f}**" if part[m] == top else f"{part[m]:.{digits}f}" for m in methods]
            rows.append([f"{dataset} / {mask} / x{scale}", *cells])
    return markdown_table(["Dataset / mask / scale", *methods], rows)


def corruption_table(frame: pd.DataFrame, column: str, digits: int, scale: int) -> str:
    """Return dl and hybrid per corruption mode at one scale."""
    rows = []
    for dataset, mask in ROWS:
        for method in ("dl", "hybrid"):
            cells = [
                f"{frame[(frame.dataset == dataset) & (frame['mask'] == mask) & (frame.scale == scale) & (frame.corruption == mode) & (frame.method == method)][column].iloc[0]:.{digits}f}"
                for mode in MODES
            ]
            rows.append([f"{dataset} / {mask}", method, *cells])
    return markdown_table(["Dataset / mask", "Method", *MODES], rows)


def main() -> None:
    """Build the report, keeping the USN line that is already in the file."""
    first_line = REPORT.read_text(encoding="utf-8").splitlines()[0]
    results = pd.read_csv(OUTPUTS / "results.csv")
    ablation = pd.read_csv(OUTPUTS / "results_ablation_test_final.csv")
    benchmark = pd.read_csv(OUTPUTS / "benchmark.csv")
    classical = results[results.method.isin(METHODS[:5]) & (results.corruption == "lr")]
    final = ablation[ablation.method.isin(["dl", "hybrid"])]
    clean = pd.concat([classical, final])
    corrupted = results[results.method.isin(["dl", "hybrid"]) & results.corruption.isin(MODES)]
    n_images = {d: int(clean[clean.dataset == d].n_images.iloc[0]) for d in ("nyu", "middlebury")}
    if len(corrupted) != 2 * 5 * 3 * 3:
        raise ValueError(f"Expected 90 corrupted rows, found {len(corrupted)}")

    out = [first_line, "", "# DepthGuard360 final metrics (PS4)", ""]
    out += [
        f"Test sets: NYU Depth v2 ({n_images['nyu']} images, official test split) and Middlebury 2014 ({n_images['middlebury']} scenes).",
        "NYU is scored on two masks: `all` (`depths` > 0, including inpainted pixels) and `measured` (`rawDepths` > 0, still compared with `depths`).",
        "`hybrid` is the final method: network, keep measured depth, then boundary refinement, with confidence fusion off. Fusion was turned off based on the NYU val ablation.",
        "Classical settings were tuned on NYU val at x4. Nothing was tuned on test. Hybrid thresholds are untuned defaults.",
        "Sections 1 to 5 use the `lr` input: area downsampling with no other corruption. The best method in each row is bold. Metrics are the mean per image.",
        "",
    ]
    for number, (title, column, digits, best) in enumerate(METRICS, 1):
        out += [f"## {number}. {title}", "", metric_table(clean, column, digits, best, METHODS), ""]
    out += [
        "Nearest scores high on boundary F at x2 and x4 because its stair-stepped edges fall close to true edges; its RMSE is high.",
        "",
        "## 6. Per-scale",
        "",
        "Final `hybrid`, `lr` input, against the best other method on RMSE at the same scale.",
        "",
    ]
    rows = []
    for dataset, mask in ROWS:
        for scale in (2, 4, 8):
            part = clean[(clean.dataset == dataset) & (clean["mask"] == mask) & (clean.scale == scale)].set_index("method")
            h = part.loc["hybrid"]
            other = part.drop(index="hybrid").rmse_m.idxmin()
            rows.append([f"{dataset} / {mask}", f"x{scale}", f"{h.rmse_m:.4f}", f"{h.mae_m:.4f}", f"{h.psnr_db:.2f}", f"{h.ssim:.4f}",
                         f"{h.boundary_f:.4f}", f"{other} {part.loc[other].rmse_m:.4f}"])
    out += [markdown_table(["Dataset / mask", "Scale", "RMSE", "MAE", "PSNR", "SSIM", "Boundary F", "Best other RMSE"], rows), ""]
    out += ["**Known weakness at x2:** bicubic has lower RMSE and higher PSNR than `dl` and `hybrid` on every dataset and mask at x2.", ""]
    out += ["### Corrupted inputs", "", "`dl` and `hybrid`, medium severity. Classical methods were not run on these inputs.", ""]
    for scale in (2, 4, 8):
        for title, column, digits, _ in (METRICS[0], METRICS[4]):
            out += [f"{title}, x{scale}", "", corruption_table(corrupted, column, digits, scale), ""]
    out += ["### Ablation", "", "Each variant changes one step of the final hybrid. `lr` input, RMSE (m).", "",
            metric_table(ablation, "rmse_m", 4, min, ABLATION), "", "Boundary F-score.", "",
            metric_table(ablation, "boundary_f", 4, max, ABLATION), ""]

    device_rows = [
        [row.device, f"x{row.scale}", f"{row.width}x{row.height}", f"{row.mean_ms:.1f}", f"{row.median_ms:.1f}", f"{row.std_ms:.1f}"]
        for row in benchmark.itertuples()
    ]
    out += ["## 7. Latency", "",
            f"`src/inference.py` `predict()` on one NYU val image, including diagnosis and refinement, mean of {benchmark.repeats.iloc[0]} runs after 3 warm-up runs. Apple M5 with 24 GB of memory.", "",
            markdown_table(["Device", "Scale", "Output", "Mean (ms)", "Median (ms)", "Std (ms)"], device_rows), ""]
    memory_rows = [[row.device, f"x{row.scale}", f"{row.process_peak_mb:.0f}", "n/a" if pd.isna(row.device_mb) or row.device == "cpu" else f"{row.device_mb:.0f}"]
                   for row in benchmark.itertuples()]
    out += ["## 8. Memory", "",
            f"Network parameters: {int(benchmark.parameters.iloc[0]):,}. Checkpoint file size: {(PROJECT_ROOT / 'checkpoints' / 'best.pt').stat().st_size / 1024**2:.1f} MB, which includes optimizer state.",
            "Each row is a fresh process. Peak process memory includes Python and the libraries. The MPS figure is the driver allocation after inference, not a peak.", "",
            markdown_table(["Device", "Scale", "Peak process (MB)", "MPS driver (MB)"], memory_rows), ""]
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
