"""Train DepthGuardNet on NYU. Only the train and val splits are ever loaded.

Run from the project root, and add ``--smoke`` for a short check run:

    python src/training/train.py
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Sampler  # noqa: E402

from src.datasets.corruption import MODES, NetworkDataset, get_corruption_params  # noqa: E402
from src.datasets.nyu import build_nyu_dataset  # noqa: E402
from src.models.depthguard_net import DepthGuardNet, count_parameters  # noqa: E402
from src.training.losses import DepthGuardLoss  # noqa: E402
from src.utils import get_device, load_config, resolve_project_path  # noqa: E402

LOSS_TERMS = ("total", "depth", "gradient", "boundary")


def set_seed(seed: int) -> None:
    """Seed all random number generators. GPU results may still vary slightly between runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_init(_: int) -> None:
    """Stop OpenCV from starting its own threads inside each DataLoader worker."""
    cv2.setNumThreads(0)


class RandomScaleBatchSampler(Sampler):
    """Yield shuffled batches of (index, scale, epoch, None), with one random scale per batch."""

    def __init__(self, indices: list[int], batch_size: int, scales: list[int], seed: int) -> None:
        """Store the sampling settings. Incomplete final batches are dropped."""
        self.indices, self.batch_size, self.scales, self.seed = indices, batch_size, scales, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch, which determines the shuffle and scales."""
        self.epoch = epoch

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return len(self.indices) // self.batch_size

    def __iter__(self) -> Iterator[list[tuple[int, int, int, None]]]:
        """Yield this epoch's batches."""
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(self.indices)
        for start in range(len(self)):
            scale = int(rng.choice(self.scales))
            batch = order[start * self.batch_size : (start + 1) * self.batch_size]
            yield [(int(i), scale, self.epoch, None) for i in batch]


def validation_items(indices: list[int], scales: list[int]) -> list[tuple[int, int, int, str]]:
    """Return a fixed val set: every image at every scale, cycling through corruption modes."""
    return [
        (index, scale, 0, MODES[(position + offset) % len(MODES)])
        for position, index in enumerate(indices)
        for offset, scale in enumerate(scales)
    ]


def build_dataloaders(config: dict[str, Any], smoke: bool) -> tuple[DataLoader, DataLoader, RandomScaleBatchSampler]:
    """Build the train and val loaders, and return the train sampler for epoch updates."""
    settings = config["train"]
    train_base = build_nyu_dataset(config, "train")
    val_base = build_nyu_dataset(config, "val")
    params = get_corruption_params(config)
    seed = settings["seed"]
    train_data = NetworkDataset(
        train_base, config["corruption"]["modes"], params, seed,
        crop_size_hw=tuple(settings["crop_size"]), flip=True,
    )
    val_data = NetworkDataset(val_base, config["corruption"]["modes"], params, seed)
    train_count = settings["smoke_images"] if smoke else len(train_data)
    val_count = settings["smoke_val_images"] if smoke else len(val_data)
    sampler = RandomScaleBatchSampler(
        list(range(train_count)), settings["batch_size"], config["scale_factors"], seed
    )
    workers = settings["num_workers"]
    common = {"num_workers": workers, "worker_init_fn": worker_init, "persistent_workers": workers > 0}
    train_loader = DataLoader(train_data, batch_sampler=sampler, **common)
    val_loader = DataLoader(
        val_data,
        sampler=validation_items(list(range(val_count)), config["scale_factors"]),
        batch_size=4,
        **common,
    )
    return train_loader, val_loader, sampler


def _forward(model: nn.Module, batch: dict[str, Any], device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """Move a batch to the device and run the model."""
    output = model(batch["rgb"].to(device), batch["depth_input"].to(device), batch["input_valid"].to(device))
    return output, batch["target"].to(device), batch["target_valid"].to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> dict[str, float]:
    """Train for one epoch and return average losses, raising on a non-finite loss."""
    model.train()
    totals = dict.fromkeys(LOSS_TERMS, 0.0)
    for batch in loader:
        output, target, mask = _forward(model, batch, device)
        terms = criterion(output, target, mask)
        if not torch.isfinite(terms["total"]):
            raise RuntimeError(f"Non-finite loss: { {k: float(v) for k, v in terms.items()} }")
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        for name in LOSS_TERMS:
            totals[name] += terms[name].item()
    return {name: value / len(loader) for name, value in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Return average val losses, plus RMSE for the network and its bicubic input.

    RMSE is computed per image over valid pixels and then averaged, as in evaluation.
    """
    model.eval()
    totals = dict.fromkeys(LOSS_TERMS, 0.0)
    errors: list[float] = []
    input_errors: list[float] = []
    for batch in loader:
        output, target, mask = _forward(model, batch, device)
        terms = criterion(output, target, mask)
        for name in LOSS_TERMS:
            totals[name] += terms[name].item()
        depth_input = batch["depth_input"].to(device)
        for i in range(target.shape[0]):
            m = mask[i]
            errors.append(((output.depth[i][m] - target[i][m]) ** 2).mean().sqrt().item())
            input_errors.append(((depth_input[i][m] - target[i][m]) ** 2).mean().sqrt().item())
    results = {name: value / len(loader) for name, value in totals.items()}
    results["rmse_m"] = float(np.mean(errors))
    results["input_rmse_m"] = float(np.mean(input_errors))
    return results


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    """Save the model, optimizer and scheduler state and the config used."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "metrics": metrics, "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def plot_curves(history: list[dict[str, float]], path: Path) -> None:
    """Save small multiples of each loss term, plus val RMSE against the bicubic input."""
    train_colour, val_colour = "#2a78d6", "#eb6834"
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.4), layout="constrained")
    for axis, name in zip(axes.flat, LOSS_TERMS):
        axis.plot(epochs, [r[f"train_{name}"] for r in history], color=train_colour, lw=2, marker="o", ms=4, label="Train")
        axis.plot(epochs, [r[f"val_{name}"] for r in history], color=val_colour, lw=2, marker="s", ms=4, label="Val")
        axis.set_title(f"{name.capitalize()} loss", loc="left", fontsize=11)
        axis.legend(frameon=False)
    rmse_axis = axes.flat[4]
    rmse_axis.plot(epochs, [r["val_rmse_m"] for r in history], color=val_colour, lw=2, marker="s", ms=4, label="Network")
    rmse_axis.plot(epochs, [r["val_input_rmse_m"] for r in history], color="#6b6b6b", lw=2, ls="--", label="Bicubic input")
    rmse_axis.set_title("Val RMSE (m)", loc="left", fontsize=11)
    rmse_axis.legend(frameon=False)
    lr_axis = axes.flat[5]
    lr_axis.plot(epochs, [r["lr"] for r in history], color=train_colour, lw=2)
    lr_axis.set_title("Learning rate", loc="left", fontsize=11)
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.get_major_locator().set_params(integer=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def write_history(history: list[dict[str, float]], path: Path) -> None:
    """Write one row per epoch to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(config: dict[str, Any], smoke: bool = False) -> list[dict[str, float]]:
    """Run the training loop with validation, checkpoints, curves and early stopping."""
    settings = config["train"]
    set_seed(settings["seed"])
    device = get_device()
    epochs = settings["smoke_epochs"] if smoke else settings["epochs"]
    subfolder = "smoke" if smoke else ""
    prefix = "smoke_" if smoke else ""
    checkpoints = resolve_project_path(config["paths"]["checkpoints_dir"]) / subfolder
    outputs = resolve_project_path(config["paths"]["outputs_dir"])
    outputs.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, sampler = build_dataloaders(config, smoke)
    model = DepthGuardNet(config["model"]["channels"], config["model"]["blocks_per_level"]).to(device)
    criterion = DepthGuardLoss(
        settings["loss_weights"], settings["laplace_reference_m"], settings["boundary_edge_threshold"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f"device={device} params={count_parameters(model):,} epochs={epochs} "
          f"train_batches={len(train_loader)} val_samples={len(val_loader.sampler)} smoke={smoke}", flush=True)

    history: list[dict[str, float]] = []
    best_rmse = float("inf")
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        sampler.set_epoch(epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        train_terms = train_one_epoch(model, train_loader, criterion, optimizer, device, settings["grad_clip_norm"])
        train_seconds = time.perf_counter() - start
        val_terms = validate(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch, "lr": learning_rate,
            **{f"train_{k}": v for k, v in train_terms.items()},
            **{f"val_{k}": v for k, v in val_terms.items()},
            "train_seconds": train_seconds, "epoch_seconds": time.perf_counter() - start,
        }
        history.append(row)
        improved = val_terms["rmse_m"] < best_rmse
        print(
            f"epoch {epoch}/{epochs} | "
            + " ".join(f"train_{k}={train_terms[k]:.4f}" for k in LOSS_TERMS) + " | "
            + " ".join(f"val_{k}={val_terms[k]:.4f}" for k in LOSS_TERMS)
            + f" | val_rmse={val_terms['rmse_m']:.4f} input_rmse={val_terms['input_rmse_m']:.4f}"
            + f" | lr={learning_rate:.2e} time={row['epoch_seconds']:.1f}s{' best' if improved else ''}",
            flush=True,
        )
        save_checkpoint(checkpoints / "last.pt", model, optimizer, scheduler, epoch, val_terms, config)
        if improved:
            best_rmse = val_terms["rmse_m"]
            stale_epochs = 0
            save_checkpoint(checkpoints / "best.pt", model, optimizer, scheduler, epoch, val_terms, config)
        else:
            stale_epochs += 1
        write_history(history, outputs / f"{prefix}train_history.csv")
        plot_curves(history, outputs / f"{prefix}loss_curves.png")
        if stale_epochs >= settings["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}: no val RMSE improvement for {stale_epochs} epochs.", flush=True)
            break
    print(f"TRAINING COMPLETE best_val_rmse={best_rmse:.4f} epochs_run={len(history)}", flush=True)
    return history


def main() -> None:
    """Parse options and train."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="short check run")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    train(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
