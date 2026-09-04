#!/usr/bin/env python3
"""Plot one hand-restoration training CSV, including regional loss statistics."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Hand Restoration Training")
    args = parser.parse_args()

    data = pd.read_csv(args.log)
    required = {"total_step", "epoch", "loss", "loss_ema", "validation_loss"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Training log is empty")

    regional = {
        "unweighted_loss",
        "hand_region_loss",
        "background_region_loss",
        "hand_latent_fraction",
    }.issubset(data.columns)
    validation_regional = {
        "validation_unweighted_loss",
        "validation_hand_region_loss",
        "validation_background_region_loss",
    }.issubset(data.columns)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(args.title, fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(data.total_step, data.loss, alpha=0.24, linewidth=0.7, label="Weighted batch loss")
    ax.plot(data.total_step, data.loss_ema, color="crimson", linewidth=1.8, label="Weighted loss EMA")
    ax.set(title="Weighted training loss", xlabel="Optimizer step", ylabel="Loss")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    if regional:
        ax.plot(data.total_step, data.hand_region_loss, alpha=0.45, linewidth=0.8, label="Hand region")
        ax.plot(data.total_step, data.background_region_loss, alpha=0.45, linewidth=0.8, label="Background")
        ax.plot(data.total_step, data.unweighted_loss, alpha=0.7, linewidth=1.0, label="Unweighted total")
        ax.set(title="Regional training MSE", xlabel="Optimizer step", ylabel="MSE")
    else:
        epoch = data.groupby("epoch", sort=True).agg(
            mean_loss=("loss", "mean"), final_ema=("loss_ema", "last")
        )
        ax.plot(epoch.index + 1, epoch.mean_loss, "o-", linewidth=2, label="Mean batch loss")
        ax.plot(epoch.index + 1, epoch.final_ema, "s--", linewidth=1.6, label="Final EMA")
        ax.set(title="Loss by epoch", xlabel="Completed epoch", ylabel="Loss")
        ax.set_xticks(epoch.index + 1)
    ax.legend()
    ax.grid(alpha=0.25)

    validation = data.dropna(subset=["validation_loss"])
    ax = axes[1, 0]
    if validation.empty:
        ax.text(0.5, 0.5, "No validation points", ha="center", va="center")
    else:
        ax.plot(validation.total_step, validation.validation_loss, "o-", linewidth=2, label="Weighted")
        if validation_regional:
            ax.plot(validation.total_step, validation.validation_hand_region_loss, "o-", label="Hand region")
            ax.plot(validation.total_step, validation.validation_background_region_loss, "o-", label="Background")
            ax.plot(validation.total_step, validation.validation_unweighted_loss, "o-", label="Unweighted")
        ax.legend()
    ax.set(title="In-domain validation loss", xlabel="Optimizer step", ylabel="Loss")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(data.total_step, data.learning_rate, color="royalblue", linewidth=1.6, label="Learning rate")
    ax2.plot(data.total_step, data.grad_norm, color="darkorange", alpha=0.35, linewidth=0.7, label="Gradient norm")
    ax.set(title="Learning rate and gradient norm", xlabel="Optimizer step", ylabel="Learning rate")
    ax2.set_ylabel("Gradient norm")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.grid(alpha=0.25)

    final_validation = "n/a" if validation.empty else f"{validation.validation_loss.iloc[-1]:.6f}"
    fig.text(
        0.5,
        0.015,
        f"steps={len(data):,}  final EMA={data.loss_ema.iloc[-1]:.6f}  "
        f"final validation={final_validation}",
        ha="center",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
