"""Plot the effective HUGG ARIA weight-10 training trajectory.

When a run is resumed from an epoch checkpoint, later CSV rows replace any
overlapping rows from an interrupted, unsaved branch of the original run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = [
    "loss",
    "loss_ema",
    "unweighted_loss",
    "hand_region_loss",
    "background_region_loss",
    "hand_latent_fraction",
    "validation_loss",
    "validation_unweighted_loss",
    "validation_hand_region_loss",
    "validation_background_region_loss",
    "validation_hand_latent_fraction",
    "learning_rate",
    "grad_norm",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch-csv", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=15600)
    parser.add_argument("--steps-per-epoch", type=int, default=1560)
    parser.add_argument("--resume-step", type=int, default=21840)
    args = parser.parse_args()

    raw = pd.read_csv(args.log)
    for column in ["total_step", *METRIC_COLUMNS]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    # The continuation rows occur later in the appended CSV. Keeping the final
    # occurrence yields the checkpoint-backed path instead of the abandoned
    # post-checkpoint branch from the interrupted run.
    data = (
        raw.loc[raw["total_step"] > args.start_step]
        .dropna(subset=["total_step"])
        .drop_duplicates(subset="total_step", keep="last")
        .sort_values("total_step")
        .copy()
    )
    data["weight10_epoch"] = (
        ((data["total_step"] - args.start_step - 1) // args.steps_per_epoch) + 1
    ).astype(int)

    train_columns = [
        "loss",
        "unweighted_loss",
        "hand_region_loss",
        "background_region_loss",
        "hand_latent_fraction",
    ]
    epoch = data.groupby("weight10_epoch", as_index=False)[train_columns].mean()
    validation = data.dropna(subset=["validation_loss"])[
        [
            "weight10_epoch",
            "total_step",
            "validation_loss",
            "validation_unweighted_loss",
            "validation_hand_region_loss",
            "validation_background_region_loss",
            "validation_hand_latent_fraction",
        ]
    ]
    epoch = epoch.merge(validation, on="weight10_epoch", how="left")
    epoch.to_csv(args.epoch_csv, index=False)

    rolling = data[train_columns + ["grad_norm"]].rolling(200, min_periods=20).mean()
    x = data["total_step"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
    fig.suptitle(
        "HUGG ARIA Hand Restoration — Hand Weight 10 (Effective 10-Epoch Path)",
        fontsize=18,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(x, data["loss"], color="#5DA5DA", alpha=0.10, linewidth=0.6, label="batch loss")
    ax.plot(x, data["loss_ema"], color="#D62728", linewidth=1.5, label="logged EMA")
    ax.plot(x, rolling["loss"], color="#1F77B4", linewidth=1.3, label="rolling mean (200)")
    ax.set_title("Weighted training loss")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, rolling["hand_region_loss"], label="hand", color="#E45756")
    ax.plot(x, rolling["background_region_loss"], label="background", color="#4C78A8")
    ax.plot(x, rolling["unweighted_loss"], label="unweighted total", color="#54A24B")
    ax.set_title("Regional training losses (rolling mean 200)")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(validation["weight10_epoch"], validation["validation_loss"], "o-", label="weighted")
    ax.plot(validation["weight10_epoch"], validation["validation_hand_region_loss"], "o-", label="hand")
    ax.plot(validation["weight10_epoch"], validation["validation_background_region_loss"], "o-", label="background")
    ax.plot(validation["weight10_epoch"], validation["validation_unweighted_loss"], "o-", label="unweighted")
    ax.set_title("Validation losses")
    ax.set_xlabel("weight-10 epoch")
    ax.set_xticks(range(1, int(data["weight10_epoch"].max()) + 1))
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(epoch["weight10_epoch"], epoch["loss"], "o-", label="weighted")
    ax.plot(epoch["weight10_epoch"], epoch["hand_region_loss"], "o-", label="hand")
    ax.plot(epoch["weight10_epoch"], epoch["background_region_loss"], "o-", label="background")
    ax.plot(epoch["weight10_epoch"], epoch["unweighted_loss"], "o-", label="unweighted")
    ax.set_title("Average training loss per epoch")
    ax.set_xlabel("weight-10 epoch")
    ax.set_xticks(range(1, int(data["weight10_epoch"].max()) + 1))
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(x, data["learning_rate"], color="#6F4EDE", label="learning rate")
    ax.set_title("Learning rate and gradient norm")
    ax.set_xlabel("total step")
    ax.set_ylabel("learning rate", color="#6F4EDE")
    ax.tick_params(axis="y", labelcolor="#6F4EDE")
    ax2 = ax.twinx()
    ax2.plot(x, rolling["grad_norm"], color="#F2A541", alpha=0.9, label="grad norm (rolling 200)")
    ax2.set_ylabel("gradient norm", color="#F2A541")
    ax2.tick_params(axis="y", labelcolor="#F2A541")

    ax = axes[1, 2]
    ax.plot(x, rolling["hand_latent_fraction"] * 100.0, color="#B279A2")
    ax.axhline(data["hand_latent_fraction"].mean() * 100.0, color="black", linestyle="--", linewidth=1, label="overall mean")
    ax.set_title("Hand-region latent coverage")
    ax.set_xlabel("total step")
    ax.set_ylabel("hand latent fraction (%)")
    ax.legend(fontsize=8)

    for row in axes:
        for ax in row:
            ax.grid(True, alpha=0.25)
            if ax not in (axes[0, 2], axes[1, 0]):
                ax.axvline(args.resume_step, color="#777777", linestyle="--", linewidth=1, alpha=0.8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"effective_rows={len(data)}")
    print(f"step_range={int(data.total_step.min())}-{int(data.total_step.max())}")
    print(f"validation_points={len(validation)}")
    print(f"saved_plot={args.output}")
    print(f"saved_epoch_csv={args.epoch_csv}")


if __name__ == "__main__":
    main()
