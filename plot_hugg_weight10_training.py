"""Plot the complete HUGG Aria weight-5 to weight-10 training trajectory.

For appended resumed logs, the final row for each total step is retained. This
removes the abandoned post-checkpoint branch from an interrupted run while
keeping the effective continuation path.
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


def load_log(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["total_step"] = pd.to_numeric(data["total_step"], errors="coerce")
    for column in METRIC_COLUMNS:
        if column not in data:
            data[column] = float("nan")
        else:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["total_step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weight10_log", type=Path)
    parser.add_argument("--base-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch-csv", type=Path, required=True)
    parser.add_argument("--weight10-start-step", type=int, default=15600)
    parser.add_argument("--steps-per-epoch", type=int, default=1560)
    parser.add_argument("--resume-step", type=int, default=21840)
    args = parser.parse_args()

    weight10 = (
        load_log(args.weight10_log)
        .loc[lambda frame: frame["total_step"] > args.weight10_start_step]
        .drop_duplicates(subset="total_step", keep="last")
    )
    if args.base_log:
        base = (
            load_log(args.base_log)
            .loc[lambda frame: frame["total_step"] <= args.weight10_start_step]
            .drop_duplicates(subset="total_step", keep="last")
        )
        data = pd.concat([base, weight10], ignore_index=True)
    else:
        data = weight10.copy()

    data = data.sort_values("total_step").reset_index(drop=True)
    data["global_epoch"] = (
        ((data["total_step"] - 1) // args.steps_per_epoch) + 1
    ).astype(int)
    data["hand_loss_weight"] = data["total_step"].le(
        args.weight10_start_step
    ).map({True: 5, False: 10})
    data["phase"] = data["hand_loss_weight"].map(
        {5: "hand weight 5", 10: "hand weight 10"}
    )

    train_columns = [
        "loss",
        "unweighted_loss",
        "hand_region_loss",
        "background_region_loss",
        "hand_latent_fraction",
    ]
    epoch = data.groupby(
        ["global_epoch", "phase", "hand_loss_weight"], as_index=False
    )[train_columns].mean()
    validation = data.dropna(subset=["validation_loss"])[
        [
            "global_epoch",
            "total_step",
            "validation_loss",
            "validation_unweighted_loss",
            "validation_hand_region_loss",
            "validation_background_region_loss",
            "validation_hand_latent_fraction",
        ]
    ]
    epoch = epoch.merge(validation, on="global_epoch", how="left")
    epoch.to_csv(args.epoch_csv, index=False)

    rolling = data[train_columns + ["grad_norm"]].rolling(
        200, min_periods=20
    ).mean()
    x = data["total_step"]
    region_data = data["hand_region_loss"].notna()
    region_validation = validation["validation_hand_region_loss"].notna()

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
    first_epoch = int(data["global_epoch"].min())
    final_epoch = int(data["global_epoch"].max())
    fig.suptitle(
        f"HUGG Aria Hand Restoration — Effective Epochs {first_epoch}–{final_epoch} "
        "(Hand Weight 5 → 10)",
        fontsize=18,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(
        x, data["loss"], color="#5DA5DA", alpha=0.10,
        linewidth=0.6, label="batch weighted loss",
    )
    ax.plot(
        x, rolling["loss"], color="#D62728", linewidth=1.5,
        label="continuous rolling mean (200)",
    )
    ax.set_title("Weighted training loss (objective changes at step 15600)")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(
        validation["global_epoch"], validation["validation_loss"],
        "o-", color="#2CA02C", label="weighted validation",
    )
    ax.set_title("Weighted validation loss — connected full path")
    ax.set_xlabel("global epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(range(first_epoch, final_epoch + 1))
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(epoch["global_epoch"], epoch["loss"], "o-", color="#9467BD")
    ax.set_title("Average weighted training loss per epoch")
    ax.set_xlabel("global epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(range(first_epoch, final_epoch + 1))
    ax.tick_params(axis="x", labelrotation=45)

    ax = axes[1, 0]
    ax.plot(
        data.loc[region_data, "total_step"],
        rolling.loc[region_data, "hand_region_loss"],
        label="hand", color="#E45756",
    )
    ax.plot(
        data.loc[region_data, "total_step"],
        rolling.loc[region_data, "background_region_loss"],
        label="background", color="#4C78A8",
    )
    ax.plot(
        data.loc[region_data, "total_step"],
        rolling.loc[region_data, "unweighted_loss"],
        label="unweighted total", color="#54A24B",
    )
    ax.set_title("Regional training losses (logged for weight-10 phase)")
    ax.set_xlabel("total step")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(x, data["learning_rate"], color="#6F4EDE")
    ax.set_title("Learning rate and gradient norm")
    ax.set_xlabel("total step")
    ax.set_ylabel("learning rate", color="#6F4EDE")
    ax.tick_params(axis="y", labelcolor="#6F4EDE")
    ax2 = ax.twinx()
    ax2.plot(
        x, rolling["grad_norm"], color="#F2A541", alpha=0.9,
        label="grad norm (rolling 200)",
    )
    ax2.set_ylabel("gradient norm", color="#F2A541")
    ax2.tick_params(axis="y", labelcolor="#F2A541")

    ax = axes[1, 2]
    ax.plot(
        validation.loc[region_validation, "global_epoch"],
        validation.loc[region_validation, "validation_hand_region_loss"],
        "o-", label="hand", color="#E45756",
    )
    ax.plot(
        validation.loc[region_validation, "global_epoch"],
        validation.loc[region_validation, "validation_background_region_loss"],
        "o-", label="background", color="#4C78A8",
    )
    ax.plot(
        validation.loc[region_validation, "global_epoch"],
        validation.loc[region_validation, "validation_unweighted_loss"],
        "o-", label="unweighted", color="#54A24B",
    )
    ax.set_title("Regional validation losses (weight-10 phase)")
    ax.set_xlabel("global epoch")
    ax.set_ylabel("MSE")
    ax.set_xticks(range(11, final_epoch + 1))
    ax.legend(fontsize=8)

    step_axes = (axes[0, 0], axes[1, 0], axes[1, 1])
    for ax in step_axes:
        ax.axvline(
            args.weight10_start_step, color="#C44E52", linestyle="--",
            linewidth=1.2, alpha=0.9,
        )
        ax.axvline(
            args.resume_step, color="#777777", linestyle=":",
            linewidth=1.2, alpha=0.9,
        )
    for ax in (axes[0, 1], axes[0, 2], axes[1, 2]):
        ax.axvline(10.5, color="#C44E52", linestyle="--", linewidth=1.2)

    for row in axes:
        for ax in row:
            ax.grid(True, alpha=0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"effective_rows={len(data)}")
    print(
        f"step_range={int(data.total_step.min())}-"
        f"{int(data.total_step.max())}"
    )
    print(f"epochs={first_epoch}-{final_epoch}")
    print(f"validation_points={len(validation)}")
    print(f"saved_plot={args.output}")
    print(f"saved_epoch_csv={args.epoch_csv}")


if __name__ == "__main__":
    main()
