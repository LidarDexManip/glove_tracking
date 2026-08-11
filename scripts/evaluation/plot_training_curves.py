#!/usr/bin/env python3
"""Plot train/validation loss from a hand-restoration training_log.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="Path to training_log.csv")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smooth", type=int, default=1000, help="Train-loss moving-average window")
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    averaged = (cumulative[window:] - cumulative[:-window]) / window
    return np.concatenate((np.full(window - 1, np.nan), averaged))


def draw_chart(canvas: np.ndarray, bounds: tuple[int, int, int, int],
               title: str, x_label: str, y_label: str,
               series: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int], str]]) -> None:
    left, top, right, bottom = bounds
    cv2.rectangle(canvas, (left, top), (right, bottom), (245, 245, 245), -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), (90, 90, 90), 1)
    finite_x = np.concatenate([x[np.isfinite(y)] for x, y, _, _ in series])
    finite_y = np.concatenate([y[np.isfinite(y)] for _, y, _, _ in series])
    x_min, x_max = float(finite_x.min()), float(finite_x.max())
    y_min, y_max = float(finite_y.min()), float(finite_y.max())
    margin = max((y_max - y_min) * 0.08, 1e-8)
    y_min, y_max = y_min - margin, y_max + margin
    for fraction in np.linspace(0, 1, 5):
        y = round(bottom - fraction * (bottom - top))
        value = y_min + fraction * (y_max - y_min)
        cv2.line(canvas, (left, y), (right, y), (218, 218, 218), 1)
        cv2.putText(canvas, f"{value:.4f}", (left - 78, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (75, 75, 75), 1, cv2.LINE_AA)
    for x_values, y_values, color, label in series:
        valid = np.isfinite(y_values)
        x = x_values[valid]
        y = y_values[valid]
        if not len(x):
            continue
        px = left + (x - x_min) / max(x_max - x_min, 1e-8) * (right - left)
        py = bottom - (y - y_min) / max(y_max - y_min, 1e-8) * (bottom - top)
        points = np.round(np.stack([px, py], axis=1)).astype(np.int32)
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        else:
            cv2.circle(canvas, tuple(points[0]), 4, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (left, top - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, x_label, (right - 105, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (left, top + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (60, 60, 60), 1, cv2.LINE_AA)
    legend_x = right - 235
    for index, (_, _, color, label) in enumerate(series):
        legend_y = top + 22 + index * 24
        cv2.line(canvas, (legend_x, legend_y), (legend_x + 30, legend_y), color, 3)
        cv2.putText(canvas, label, (legend_x + 38, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (40, 40, 40), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    steps, losses, emas = [], [], []
    val_steps, val_losses = [], []
    with args.log.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(row["total_step"])
            steps.append(step)
            losses.append(float(row["loss"]))
            emas.append(float(row["loss_ema"]))
            if row["validation_loss"]:
                val_steps.append(step)
                val_losses.append(float(row["validation_loss"]))

    steps_np = np.asarray(steps)
    losses_np = np.asarray(losses)
    smooth = moving_average(losses_np, min(args.smooth, len(losses_np)))
    output = args.output or args.log.with_name("loss_curves.png")

    canvas = np.full((900, 1400, 3), 255, np.uint8)
    draw_chart(
        canvas, (110, 70, 1350, 410), "Training loss", "optimizer step", "diffusion MSE",
        [(steps_np, smooth, (145, 74, 15), f"moving avg ({args.smooth})"),
         (steps_np, np.asarray(emas), (20, 105, 230), "logged EMA")],
    )
    if val_steps:
        draw_chart(
            canvas, (110, 510, 1350, 850), "Holdout validation loss", "optimizer step", "diffusion MSE",
            [(np.asarray(val_steps), np.asarray(val_losses), (69, 139, 35), "validation")],
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(output)
    print(f"saved={output} train_rows={len(steps)} validation_points={len(val_steps)}")
    if val_losses:
        best = int(np.argmin(val_losses))
        print(f"best_validation_loss={val_losses[best]:.8f} step={val_steps[best]}")
        print(f"final_validation_loss={val_losses[-1]:.8f} step={val_steps[-1]}")


if __name__ == "__main__":
    main()
