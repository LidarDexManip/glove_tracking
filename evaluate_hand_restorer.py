#!/usr/bin/env python3
"""Reproducible random-sample evaluation on the sequence-disjoint holdout set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from hand_restoration.conditions import ConditionConfig
from hand_restoration.derived_dataset import DerivedHandRestorationDataset
from hand_restoration.inference import build_restorer, run_inference, save_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--indices",
        help="Comma-separated dataset indices; overrides random --samples selection.",
    )
    parser.add_argument("--split", choices=("training", "holdout"), default="holdout")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def masked_values(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    return image.reshape(-1) if mask is None else image[mask > 0]


def pixel_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    diff = masked_values(pred - target, mask).astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(np.square(diff)))
    return {"mae": mae, "rmse": float(np.sqrt(mse)), "psnr_db": float("inf") if mse == 0 else float(-10 * np.log10(mse))}


def ssim_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Standard 11x11 Gaussian-window SSIM for images normalized to [0, 1].
    c1, c2 = 0.01**2, 0.03**2
    x, y = pred.astype(np.float32), target.astype(np.float32)
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )


def all_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    structural = np.mean(ssim_map(pred, target), axis=2)
    result = {}
    for scope, selected_mask in (("full", None), ("mask", mask)):
        for name, value in pixel_metrics(pred, target, selected_mask).items():
            result[f"{scope}_{name}"] = value
        values = structural.reshape(-1) if selected_mask is None else structural[selected_mask > 0]
        result[f"{scope}_ssim"] = float(np.mean(values))
    return result


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def save_comparison(path: Path, condition: np.ndarray, generated: np.ndarray, target: np.ndarray) -> None:
    panels = []
    for label, image in (
        ("CONDITION INPUT", condition),
        ("RAW DIFFUSION OUTPUT", generated),
        ("TARGET", target),
    ):
        panel = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (0, 0, 0), thickness=-1)
        cv2.putText(panel, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.concatenate(panels, axis=1)):
        raise OSError(f"Failed to save comparison image: {path}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    data = config["data"]
    if data.get("format") != "derived_webdataset":
        raise ValueError("This evaluator currently requires format=derived_webdataset.")
    manifest_key = "train_manifest" if args.split == "training" else "val_manifest"
    manifest = resolve_path(project_root, data[manifest_key])
    dataset = DerivedHandRestorationDataset(
        manifest,
        output_size=data.get("output_size", 512),
        grayscale=data.get("grayscale", True),
        condition=ConditionConfig(**config.get("condition", {})),
        seed=config.get("seed", 0),
        include_numpy=True,
        mask_member=data.get("mask_member", "mask.png"),
    )
    if args.indices:
        indices = np.asarray([int(value) for value in args.indices.split(",")], dtype=np.int64)
        if not len(indices) or len(set(indices.tolist())) != len(indices):
            raise ValueError("--indices must contain distinct dataset indices")
        if np.any(indices < 0) or np.any(indices >= len(dataset)):
            raise ValueError(f"--indices must be between 0 and {len(dataset) - 1}")
    else:
        if not 1 <= args.samples <= len(dataset):
            raise ValueError(f"--samples must be between 1 and {len(dataset)}")
        indices = np.sort(np.random.default_rng(args.seed).choice(len(dataset), args.samples, replace=False))
    checkpoint = args.checkpoint or args.run_dir / "controlnet_final.pt"
    output = args.output or args.run_dir / f"{args.split}_eval_n{args.samples}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)

    restorer = build_restorer(config, args.device)
    restorer.load_controlnet(checkpoint)
    restorer.controlnet.eval()

    rows = []
    for ordinal, index in enumerate(indices):
        sample = dataset[int(index)]
        result = run_inference(restorer, sample, config, steps=args.steps, seed=args.seed + ordinal)
        metadata = sample["metadata"]
        prefix = f"{ordinal:03d}_{metadata['sequence_id']}_{int(metadata['frame_id']):06d}"
        row = {"sample_index": int(index), "sequence_id": metadata["sequence_id"], "frame_id": metadata["frame_id"]}
        for label, image in (
            ("condition", sample["condition_rgb_np"]),
            ("raw_diffusion", result.generated),
            ("restored", result.restored),
        ):
            metric_mask = sample.get("geometry_mask_np", sample["edit_mask_np"])
            row.update({f"{label}_{key}": value for key, value in all_metrics(image, sample["target_rgb_np"], metric_mask).items()})
        rows.append(row)
        save_comparison(
            output / f"{prefix}_comparison.png",
            sample["condition_rgb_np"],
            result.generated,
            sample["target_rgb_np"],
        )
        print(f"[{ordinal + 1}/{len(indices)}] {metadata['sequence_id']} frame={metadata['frame_id']}")

    with (output / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [key for key in rows[0] if key not in {"sample_index", "sequence_id", "frame_id"}]
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "manifest": str(manifest.resolve()),
        "split": args.split,
        "sample_count": len(rows),
        "sampling_seed": args.seed,
        "sample_indices": [int(index) for index in indices],
        "selection": "explicit_indices" if args.indices else "seeded_random_without_replacement",
        "inference_steps": args.steps,
        "metrics": {key: {"mean": float(np.mean([row[key] for row in rows])), "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0} for key in numeric},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
