"""Audit appearance on every training crop using unrefined input MANO poses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gaussian_hand.data.dataset import GaussianHandDataset
from gaussian_hand.model import GaussianHand
from gaussian_hand.render import render_gaussian_hand

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint["config"]
    dataset = GaussianHandDataset(ROOT / config["dataset_root"], "all", 512)
    model = GaussianHand.from_config(config, checkpoint["model"]["subject_betas"][0], ROOT).to(args.device)
    model.load_resizable_state_dict(checkpoint["model"])
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)

    def render_sample(index):
        x = {k: v.to(args.device) if torch.is_tensor(v) else v
             for k, v in dataset[index].items()}
        out = render_gaussian_hand(model, x["mano_pca"], x["wrist_q_wxyz"],
                                   x["wrist_t_world"], x["world_to_camera"],
                                   x["intrinsics"], 512, 512)
        return x, out

    records = []
    for i in range(len(dataset)):
        x, out = render_sample(i)
        support = out.get("mesh_mask", x["mask"])
        reference = support * x["mask"]
        squared = (out["rgb"] - x["image"] * support).square()
        foreground_squared = float((squared * reference).sum())
        foreground_values = float(reference.sum()) * 3
        pred, target = out["alpha"] > .5, support > .5
        records.append({"index": i, "sequence": x["sequence"], "frame_index": x["frame_index"],
                        "crop_mse": float(squared.mean()),
                        "foreground_squared_error": foreground_squared,
                        "foreground_values": foreground_values,
                        "foreground_mse": foreground_squared / max(foreground_values, 1),
                        "alpha_iou": float((pred & target).sum() / (pred | target).sum().clamp_min(1))})
        if (i+1) % 250 == 0:
            print(json.dumps({"evaluated": i+1, "total": len(dataset)}), flush=True)
    mse = np.mean([r["crop_mse"] for r in records])
    foreground_mse = sum(r["foreground_squared_error"] for r in records) / sum(r["foreground_values"] for r in records)
    report = {"checkpoint": str(args.checkpoint), "frames": len(records),
              "policy": "all training frames, raw MANO, no held-out evaluation",
              "black_background_crop_psnr": float(-10 * np.log10(max(mse, 1e-12))),
              ("visible_mano_reference_psnr" if model.strict_surface else "sam_foreground_psnr"): float(-10 * np.log10(max(foreground_mse, 1e-12))),
              ("mano_alpha_iou" if model.strict_surface else "mean_alpha_iou"): float(np.mean([r["alpha_iou"] for r in records]))}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    order = sorted(records, key=lambda r: r["foreground_mse"])
    selections = [("low_error", order[:4]),
                  ("median_error", order[len(order)//2-2:len(order)//2+2]),
                  ("high_error", order[-4:])]
    panels = []
    for label, chosen in selections:
        for record in chosen:
            x, out = render_sample(record["index"])
            target = (x["image"].cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
            rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
            alpha = np.repeat((out["alpha"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), 3, -1)
            row = np.concatenate([cv2.resize(y, (256, 256)) for y in (target, rgb, alpha)], axis=1)
            cv2.putText(row, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
            cv2.putText(row, f"{record['sequence']}:{record['frame_index']}",
                        (6, 249), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            panels.append(row)
    gallery = np.concatenate([np.concatenate(panels[i:i+2], axis=1)
                              for i in range(0, len(panels), 2)], axis=0)
    cv2.imwrite(str(args.output / "quality_range.jpg"), gallery)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
