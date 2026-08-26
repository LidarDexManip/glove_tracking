#!/usr/bin/env python3
"""Verify one aligned HUGG Aria sample and its SAM-weighted edit mask."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from hand_restoration.config import load_json_config
from hand_restoration.hugg_aria_dataset import HuggAriaOverlayDataset


ROOT = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def dataset(
    config: dict, manifest: Path, condition_variant: str
) -> HuggAriaOverlayDataset:
    data = config["data"]
    return HuggAriaOverlayDataset(
        manifest=manifest,
        pinhole_root=ROOT / data["pinhole_root"],
        render_root=ROOT / data["render_root"],
        mask_root=ROOT / data["mask_root"],
        output_size=data.get("output_size", 512),
        condition_variant=condition_variant,
        render_opacity=data.get("render_opacity", 1.0),
        render_kind=data.get("render_kind", "gaussian"),
        render_alpha_filename=data.get("render_alpha_filename", "alpha.mkv"),
        alpha_threshold=data.get("alpha_threshold", 0.0),
        loss_mask_source=data.get("loss_mask_source", "overlay"),
        loss_mask_root=(
            ROOT / data["loss_mask_root"]
            if data.get("loss_mask_root")
            else None
        ),
        loss_mask_alpha_filename=data.get(
            "loss_mask_alpha_filename", "alpha.mkv"
        ),
        include_numpy=False,
        max_open_sequences=1,
    )


def main() -> None:
    args = arguments()
    config = load_json_config(args.config.resolve())
    manifest = ROOT / config["data"]["train_manifest"]
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = records[args.sample_index]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        sample = dataset(config, Path(handle.name), "sam_mask")[0]
        direct = dataset(config, Path(handle.name), "direct_overlay")[0]

    if not torch.equal(sample["target_rgb"], direct["target_rgb"]):
        raise AssertionError("SAM/direct target tensors differ")
    if not torch.any(sample["edit_mask"]):
        raise AssertionError("Aligned SAM edit mask is empty")
    metadata = sample["metadata"]
    result = {
        "status": "ok",
        "sequence": metadata["sequence_id"],
        "frame": metadata["frame_id"],
        "target_equal": True,
        "sam_edit_pixels": int(sample["edit_mask"].sum()),
        "direct_edit_pixels": int(direct["edit_mask"].sum()),
        "target_shape": list(sample["target_rgb"].shape),
        "hand_loss_weight": config["training"]["hand_loss_weight"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
