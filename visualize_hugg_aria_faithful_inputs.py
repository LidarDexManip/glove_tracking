#!/usr/bin/env python3
"""Visualize the paper-faithful HUGG Aria crop and conditioning pipeline."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from hand_restoration.config import load_json_config
from hand_restoration.hugg_aria_dataset import HuggAriaOverlayDataset
from hand_restoration.visualize import rgb_float_to_u8


ROOT = Path(__file__).resolve().parent


def absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def build_dataset(config: dict) -> HuggAriaOverlayDataset:
    data = config["data"]
    return HuggAriaOverlayDataset(
        manifest=absolute(data["train_manifest"]),
        pinhole_root=absolute(data["pinhole_root"]),
        render_root=absolute(data["render_root"]),
        mask_root=absolute(data["mask_root"]),
        output_size=data.get("output_size", 512),
        condition_variant=data.get("condition_variant", "sam_mask"),
        render_opacity=data.get("render_opacity", 1.0),
        render_kind=data.get("render_kind", "gaussian"),
        render_alpha_filename=data.get("render_alpha_filename", "alpha.mkv"),
        alpha_threshold=data.get("alpha_threshold", 0.0),
        loss_mask_source=data.get("loss_mask_source", "overlay"),
        loss_mask_root=(
            absolute(data["loss_mask_root"])
            if data.get("loss_mask_root")
            else None
        ),
        loss_mask_alpha_filename=data.get(
            "loss_mask_alpha_filename", "alpha.mkv"
        ),
        spatial_mode=data.get("spatial_mode", "full_frame"),
        crop_scale=data.get("crop_scale", 1.2),
        condition_style=data.get("condition_style", "legacy_overlay"),
        hand_mask_dilation_px=data.get("hand_mask_dilation_px", 8),
        wrist_mask_enabled=data.get("wrist_mask_enabled", True),
        wrist_geometry_source=data.get(
            "wrist_geometry_source", "silhouette_pca"
        ),
        wrist_ring_root=(
            absolute(data["wrist_ring_root"])
            if data.get("wrist_ring_root")
            else None
        ),
        wrist_length_ratio=data.get("wrist_length_ratio", 0.10),
        wrist_width_scale=data.get("wrist_width_scale", 1.10),
        wrist_sleeve_forearm_ratio=data.get(
            "wrist_sleeve_forearm_ratio", 0.60
        ),
        wrist_sleeve_hand_overlap_ratio=data.get(
            "wrist_sleeve_hand_overlap_ratio", 0.25
        ),
        wrist_ring_transverse_scale=data.get(
            "wrist_ring_transverse_scale", 1.30
        ),
        wrist_sleeve_orientation=data.get(
            "wrist_sleeve_orientation", "ring_min_area"
        ),
        condition_fill_value=data.get("condition_fill_value", 0.0),
        include_numpy=True,
        max_open_sequences=2,
    )


def diverse_indices(records: list[dict], count: int) -> list[int]:
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_sequence[str(record["sequence_id"])].append(index)
    sequences = sorted(by_sequence)
    chosen_sequences = np.linspace(
        0, len(sequences) - 1, min(count, len(sequences)), dtype=int
    )
    indices = []
    for sequence_index in chosen_sequences:
        candidates = by_sequence[sequences[int(sequence_index)]]
        indices.append(candidates[len(candidates) // 2])
    return indices


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (15, 15, 15), -1)
    cv2.putText(
        image,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def mask_panel(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask > 0] = color
    return image


def mask_on_rgb(
    rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    image = rgb_float_to_u8(rgb).astype(np.float32)
    selected = mask > 0
    image[selected] = image[selected] * 0.45 + np.asarray(color) * 0.55
    return image.astype(np.uint8)


def sample_grid(sample: dict) -> np.ndarray:
    target = rgb_float_to_u8(sample["target_rgb_np"])
    full = rgb_float_to_u8(sample["full_target_rgb_np"])
    box = sample["crop_box"]
    cv2.rectangle(full, (box.x0, box.y0), (box.x1 - 1, box.y1 - 1), (0, 255, 255), 7)
    full = cv2.resize(full, (512, 512), interpolation=cv2.INTER_AREA)
    full_condition = cv2.resize(
        rgb_float_to_u8(sample["full_condition_rgb_np"]),
        (512, 512),
        interpolation=cv2.INTER_AREA,
    )
    wrist = mask_on_rgb(
        sample["target_rgb_np"], sample["wrist_mask_np"], (255, 32, 32)
    )
    for polygon in sample["wrist_polygons_np"]:
        cv2.polylines(wrist, [polygon], True, (255, 255, 0), 3, cv2.LINE_AA)
    ring_points = sample.get("wrist_ring_points_np")
    palm_points = sample.get("wrist_palm_points_np")
    if ring_points is not None and palm_points is not None:
        for hand_points, palm_point in zip(ring_points, palm_points):
            finite_points = hand_points[np.isfinite(hand_points).all(axis=1)]
            for x, y in finite_points:
                cv2.circle(
                    wrist,
                    (int(round(x)), int(round(y))),
                    3,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )
            if finite_points.size and np.isfinite(palm_point).all():
                ring_center = finite_points.mean(axis=0)
                start = tuple(np.rint(ring_center).astype(int))
                end = tuple(np.rint(palm_point).astype(int))
                cv2.arrowedLine(
                    wrist, start, end, (64, 255, 64), 3, cv2.LINE_AA
                )
                cv2.circle(wrist, end, 5, (255, 0, 255), -1, cv2.LINE_AA)

    panels = [
        label_panel(
            full,
            f"Full GT + {sample['metadata']['crop_scale']:.1f}x union bbox",
        ),
        label_panel(target, "Cropped GT (model target)"),
        label_panel(
            rgb_float_to_u8(sample["raw_overlay_rgb_np"]),
            "Raw MANO alpha overlay",
        ),
        label_panel(
            rgb_float_to_u8(sample["overlay_rgb_np"]),
            "MANO alpha intersect SAM",
        ),
        label_panel(rgb_float_to_u8(sample["condition_rgb_np"]), "Final condition RGB"),
        label_panel(
            full_condition,
            "Full-frame condition before crop",
        ),
        label_panel(mask_panel(sample["mano_mask_np"], (64, 255, 64)), "MANO geometry/loss mask"),
        label_panel(
            mask_panel(sample["visible_hand_mask_np"], (64, 255, 64)),
            "MANO intersect SAM -> ControlNet ch5",
        ),
        label_panel(
            mask_panel(
                sample["sam_excluded_mano_mask_np"], (255, 80, 220)
            ),
            "MANO minus SAM (excluded overlay)",
        ),
        label_panel(mask_panel(sample["dilated_hand_mask_np"], (64, 160, 255)), "Visible hand dilated by 3px"),
        label_panel(
            wrist,
            f"Sleeve: {sample['metadata']['wrist_sleeve_orientation']}",
        ),
        label_panel(mask_panel(sample["edit_mask_np"], (255, 255, 255)), "Final edit/paste-back mask"),
    ]
    return np.vstack((np.hstack(panels[:6]), np.hstack(panels[6:])))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hand_restoration/faithful_input_previews"),
    )
    parser.add_argument("--count", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = load_json_config(args.config.resolve())
    dataset = build_dataset(config)
    records = dataset.samples
    indices = diverse_indices(records, args.count)
    output = absolute(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    summary = []
    for index in indices:
        sample = dataset[index]
        metadata = sample["metadata"]
        grid = sample_grid(sample)
        name = (
            f"{metadata['sequence_id']}_frame_{int(metadata['frame_id']):06d}.jpg"
        )
        cv2.imwrite(str(output / name), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        thumbnail_height = int(round(grid.shape[0] * 1024 / grid.shape[1]))
        thumbnails.append(
            cv2.resize(
                grid, (1024, thumbnail_height), interpolation=cv2.INTER_AREA
            )
        )
        summary.append(
            {
                "dataset_index": index,
                "sequence_id": metadata["sequence_id"],
                "frame_index": metadata["frame_id"],
                "crop_box_xyxy": metadata["crop_box_xyxy"],
                "preview": name,
            }
        )
    if len(thumbnails) % 2:
        thumbnails.append(np.zeros_like(thumbnails[0]))
    contact = np.vstack(
        [np.hstack(thumbnails[offset : offset + 2])
         for offset in range(0, len(thumbnails), 2)]
    )
    cv2.imwrite(
        str(output / "contact_sheet.jpg"),
        cv2.cvtColor(contact, cv2.COLOR_RGB2BGR),
    )
    (output / "samples.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "samples": summary}, indent=2))


if __name__ == "__main__":
    main()
