#!/usr/bin/env python3
"""Render full-sequence previews of the exact HUGG Aria SAM training input."""
from __future__ import annotations

import argparse
import json
import sqlite3
import zlib
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SEQUENCES = (
    "P0001_4bf4e21a",
    "P0010_41c4c626",
    "P0015_b0c5102b",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--pinhole-root", type=Path,
                        default=ROOT / "data/HUGG_ARIA_PINHOLE")
    parser.add_argument("--gaussian-root", type=Path,
                        default=ROOT / "data/HUGG_ARIA_GAUSSIANS_ALIGNED")
    parser.add_argument("--mask-root", type=Path,
                        default=ROOT / "outputs/sam2_hugg_aria_masks_v3_pilot")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/hugg_aria_aligned_training_input_previews")
    parser.add_argument("--model-size", type=int, default=512)
    parser.add_argument("--display-size", type=int, default=384)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--gaussian-threshold", type=int, default=16)
    parser.add_argument("--rotate-panels-clockwise", action="store_true",
                        help="Rotate only the four panel images 90 degrees clockwise")
    parser.add_argument("--output-name",
                        help="Custom MP4 filename (requires exactly one sequence)")
    return parser.parse_args()


def read_at(capture: cv2.VideoCapture, index: int, next_index: int) -> tuple[np.ndarray, int]:
    if index != next_index:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"Could not decode video frame {index}")
    return image, index + 1


def label_panel(image: np.ndarray, title: str, size: int) -> np.ndarray:
    panel = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (size, 38), (0, 0, 0), -1)
    cv2.putText(panel, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                .62, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def mask_overlay(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = rgb.copy()
    for value, color in ((1, (70, 210, 70)), (2, (40, 140, 255))):
        selected = labels == value
        if not selected.any():
            continue
        color_array = np.asarray(color, np.float32)
        result[selected] = (
            .45 * result[selected].astype(np.float32) + .55 * color_array
        ).astype(np.uint8)
        contours, _ = cv2.findContours(selected.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
    return result


def exact_model_input(rgb: np.ndarray, gaussian: np.ndarray,
                      labels: np.ndarray, size: int,
                      gaussian_threshold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dimensions = (size, size)
    target = cv2.resize(rgb, dimensions, interpolation=cv2.INTER_AREA)
    render = cv2.resize(gaussian, dimensions, interpolation=cv2.INTER_AREA)
    resized_labels = cv2.resize(labels, dimensions, interpolation=cv2.INTER_NEAREST)
    gaussian_foreground = render.max(axis=2) > gaussian_threshold
    overlay_region = gaussian_foreground & (resized_labels > 0)
    condition = target.copy()
    condition[overlay_region] = render[overlay_region]
    return target, render, condition, overlay_region


def header(width: int, sequence: str, source_index: int,
           gaussian_index: int | None, eligible: bool, reason: str,
           left_area: int, right_area: int) -> np.ndarray:
    result = np.zeros((76, width, 3), np.uint8)
    state = "KEEP / SENT TO MODEL" if eligible else "FILTERED / NOT SENT TO MODEL"
    state_color = (70, 220, 70) if eligible else (70, 70, 240)
    mapping = "none" if gaussian_index is None else str(gaussian_index)
    cv2.putText(result, f"{sequence}   source={source_index}   gaussian={mapping}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .62,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, state, (12, 59), cv2.FONT_HERSHEY_SIMPLEX,
                .62, state_color, 2, cv2.LINE_AA)
    details = f"reason={reason}   SAM left={left_area}px right={right_area}px"
    cv2.putText(result, details, (310, 59), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (220, 220, 220), 1, cv2.LINE_AA)
    return result


def render_sequence(args: argparse.Namespace, sequence: str) -> dict:
    rgb_path = args.pinhole_root / sequence / "rgb_214_1_pinhole.mp4"
    gaussian_path = args.gaussian_root / sequence / "reconstruction.mp4"
    database_path = args.mask_root / sequence / "masks.sqlite"
    for path in (rgb_path, gaussian_path, database_path):
        if not path.exists():
            raise FileNotFoundError(path)

    database = sqlite3.connect(f"file:{database_path.resolve()}?mode=ro", uri=True)
    metadata = {
        key: json.loads(value)
        for key, value in database.execute("SELECT key,value_json FROM metadata")
    }
    rows = database.execute(
        "SELECT frame_index,labels_zlib,area_left,area_right,training_eligible,"
        "training_filter_reason FROM frames ORDER BY frame_index"
    ).fetchall()
    height, width = int(metadata["height"]), int(metadata["width"])
    rgb_capture = cv2.VideoCapture(str(rgb_path))
    gaussian_capture = cv2.VideoCapture(str(gaussian_path))
    if not rgb_capture.isOpened() or not gaussian_capture.isOpened():
        raise RuntimeError(f"Could not open source videos for {sequence}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{sequence}_exact_sam_gaussian_condition.mp4"
    output = args.output_root / output_name
    panel_size = int(args.display_size)
    frame_width, frame_height = panel_size * 2, panel_size * 2 + 76
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(args.fps), (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output}")

    next_rgb = next_gaussian = -1
    eligible_count = 0
    try:
        for frame_index, blob, left, right, eligible_value, reason in rows:
            frame_index = int(frame_index)
            eligible = bool(eligible_value)
            gaussian_index = frame_index
            rgb, next_rgb = read_at(rgb_capture, frame_index, next_rgb)
            gaussian, next_gaussian = read_at(
                gaussian_capture, gaussian_index, next_gaussian
            )
            labels = np.frombuffer(zlib.decompress(blob), np.uint8).reshape(height, width)
            target, render, condition, overlay_region = exact_model_input(
                rgb, gaussian, labels, int(args.model_size),
                int(args.gaussian_threshold)
            )
            overlay = mask_overlay(target, cv2.resize(
                labels, (args.model_size, args.model_size), interpolation=cv2.INTER_NEAREST
            ))
            if eligible:
                eligible_count += 1
                input_title = "Gaussian foreground AND SAM (KEEP)"
            else:
                condition = (condition.astype(np.float32) * .42).astype(np.uint8)
                input_title = "Hypothetical condition (FILTERED)"
            if args.rotate_panels_clockwise:
                target, render, overlay, condition = (
                    cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    for image in (target, render, overlay, condition)
                )
            if not eligible:
                cv2.putText(condition, "NOT USED FOR TRAINING", (42, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, .92, (40, 40, 255), 3,
                            cv2.LINE_AA)
            panels = np.concatenate((
                np.concatenate((label_panel(target, "Target RGB", panel_size),
                                label_panel(render, "Aligned Gaussian render", panel_size)), axis=1),
                np.concatenate((label_panel(overlay, "SAM labels", panel_size),
                                label_panel(condition, input_title, panel_size)), axis=1),
            ), axis=0)
            frame = np.concatenate((header(
                frame_width, sequence, frame_index, gaussian_index, eligible,
                str(reason), int(left), int(right)
            ), panels), axis=0)
            writer.write(frame)
    finally:
        writer.release()
        rgb_capture.release()
        gaussian_capture.release()
        database.close()
    return {
        "sequence": sequence,
        "frames": len(rows),
        "training_eligible": eligible_count,
        "filtered": len(rows) - eligible_count,
        "output": str(output.resolve()),
        "bytes": output.stat().st_size,
    }


def main() -> None:
    args = arguments()
    sequences = args.sequence or list(DEFAULT_SEQUENCES)
    if args.output_name:
        if len(sequences) != 1:
            raise ValueError("--output-name requires exactly one --sequence")
        if Path(args.output_name).name != args.output_name:
            raise ValueError("--output-name must be a filename, not a path")
        if not args.output_name.lower().endswith(".mp4"):
            raise ValueError("--output-name must end in .mp4")
    summaries = []
    for sequence in sequences:
        summary = render_sequence(args, sequence)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    manifest = {
        "description": "Frame-aligned Gaussian foreground intersected with SAM at opacity=1.0; filtered frames are never sent to training",
        "overlay_logic": "(gaussian.max(channel) > gaussian_threshold) AND (sam_label > 0)",
        "gaussian_threshold": args.gaussian_threshold,
        "model_size": args.model_size,
        "display_size": args.display_size,
        "rotate_panels_clockwise": args.rotate_panels_clockwise,
        "sequences": summaries,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
