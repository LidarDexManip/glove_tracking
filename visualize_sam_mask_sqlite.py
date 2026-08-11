#!/usr/bin/env python3
"""Overlay stored SAM hand labels from masks.sqlite on the derived pinhole video."""
from __future__ import annotations

import argparse
import json
import sqlite3
import zlib
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data/HUGG_ARIA_PINHOLE"
DEFAULT_MASKS = ROOT / "outputs/sam2_hugg_aria_masks"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", help="Sequence name, e.g. P0002_59a84a3a")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame", type=int, help="Write one PNG instead of a video")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="Zero means until video end")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--scale", type=float, default=0.75)
    return parser.parse_args()


def overlay_frame(
    image: np.ndarray,
    labels: np.ndarray,
    frame_index: int,
    chunk_status: str,
    alpha: float,
) -> np.ndarray:
    result = image.copy()
    colors = {1: np.array((70, 210, 70), dtype=np.float32), 2: np.array((40, 140, 255), dtype=np.float32)}
    for label, color in colors.items():
        mask = labels == label
        if mask.any():
            result[mask] = ((1.0 - alpha) * result[mask] + alpha * color).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color.astype(int).tolist(), 2)

    left_area = int((labels == 1).sum())
    right_area = int((labels == 2).sum())
    lines = [f"frame={frame_index}", f"left={left_area} right={right_area}"]
    if chunk_status == "no_prompt":
        lines.append("NO MANO PROMPT - MASK UNAVAILABLE")
    else:
        lines.append("SAM2.1 Large video predictor")
    for row, text in enumerate(lines):
        y = 34 + row * 30
        color = (50, 50, 255) if chunk_status == "no_prompt" and row == 2 else (255, 255, 255)
        cv2.putText(result, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(result, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 1, cv2.LINE_AA)
    return result


def main() -> None:
    args = arguments()
    sequence_dir = args.dataset_root.resolve() / args.sequence
    database = args.mask_root.resolve() / args.sequence / "masks.sqlite"
    video = sequence_dir / "rgb_214_1_pinhole.mp4"
    for path in (database, video):
        if not path.exists():
            raise FileNotFoundError(path)

    connection = sqlite3.connect(database)
    metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key,value_json FROM metadata")}
    height, width = int(metadata["height"]), int(metadata["width"])
    chunk_status = {
        int(index): status
        for index, status in connection.execute("SELECT chunk_index,status FROM chunks")
    }
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0

    start = args.frame if args.frame is not None else args.start
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    output = args.output
    if output is None:
        suffix = f"frame_{start:06d}.png" if args.frame is not None else "sam_overlay.mp4"
        output = args.mask_root.resolve() / args.sequence / suffix
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    limit = 1 if args.frame is not None else args.max_frames
    processed = 0
    try:
        while limit == 0 or processed < limit:
            frame_index = start + processed
            ok, image = capture.read()
            if not ok:
                break
            row = connection.execute(
                "SELECT chunk_index,labels_zlib FROM frames WHERE frame_index=?", (frame_index,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Frame {frame_index} is absent from {database}")
            chunk_index, blob = row
            labels = np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(height, width)
            rendered = overlay_frame(
                image, labels, frame_index, chunk_status[int(chunk_index)], float(args.alpha)
            )
            if args.scale != 1.0:
                rendered = cv2.resize(rendered, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            if args.frame is not None:
                if not cv2.imwrite(str(output), rendered):
                    raise RuntimeError(f"Could not write {output}")
            else:
                if writer is None:
                    out_height, out_width = rendered.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create {output}")
                writer.write(rendered)
            processed += 1
    finally:
        capture.release()
        connection.close()
        if writer is not None:
            writer.release()

    print(json.dumps({"sequence": args.sequence, "frames": processed, "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
