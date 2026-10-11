#!/usr/bin/env python3
"""Render source RGB beside a Gaussian glove on a pure-white background."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gaussian_glove.render_overlay_video import (
    ROOT,
    camera_matrix,
    finish_encoder,
    load_renderer,
    panel,
    paste_hand,
    render_hand,
    start_encoder,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--left-checkpoint", type=Path,
        default=ROOT / "gaussian_glove/outputs/black_glove_left_sam31_offset/step_120000.pt")
    parser.add_argument(
        "--right-checkpoint", type=Path,
        default=ROOT / "gaussian_glove/outputs/black_glove_right_sam31_offset/step_120000.pt")
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = arguments()
    metadata = json.loads((args.session / "metadata.json").read_text())
    frame_count = int(metadata["frame_count"])
    camera = metadata["camera"]
    width, height = int(camera["width"]), int(camera["height"])
    fps = float(camera["fps_nominal"])
    intrinsics = camera_matrix(metadata)
    hands = list(metadata["hands_present"])
    checkpoint_by_side = {
        "left": args.left_checkpoint,
        "right": args.right_checkpoint,
    }
    renderers = {
        side: load_renderer(checkpoint_by_side[side], side, args.device)
        for side in hands
    }
    arrays = np.load(args.session / "mano.npz", allow_pickle=False)
    capture = cv2.VideoCapture(str(args.session / "rgb.mp4"))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.session / 'rgb.mp4'}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel_width = round(width * args.panel_height / height)
    encoder = start_encoder(
        args.output, 2 * panel_width, args.panel_height, fps)
    processed = 0
    try:
        for frame_index in range(frame_count):
            ok, source_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Video decode failed at frame {frame_index}")
            source_rgb = source_bgr[..., ::-1].astype(np.float32) / 255.0
            # render_gaussian_hand returns premultiplied RGB, so compositing it
            # over ones gives an exact pure-white background outside alpha.
            gaussian_rgb = np.ones_like(source_rgb)
            unused_composite = source_rgb.copy()
            unused_mask = np.zeros((height, width, 1), dtype=np.float32)
            for side in hands:
                rendered = render_hand(
                    renderers[side], arrays[f"{side}_hand_pose"][frame_index],
                    arrays[f"{side}_global_orient"][frame_index],
                    arrays[f"{side}_transl"][frame_index], intrinsics,
                    width, height, args.render_size, args.crop_scale, args.device)
                if rendered is not None:
                    paste_hand(
                        unused_composite, gaussian_rgb, unused_mask, rendered)
            gaussian_bgr = np.rint(
                np.clip(gaussian_rgb[..., ::-1], 0, 1) * 255).astype(np.uint8)
            row = np.concatenate([
                panel(source_bgr, args.panel_height, "SOURCE RGB"),
                panel(gaussian_bgr, args.panel_height,
                      "GAUSSIAN RENDER - WHITE BACKGROUND"),
            ], axis=1)
            if encoder.stdin is None:
                raise RuntimeError("ffmpeg encoder pipe closed")
            encoder.stdin.write(row.tobytes())
            processed += 1
            if processed % 60 == 0:
                print(json.dumps({
                    "session": args.session.name,
                    "rendered_frames": processed,
                    "total_frames": frame_count,
                }), flush=True)
    finally:
        capture.release()
        arrays.close()
        finish_encoder(encoder, "source-vs-gaussian-white")
    print(json.dumps({
        "status": "complete",
        "session": args.session.name,
        "frames": processed,
        "duration_seconds": processed / fps,
        "background": "pure_white",
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
