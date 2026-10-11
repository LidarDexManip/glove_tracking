#!/usr/bin/env python3
"""Render the learned black-glove appearance over a HOT3D Aria sequence."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from smplx.lbs import blend_shapes, vertices2joints

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hot3d/hot3d"))

from data_loaders.HeadsetPose3dProvider import load_headset_pose_provider_from_csv

from gaussian_glove.render_overlay_video import (
    finish_encoder,
    load_renderer,
    panel,
    paste_hand,
    render_hand,
    start_encoder,
)
from gaussian_hand.model.mano import LearnableMANO
from gaussian_hand.render_sequence import TOLERANCE_NS, world_to_camera


HAND_KEY = {"left": "0", "right": "1"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--pinhole-root", type=Path, default=ROOT / "data/HUGG_ARIA_PINHOLE")
    parser.add_argument(
        "--left-checkpoint", type=Path,
        default=ROOT / "gaussian_glove/outputs/black_glove_left_sam31_offset/step_120000.pt")
    parser.add_argument(
        "--right-checkpoint", type=Path,
        default=ROOT / "gaussian_glove/outputs/black_glove_right_sam31_offset/step_120000.pt")
    parser.add_argument("--sides", nargs="+", choices=("left", "right"),
                        default=("left", "right"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--panel-height", type=int, default=512)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def nearest_pose(
    timestamps: list[int], poses: list[dict], timestamp: int,
) -> dict | None:
    index = bisect.bisect_left(timestamps, timestamp)
    choices = [i for i in (index - 1, index) if 0 <= i < len(timestamps)]
    if not choices:
        return None
    best = min(choices, key=lambda i: abs(timestamps[i] - timestamp))
    return poses[best] if abs(timestamps[best] - timestamp) <= TOLERANCE_NS else None


def load_pca_converter(side: str) -> LearnableMANO:
    path = ROOT / "mano_v1_2/models" / f"MANO_{side.upper()}.pkl"
    model = LearnableMANO(
        str(path), handedness=side, pose_representation="pca")
    model.eval()
    return model


def pca_to_axis_angle(
    pose: list[float],
    converter: LearnableMANO,
    axis_model: LearnableMANO,
) -> np.ndarray:
    pca = torch.tensor(pose, dtype=torch.float32)
    axis_angle = (
        pca @ converter.layer.hand_components
        + converter.layer.pose_mean[3:]
        - axis_model.layer.pose_mean[3:].cpu()
    )
    return axis_angle.numpy()


def mano_root(model: LearnableMANO, betas: torch.Tensor) -> np.ndarray:
    """Return the shaped MANO rotation pivot used by global_orient."""
    layer = model.layer
    shaped = layer.v_template[None] + blend_shapes(
        betas[None], layer.shapedirs)
    joints = vertices2joints(layer.J_regressor, shaped)
    return joints[0, 0].detach().cpu().numpy().astype(np.float64)


def wrist_in_camera(
    hand: dict, world_to_cam: np.ndarray, root: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MANO global pose with its root-pivot translation convention."""
    wrist = hand["wrist_xform"]
    quaternion = np.asarray(wrist["q_wxyz"], dtype=np.float64)
    rotation_world = Rotation.from_quat(
        quaternion[[1, 2, 3, 0]]).as_matrix()
    rotation_camera = world_to_cam[:3, :3] @ rotation_world
    translation_world = np.asarray(wrist["t_xyz"], dtype=np.float64)
    # MANO rotates around its shaped root J, so transl is not the origin of a
    # conventional 4x4 wrist transform. Preserve R(v-J)+J+t under the
    # world-to-camera rigid transform.
    translation = (
        world_to_cam[:3, :3] @ (root + translation_world)
        + world_to_cam[:3, 3] - root)
    orient = Rotation.from_matrix(rotation_camera).as_rotvec()
    return orient.astype(np.float32), translation.astype(np.float32)


def intrinsics(calibration: dict) -> np.ndarray:
    fx, fy = calibration["focal_lengths"]
    cx, cy = calibration["principal_point"]
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64)


@torch.no_grad()
def main() -> None:
    args = arguments()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    source = args.pinhole_root / args.sequence
    if not source.is_dir():
        raise FileNotFoundError(source)
    args.output.mkdir(parents=True, exist_ok=True)

    with (source / "frame_timestamps_214_1.csv").open(newline="") as handle:
        frame_timestamps = [
            int(row["timestamp_ns"]) for row in csv.DictReader(handle)]
    calibrations = [
        json.loads(line)
        for line in (source / "frame_pinhole_calibration_214_1.jsonl")
        .read_text().splitlines() if line.strip()
    ]
    pose_items = []
    with (source / "mano_hand_pose_trajectory.jsonl").open() as handle:
        for line in handle:
            item = json.loads(line)
            pose_items.append((int(item["timestamp_ns"]), item["hand_poses"]))
    pose_items.sort(key=lambda value: value[0])
    pose_timestamps = [value[0] for value in pose_items]
    poses = [value[1] for value in pose_items]

    if not 0 <= args.start_frame < len(frame_timestamps):
        raise ValueError(
            f"--start-frame must be in [0, {len(frame_timestamps) - 1}]")
    stop = min(len(frame_timestamps), args.start_frame + args.frames)
    checkpoint_by_side = {
        "left": args.left_checkpoint,
        "right": args.right_checkpoint,
    }
    renderers = {
        side: load_renderer(checkpoint_by_side[side], side, args.device)
        for side in args.sides
    }
    converters = {side: load_pca_converter(side) for side in args.sides}
    roots = {
        side: mano_root(renderers[side].model.mano, renderers[side].betas)
        for side in args.sides
    }
    headset = load_headset_pose_provider_from_csv(
        str(source / "headset_trajectory.csv"))

    capture = cv2.VideoCapture(str(source / "rgb_214_1_pinhole.mp4"))
    if not capture.isOpened():
        raise RuntimeError("Cannot open Aria pinhole video")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    first_calibration = calibrations[args.start_frame]
    width = int(first_calibration["image_width"])
    height = int(first_calibration["image_height"])
    panel_width = round(width * args.panel_height / height)
    video_path = args.output / f"{args.sequence}_black_glove_aria_overlay.mp4"
    encoder = start_encoder(
        video_path, panel_width * 3, args.panel_height, args.fps)
    frame_records = []
    rendered_counts = {side: 0 for side in args.sides}

    try:
        for frame_index in range(args.start_frame, stop):
            ok, source_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Aria video decode failed at frame {frame_index}")
            calibration = calibrations[frame_index]
            if int(calibration["frame_index"]) != frame_index:
                raise ValueError("Calibration/frame alignment mismatch")
            timestamp = frame_timestamps[frame_index]
            hand_poses = nearest_pose(pose_timestamps, poses, timestamp)
            world_to_cam = world_to_camera(calibration, headset, timestamp)
            source_rgb = source_bgr[..., ::-1].astype(np.float32) / 255.0
            composite_rgb = source_rgb.copy()
            gaussian_rgb = np.zeros_like(source_rgb)
            support = np.zeros((height, width, 1), dtype=np.float32)
            rendered_sides = []

            if hand_poses is not None and world_to_cam is not None:
                camera_k = intrinsics(calibration)
                for side in args.sides:
                    hand = hand_poses.get(HAND_KEY[side])
                    if hand is None:
                        continue
                    axis_angle = pca_to_axis_angle(
                        hand["pose"], converters[side], renderers[side].model.mano)
                    orient, translation = wrist_in_camera(
                        hand, world_to_cam, roots[side])
                    rendered = render_hand(
                        renderers[side], axis_angle, orient, translation,
                        camera_k, width, height, args.render_size,
                        args.crop_scale, args.device)
                    if rendered is not None:
                        paste_hand(
                            composite_rgb, gaussian_rgb, support, rendered)
                        rendered_sides.append(side)
                        rendered_counts[side] += 1

            composite_bgr = np.rint(
                np.clip(composite_rgb[..., ::-1], 0, 1) * 255).astype(np.uint8)
            gaussian_bgr = np.rint(
                np.clip(gaussian_rgb[..., ::-1], 0, 1) * 255).astype(np.uint8)
            row = np.concatenate([
                panel(source_bgr, args.panel_height, "ARIA SOURCE RGB"),
                panel(gaussian_bgr, args.panel_height, "BLACK-GLOVE GAUSSIAN"),
                panel(composite_bgr, args.panel_height, "GAUSSIAN OVERLAY"),
            ], axis=1)
            if encoder.stdin is None:
                raise RuntimeError("ffmpeg encoder pipe closed")
            encoder.stdin.write(row.tobytes())
            frame_records.append({
                "output_frame": len(frame_records),
                "source_frame": frame_index,
                "timestamp_ns": timestamp,
                "rendered_sides": rendered_sides,
            })
            if len(frame_records) % 30 == 0:
                print(json.dumps({
                    "rendered_frames": len(frame_records),
                    "hand_counts": rendered_counts,
                }), flush=True)
    finally:
        capture.release()
        finish_encoder(encoder, "aria_overlay")

    with (args.output / "frames.jsonl").open("w") as handle:
        for record in frame_records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    report = {
        "sequence": args.sequence,
        "source_type": "HOT3D Aria pinhole RGB stream 214-1",
        "source_video": str(source / "rgb_214_1_pinhole.mp4"),
        "start_frame": args.start_frame,
        "stop_frame_exclusive": args.start_frame + len(frame_records),
        "frames": len(frame_records),
        "fps": args.fps,
        "duration_seconds": len(frame_records) / args.fps,
        "rendered_hand_frames": rendered_counts,
        "pose_conversion": (
            "HOT3D 15D MANO PCA converted exactly to 45D axis-angle; "
            "world wrist pose converted to the Aria camera frame with shaped "
            "MANO root-pivot compensation"),
        "appearance_identity": "black-glove checkpoints; Aria subject betas not used",
        "checkpoints": {
            side: str(checkpoint_by_side[side]) for side in args.sides},
        "video": video_path.name,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
