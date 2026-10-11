#!/usr/bin/env python3
"""Render the learned glove appearance for one full 45D MANO pose."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gaussian_hand.data.geometry import crop_intrinsics, project_points, square_crop
from gaussian_hand.model import GaussianHand
from gaussian_hand.model.pose_refinement import axis_angle_to_quaternion
from gaussian_hand.render import render_gaussian_hand


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--session", type=Path,
        help="Downloaded session directory containing metadata.json and mano.npz.")
    source.add_argument(
        "--pose-json", type=Path,
        help="JSON with hand_pose[45], and optional global_orient/transl/camera.")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index when --session is used.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def camera_from_dict(camera: dict, default_size: int) -> tuple[np.ndarray, int, int]:
    width = int(camera.get("width", default_size))
    height = int(camera.get("height", default_size))
    focal = float(camera.get("focal", 0.9 * max(width, height)))
    fx = float(camera.get("fx", focal))
    fy = float(camera.get("fy", focal))
    cx = float(camera.get("cx", (width - 1) / 2))
    cy = float(camera.get("cy", (height - 1) / 2))
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    ), width, height


def load_input(args: argparse.Namespace, side: str) -> tuple[dict, np.ndarray, int, int]:
    if args.session is not None:
        metadata = json.loads((args.session / "metadata.json").read_text())
        if side not in metadata["hands_present"]:
            raise ValueError(f"{args.session.name} has no {side} hand")
        with np.load(args.session / "mano.npz", allow_pickle=False) as data:
            count = len(data[f"{side}_hand_pose"])
            if not 0 <= args.frame < count:
                raise IndexError(f"frame must be in [0, {count - 1}]")
            item = {
                "hand_pose": data[f"{side}_hand_pose"][args.frame].tolist(),
                "global_orient": data[f"{side}_global_orient"][args.frame].tolist(),
                "transl": data[f"{side}_transl"][args.frame].tolist(),
                "source_session": args.session.name,
                "source_frame": args.frame,
            }
        intrinsics, width, height = camera_from_dict(
            metadata["camera"], args.size)
        return item, intrinsics, width, height

    item = json.loads(args.pose_json.read_text())
    if len(item["hand_pose"]) != 45:
        raise ValueError("hand_pose must contain 45 axis-angle values")
    item.setdefault("global_orient", [0.0, 0.0, 0.0])
    item.setdefault("transl", [0.0, 0.0, 0.5])
    intrinsics, width, height = camera_from_dict(
        item.get("camera", {}), args.size)
    return item, intrinsics, width, height


@torch.no_grad()
def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint["config"]
    binding = checkpoint["binding"]
    if config.get("pose_representation") != "axis_angle":
        raise ValueError("This renderer expects a full 45D axis-angle checkpoint")
    side = binding["handedness"]
    state = checkpoint["model"]
    betas = state["subject_betas"][0].to(args.device)
    model = GaussianHand.from_config(config, betas, ROOT).to(args.device)
    model.load_resizable_state_dict(state)
    model.eval()

    item, intrinsics, width, height = load_input(args, side)
    pose = torch.tensor(
        item["hand_pose"], dtype=torch.float32, device=args.device)
    global_orient = torch.tensor(
        item["global_orient"], dtype=torch.float32, device=args.device)
    quaternion = axis_angle_to_quaternion(global_orient)
    translation = torch.tensor(
        item["transl"], dtype=torch.float32, device=args.device)
    vertices = model.mano(pose, quaternion, translation, betas)[0]
    uv, depth = project_points(
        vertices.cpu().numpy(), np.eye(4), intrinsics)
    valid = depth > 0.01
    if not valid.any():
        raise ValueError("Pose is behind the camera")
    box = square_crop(uv[valid], width, height, args.crop_scale)
    if box is None:
        raise ValueError("Could not form an in-frame crop for this pose")
    crop_k = crop_intrinsics(intrinsics, box, args.size)
    result = render_gaussian_hand(
        model, pose, quaternion, translation,
        torch.eye(4, device=args.device),
        torch.tensor(crop_k, dtype=torch.float32, device=args.device),
        args.size, args.size,
    )
    rgb = result["rgb"].clamp(0, 1).cpu().numpy()
    alpha = result["alpha"].clamp(0, 1).cpu().numpy()
    black_rgb = np.rint(rgb * 255).astype(np.uint8)
    alpha_u8 = np.rint(alpha[..., 0] * 255).astype(np.uint8)
    straight = np.divide(
        rgb, np.maximum(alpha, 1e-6), out=np.zeros_like(rgb), where=alpha > 1e-6)
    straight = np.rint(np.clip(straight, 0, 1) * 255).astype(np.uint8)
    bgra = np.concatenate((straight[..., ::-1], alpha_u8[..., None]), axis=-1)
    if not cv2.imwrite(str(args.output / "render_rgb.png"), black_rgb[..., ::-1]):
        raise RuntimeError("Failed to write render_rgb.png")
    if not cv2.imwrite(str(args.output / "alpha.png"), alpha_u8):
        raise RuntimeError("Failed to write alpha.png")
    if not cv2.imwrite(str(args.output / "render_rgba.png"), bgra):
        raise RuntimeError("Failed to write render_rgba.png")
    report = {
        "checkpoint": str(args.checkpoint),
        "model_id": config["model_id"],
        "handedness": side,
        "pose_representation": "axis_angle",
        "crop_xyxy": list(map(int, box)),
        "crop_intrinsics": crop_k.tolist(),
        "render_size": args.size,
        "input": item,
        "outputs": ["render_rgb.png", "alpha.png", "render_rgba.png"],
    }
    (args.output / "render.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
