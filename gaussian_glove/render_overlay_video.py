#!/usr/bin/env python3
"""Render Gaussian gloves over a compact RGB/MANO session video."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
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
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument(
        "--left-checkpoint", type=Path,
        default=ROOT / "gaussian_glove/outputs/black_glove_left_sam31_offset/step_120000.pt")
    parser.add_argument(
        "--right-checkpoint", type=Path,
        default=ROOT / "gaussian_glove/outputs/black_glove_right_sam31_offset/step_120000.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@dataclass
class HandRenderer:
    side: str
    model: GaussianHand
    betas: torch.Tensor


def camera_matrix(metadata: dict) -> np.ndarray:
    camera = metadata["camera"]
    return np.asarray([
        [camera["fx"], 0.0, camera["cx"]],
        [0.0, camera["fy"], camera["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def load_renderer(path: Path, expected_side: str, device: str) -> HandRenderer:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    side = checkpoint["binding"]["handedness"]
    if side != expected_side:
        raise ValueError(f"{path} is bound to {side}, expected {expected_side}")
    config = checkpoint["config"]
    if config.get("pose_representation") != "axis_angle":
        raise ValueError(f"{path} does not use the required 45D axis-angle pose")
    state = checkpoint["model"]
    betas = state["subject_betas"][0].to(device)
    model = GaussianHand.from_config(config, betas, ROOT).to(device)
    model.load_resizable_state_dict(state)
    model.eval()
    return HandRenderer(side, model, betas)


def start_encoder(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.8g}", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def finish_encoder(process: subprocess.Popen, name: str) -> None:
    if process.stdin is None:
        raise RuntimeError(f"{name} encoder has no stdin")
    process.stdin.close()
    code = process.wait()
    if code:
        raise RuntimeError(f"{name} ffmpeg failed with exit code {code}")


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        result, text, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, .75,
        (255, 255, 255), 2, cv2.LINE_AA)
    return result


def panel(image: np.ndarray, height: int, text: str) -> np.ndarray:
    width = round(image.shape[1] * height / image.shape[0])
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return label(resized, text)


@torch.no_grad()
def render_hand(
    renderer: HandRenderer,
    pose: np.ndarray,
    orient: np.ndarray,
    translation: np.ndarray,
    intrinsics: np.ndarray,
    frame_width: int,
    frame_height: int,
    render_size: int,
    crop_scale: float,
    device: str,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, np.ndarray] | None:
    pose_tensor = torch.from_numpy(pose).float().to(device)
    orient_tensor = torch.from_numpy(orient).float().to(device)
    quaternion = axis_angle_to_quaternion(orient_tensor)
    translation_tensor = torch.from_numpy(translation).float().to(device)
    vertices = renderer.model.mano(
        pose_tensor, quaternion, translation_tensor, renderer.betas)[0]
    uv, depth = project_points(vertices.cpu().numpy(), np.eye(4), intrinsics)
    valid = depth > .01
    if not valid.any():
        return None
    box = square_crop(uv[valid], frame_width, frame_height, crop_scale)
    if box is None:
        return None
    crop_k = crop_intrinsics(intrinsics, box, render_size)
    result = render_gaussian_hand(
        renderer.model, pose_tensor, quaternion, translation_tensor,
        torch.eye(4, dtype=torch.float32, device=device),
        torch.tensor(crop_k, dtype=torch.float32, device=device),
        render_size, render_size,
    )
    rgb = result["rgb"].clamp(0, 1).cpu().numpy()
    alpha = result["alpha"].clamp(0, 1).cpu().numpy()
    support = result["mesh_mask"].clamp(0, 1).cpu().numpy()
    return tuple(map(int, box)), rgb, alpha, support


def paste_hand(
    frame_rgb: np.ndarray,
    gaussian_rgb: np.ndarray,
    mask: np.ndarray,
    rendered: tuple[
        tuple[int, int, int, int], np.ndarray, np.ndarray, np.ndarray],
) -> None:
    (x0, y0, x1, y1), patch_rgb, patch_alpha, patch_support = rendered
    size = (x1 - x0, y1 - y0)
    patch_rgb = cv2.resize(patch_rgb, size, interpolation=cv2.INTER_LINEAR)
    patch_alpha = cv2.resize(
        patch_alpha[..., 0], size, interpolation=cv2.INTER_LINEAR)[..., None]
    patch_support = cv2.resize(
        patch_support[..., 0], size, interpolation=cv2.INTER_NEAREST)[..., None]
    target = frame_rgb[y0:y1, x0:x1]
    target[:] = patch_rgb + (1.0 - patch_alpha) * target
    gaussian_target = gaussian_rgb[y0:y1, x0:x1]
    gaussian_target[:] = patch_rgb + (1.0 - patch_alpha) * gaussian_target
    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], patch_support)


def mask_views(source: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = mask[..., 0] >= .5
    overlay = source.copy()
    green = np.zeros_like(overlay)
    green[..., 1] = 255
    overlay[selected] = np.rint(
        .55 * overlay[selected] + .45 * green[selected]).astype(np.uint8)
    contours, _ = cv2.findContours(
        selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 3, cv2.LINE_AA)
    masked = np.zeros_like(source)
    masked[selected] = source[selected]
    return overlay, masked


@torch.no_grad()
def main() -> None:
    args = arguments()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.session / "metadata.json").read_text())
    frame_count = int(metadata["frame_count"])
    if not 0 <= args.start_frame < frame_count:
        raise ValueError(f"--start-frame must be in [0, {frame_count - 1}]")
    stop_frame = min(frame_count, args.start_frame + args.frames)
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
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    panel_width = round(width * args.panel_height / height)
    output_width = 3 * panel_width
    mask_path = args.output / f"{args.session.name}_mano_mask_quality.mp4"
    overlay_path = args.output / f"{args.session.name}_gaussian_overlay.mp4"
    mask_encoder = start_encoder(mask_path, output_width, args.panel_height, fps)
    overlay_encoder = start_encoder(
        overlay_path, output_width, args.panel_height, fps)
    processed = 0
    try:
        for frame_index in range(args.start_frame, stop_frame):
            ok, source_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Video decode failed at frame {frame_index}")
            source_rgb = source_bgr[..., ::-1].astype(np.float32) / 255.0
            composite_rgb = source_rgb.copy()
            gaussian_rgb = np.zeros_like(source_rgb)
            union_mask = np.zeros((height, width, 1), dtype=np.float32)
            for side in hands:
                rendered = render_hand(
                    renderers[side], arrays[f"{side}_hand_pose"][frame_index],
                    arrays[f"{side}_global_orient"][frame_index],
                    arrays[f"{side}_transl"][frame_index], intrinsics,
                    width, height, args.render_size, args.crop_scale, args.device)
                if rendered is not None:
                    paste_hand(composite_rgb, gaussian_rgb, union_mask, rendered)
            composite_bgr = np.rint(
                np.clip(composite_rgb[..., ::-1], 0, 1) * 255).astype(np.uint8)
            gaussian_bgr = np.rint(
                np.clip(gaussian_rgb[..., ::-1], 0, 1) * 255).astype(np.uint8)
            mask_overlay, masked_source = mask_views(source_bgr, union_mask)
            mask_row = np.concatenate([
                panel(source_bgr, args.panel_height, "SOURCE RGB"),
                panel(mask_overlay, args.panel_height, "MANO MASK (NO SAM)"),
                panel(masked_source, args.panel_height, "TRAINING PIXELS"),
            ], axis=1)
            overlay_row = np.concatenate([
                panel(source_bgr, args.panel_height, "SOURCE RGB"),
                panel(gaussian_bgr, args.panel_height, "GAUSSIAN RENDER"),
                panel(composite_bgr, args.panel_height, "GAUSSIAN OVERLAY"),
            ], axis=1)
            if mask_encoder.stdin is None or overlay_encoder.stdin is None:
                raise RuntimeError("ffmpeg encoder pipe closed")
            mask_encoder.stdin.write(mask_row.tobytes())
            overlay_encoder.stdin.write(overlay_row.tobytes())
            processed += 1
            if processed % 30 == 0:
                print(json.dumps({"rendered_frames": processed}), flush=True)
    finally:
        capture.release()
        arrays.close()
        finish_encoder(mask_encoder, "mask")
        finish_encoder(overlay_encoder, "overlay")
    report = {
        "session": args.session.name,
        "source_video": str(args.session / "rgb.mp4"),
        "start_frame": args.start_frame,
        "stop_frame_exclusive": args.start_frame + processed,
        "frames": processed,
        "fps": fps,
        "duration_seconds": processed / fps,
        "hands": hands,
        "mask_source": "rasterized fixed-shape MANO triangle silhouette; SAM not used",
        "checkpoints": {side: str(checkpoint_by_side[side]) for side in hands},
        "render_size_per_hand": args.render_size,
        "outputs": [mask_path.name, overlay_path.name],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
