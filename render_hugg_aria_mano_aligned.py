#!/usr/bin/env python3
"""Render frame-aligned MANO RGB plus lossless alpha for HUGG Aria.

SAM training eligibility is read directly from the per-sequence SQLite assets.
Every source frame is emitted, and the final manifest later adds the nonempty
MANO-alpha criterion to the existing SAM/QA filters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "hot3d/hot3d"))

from data_loaders.HandDataProviderBase import Handedness  # noqa: E402
from data_loaders.HeadsetPose3dProvider import (  # noqa: E402
    load_headset_pose_provider_from_csv,
)
from data_loaders.ManoHandDataProvider import MANOHandDataProvider  # noqa: E402
from data_loaders.mano_layer import MANOHandModel  # noqa: E402
from hand_tracking_toolkit import rasterizer  # noqa: E402
from hand_tracking_toolkit.camera import PinholePlaneCameraModel  # noqa: E402

import hugg_aria_sam_policy as policy  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=ROOT / "outputs/sam2_hugg_aria_masks_v3_pilot",
    )
    parser.add_argument("--expected-sequences", type=int, default=136)
    parser.add_argument(
        "--pinhole-root", type=Path, default=ROOT / "data/HUGG_ARIA_PINHOLE"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "data/HUGG_ARIA_MANO_ALIGNED"
    )
    parser.add_argument(
        "--mano-model-dir", type=Path, default=ROOT / "mano_v1_2/models"
    )
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--codec", choices=("h264_nvenc", "libx264"), default="libx264")
    parser.add_argument("--cq", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


SEQUENCE_RE = re.compile(r"^P[0-9]+_[0-9a-f]+$")


def load_training_eligible(
    mask_root: Path, expected_sequences: int
) -> dict[str, set[int]]:
    grouped: dict[str, set[int]] = {}
    sequence_dirs = sorted(
        path
        for path in mask_root.iterdir()
        if path.is_dir()
        and SEQUENCE_RE.fullmatch(path.name)
        and (path / "masks.sqlite").is_file()
        and (path / "_SUCCESS.json").is_file()
    )
    if len(sequence_dirs) != expected_sequences:
        raise RuntimeError(
            f"Expected {expected_sequences} complete SAM sequences, "
            f"found {len(sequence_dirs)}"
        )
    for path in sequence_dirs:
        connection = sqlite3.connect(path / "masks.sqlite")
        try:
            rows = connection.execute(
                "SELECT frame_index FROM frames "
                "WHERE training_eligible=1 ORDER BY frame_index"
            ).fetchall()
        finally:
            connection.close()
        grouped[path.name] = {int(row[0]) for row in rows}
    return grouped


def eligible_sha256(eligible: set[int]) -> str:
    payload = ",".join(str(value) for value in sorted(eligible)).encode()
    return hashlib.sha256(payload).hexdigest()


def read_timestamps(path: Path) -> list[int]:
    with path.open(newline="") as handle:
        return [int(row["timestamp_ns"]) for row in csv.DictReader(handle)]


def read_calibrations(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scaled_camera(calibration: dict, t_camera_world: np.ndarray, size: int):
    width = int(calibration["image_width"])
    height = int(calibration["image_height"])
    sx, sy = size / width, size / height
    fx, fy = (float(value) for value in calibration["focal_lengths"])
    cx, cy = (float(value) for value in calibration["principal_point"])
    # OpenCV resize maps pixel centres with (x + 0.5) * scale - 0.5.
    center = ((cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5)
    return PinholePlaneCameraModel(
        size,
        size,
        (fx * sx, fy * sy),
        center,
        (),
        T_world_from_eye=np.linalg.inv(t_camera_world),
    )


def render_frame(
    timestamp: int,
    calibration: dict,
    size: int,
    model,
    hand_provider,
    headset_provider,
) -> tuple[np.ndarray, np.ndarray]:
    collection = policy.pose_collection(hand_provider, timestamp)
    t_camera_world = policy.world_to_camera(
        calibration, headset_provider, timestamp
    )
    if collection is None or t_camera_world is None:
        raise RuntimeError("missing MANO or headset pose")
    camera = scaled_camera(calibration, t_camera_world, size)
    image = np.zeros((size, size, 3), np.uint8)
    alpha = np.zeros((size, size), np.uint8)
    z_buffer = np.full((size, size), np.inf, np.float32)
    rendered_hands = 0
    for handedness, layer in (
        (Handedness.Left, model.mano_layer_left),
        (Handedness.Right, model.mano_layer_right),
    ):
        pose = collection.poses.get(handedness)
        if pose is None:
            continue
        vertices = hand_provider.get_hand_mesh_vertices(pose)
        if vertices is None:
            continue
        shaded, mask, depth = rasterizer.rasterize_mesh(
            verts=vertices.detach().cpu().numpy(),
            faces=layer.faces.astype(np.int64),
            camera=camera,
            vert_normals=None,
            ambient=(0.22, 0.22, 0.22),
            diffuse=(0.58, 0.58, 0.58),
            specular=(0.08, 0.08, 0.08),
            shininess=24,
        )
        update = mask.astype(bool) & (depth > 0) & (depth < z_buffer)
        if np.any(update):
            image[update] = shaded[update]
            alpha[update] = 255
            z_buffer[update] = depth[update]
            rendered_hands += 1
    if rendered_hands == 0:
        raise RuntimeError("MANO raster is empty")
    return image, alpha


def encode_command(
    output: Path, size: int, fps: float, codec: str, cq: int
) -> list[str]:
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", codec,
    ]
    if codec == "h264_nvenc":
        command += [
            "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(cq),
            "-b:v", "0",
        ]
    else:
        command += ["-preset", "veryfast", "-crf", str(cq)]
    return command + [
        "-pix_fmt", "yuv420p", "-g", str(round(fps)),
        "-movflags", "+faststart", str(output),
    ]


def encode_alpha_command(output: Path, size: int, fps: float) -> list[str]:
    return [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "gray", "-s", f"{size}x{size}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "ffv1", "-level", "3",
        "-pix_fmt", "gray", str(output),
    ]


def validate_video(path: Path, expected_frames: int, size: int) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open rendered video: {path}")
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    if (frames, width, height) != (expected_frames, size, size):
        raise RuntimeError(
            f"Encoded video mismatch: got frames/size={frames}/{width}x{height}, "
            f"expected {expected_frames}/{size}x{size}"
        )


def process_sequence(task: tuple) -> dict:
    (
        sequence,
        eligible,
        pinhole_root,
        output_root,
        mano_model_dir,
        size,
        fps,
        codec,
        cq,
        overwrite,
        eligible_digest,
        torch_threads,
    ) = task
    import torch

    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    source = Path(pinhole_root) / sequence
    output_root = Path(output_root)
    final = output_root / sequence
    success = final / "_SUCCESS.json"
    if success.is_file() and not overwrite:
        metadata = json.loads(success.read_text())
        expected = {
            "format_version": 3,
            "eligible_sha256": eligible_digest,
            "eligible_frames": len(eligible),
            "output_size": size,
        }
        complete = (
            all(metadata.get(key) == value for key, value in expected.items())
            and (final / "reconstruction.mp4").is_file()
            and (final / "alpha.mkv").is_file()
            and (final / "mano_availability.csv").is_file()
        )
        if complete:
            return {"sequence": sequence, "status": "already_complete"}
        raise RuntimeError(f"Stale MANO render metadata: {success}")
    if final.exists() and not overwrite:
        raise FileExistsError(f"Incomplete MANO output exists: {final}")

    required = (
        source / "_SUCCESS.json",
        source / "frame_timestamps_214_1.csv",
        source / "frame_pinhole_calibration_214_1.jsonl",
        source / "mano_hand_pose_trajectory.jsonl",
        source / "headset_trajectory.csv",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    total_frames = int(json.loads(required[0].read_text())["frames"])
    timestamps = read_timestamps(required[1])
    calibrations = read_calibrations(required[2])
    if len(timestamps) != total_frames or len(calibrations) != total_frames:
        raise RuntimeError(f"Frame sidecar mismatch for {sequence}")
    if not eligible or min(eligible) < 0 or max(eligible) >= total_frames:
        raise RuntimeError(f"Eligible frame range is invalid for {sequence}")

    model = MANOHandModel(str(mano_model_dir))
    hand_provider = MANOHandDataProvider(str(required[3]), model)
    headset_provider = load_headset_pose_provider_from_csv(str(required[4]))
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{sequence}.partial.", dir=output_root)
    )
    rgb_path = temporary / "reconstruction.mp4"
    alpha_path = temporary / "alpha.mkv"
    rgb_encoder = subprocess.Popen(
        encode_command(rgb_path, size, fps, codec, cq),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    alpha_encoder = subprocess.Popen(
        encode_alpha_command(alpha_path, size, fps),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    blank_rgb = np.zeros((size, size, 3), np.uint8)
    blank_alpha = np.zeros((size, size), np.uint8)
    availability = []
    started = time.perf_counter()
    encoders = (("RGB", rgb_encoder), ("alpha", alpha_encoder))
    try:
        for frame_index, (timestamp, calibration) in enumerate(
            zip(timestamps, calibrations)
        ):
            rgb, alpha = blank_rgb, blank_alpha
            if frame_index in eligible:
                try:
                    rgb, alpha = render_frame(
                        timestamp,
                        calibration,
                        size,
                        model,
                        hand_provider,
                        headset_provider,
                    )
                    availability.append((frame_index, 1, "ok"))
                except RuntimeError as exc:
                    reason = str(exc)
                    if reason not in {
                        "missing MANO or headset pose",
                        "MANO raster is empty",
                    }:
                        raise
                    availability.append((frame_index, 0, reason))
            for name, encoder, frame in (
                ("RGB", rgb_encoder, rgb),
                ("alpha", alpha_encoder, alpha),
            ):
                try:
                    encoder.stdin.write(
                        np.ascontiguousarray(frame).tobytes()
                    )
                except BrokenPipeError as exc:
                    stderr = encoder.stderr.read().decode(
                        "utf-8", errors="replace"
                    )
                    encoder.wait()
                    raise RuntimeError(
                        f"{name} ffmpeg pipe failed: {stderr}"
                    ) from exc

        for _, encoder in encoders:
            encoder.stdin.close()
        failures = []
        for name, encoder in encoders:
            stderr = encoder.stderr.read().decode(
                "utf-8", errors="replace"
            )
            return_code = encoder.wait()
            if return_code:
                failures.append(f"{name} ffmpeg ({return_code}): {stderr}")
        if failures:
            raise RuntimeError("; ".join(failures))

        validate_video(rgb_path, total_frames, size)
        validate_video(alpha_path, total_frames, size)
        with (temporary / "mano_availability.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(("frame_index", "mano_valid", "reason"))
            writer.writerows(availability)

        valid_count = sum(row[1] for row in availability)
        metadata = {
            "format_version": 3,
            "sequence": sequence,
            "render_kind": "mano",
            "frame_alignment": "output_frame_index_equals_source_frame_index",
            "rgb_semantics": "straight_rgb; composite_with_alpha",
            "alpha_semantics": "binary_visible_mano_raster",
            "alpha_filename": "alpha.mkv",
            "alpha_codec": "ffv1_lossless_gray8",
            "total_frames": total_frames,
            "eligible_frames": len(eligible),
            "mano_valid_frames": valid_count,
            "mano_invalid_eligible_frames": len(eligible) - valid_count,
            "blank_frames": total_frames - valid_count,
            "output_size": size,
            "fps": fps,
            "rgb_codec": codec,
            "cq": cq,
            "eligible_sha256": eligible_digest,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (temporary / "_SUCCESS.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        if final.exists():
            shutil.rmtree(final)
        temporary.replace(final)
        return {"sequence": sequence, "status": "rendered", **metadata}
    except BaseException:
        for _, encoder in encoders:
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = arguments()
    pinhole_root = args.pinhole_root.resolve()
    readme = pinhole_root / "README.md"
    if not readme.is_file() or "# HUGG ARIA Pinhole" not in (
        readme.read_text(encoding="utf-8")
    ):
        raise RuntimeError(
            "pinhole_root must be a local snapshot of "
            "LIDAR-GT/HUGG_ARIA_PINHOLE"
        )

    grouped = load_training_eligible(
        args.mask_root.resolve(), args.expected_sequences
    )
    sequences = sorted(grouped)
    if args.sequence:
        requested = set(args.sequence)
        missing = sorted(requested - set(sequences))
        if missing:
            raise KeyError(f"Sequences are absent from SAM assets: {missing}")
        sequences = [
            sequence for sequence in sequences if sequence in requested
        ]
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise RuntimeError("No sequences selected")

    tasks = [
        (
            sequence,
            grouped[sequence],
            str(pinhole_root),
            str(args.output_root.resolve()),
            str(args.mano_model_dir.resolve()),
            args.size,
            args.fps,
            args.codec,
            args.cq,
            args.overwrite,
            eligible_sha256(grouped[sequence]),
            args.torch_threads,
        )
        for sequence in sequences
    ]
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        futures = [
            executor.submit(process_sequence, task) for task in tasks
        ]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
