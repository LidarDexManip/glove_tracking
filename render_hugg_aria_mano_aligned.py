#!/usr/bin/env python3
"""Render frame-aligned shaded MANO videos for the HUGG Aria ablation.

Only frames present in the filtered training manifest are rendered. Every other
source frame is encoded as black, so output frame indices remain identical to
the pinhole RGB video and to the aligned Gaussian renderer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
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
        "--manifest",
        type=Path,
        default=ROOT / "data/derived/hugg_aria_diffusion_aligned/train_manifest.jsonl",
    )
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


def load_manifest(path: Path) -> dict[str, set[int]]:
    grouped: dict[str, set[int]] = defaultdict(set)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            source = int(record["frame_index"])
            aligned = int(record["gaussian_frame_index"])
            if source != aligned:
                raise RuntimeError(
                    f"Manifest is not frame aligned: {record['sequence_id']} "
                    f"source={source} render={aligned}"
                )
            grouped[str(record["sequence_id"])].add(source)
    if not grouped:
        raise RuntimeError(f"Manifest has no samples: {path}")
    return dict(grouped)


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
) -> np.ndarray:
    collection = policy.pose_collection(hand_provider, timestamp)
    t_camera_world = policy.world_to_camera(
        calibration, headset_provider, timestamp
    )
    if collection is None or t_camera_world is None:
        raise RuntimeError("missing MANO or headset pose")
    camera = scaled_camera(calibration, t_camera_world, size)
    image = np.zeros((size, size, 3), np.uint8)
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
            z_buffer[update] = depth[update]
            rendered_hands += 1
    if rendered_hands == 0:
        raise RuntimeError("MANO raster is empty")
    return image


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
        manifest_hash,
        torch_threads,
    ) = task
    import torch

    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # A reused ProcessPool worker has already fixed this global value.

    source = Path(pinhole_root) / sequence
    output_root = Path(output_root)
    final = output_root / sequence
    success = final / "_SUCCESS.json"
    if success.is_file() and not overwrite:
        metadata = json.loads(success.read_text())
        expected = {
            "format_version": 2,
            "manifest_sha256": manifest_hash,
            "eligible_frames": len(eligible),
            "output_size": size,
        }
        if all(metadata.get(key) == value for key, value in expected.items()) and (
            final / "mano_availability.csv"
        ).is_file():
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
    temporary = Path(tempfile.mkdtemp(prefix=f".{sequence}.partial.", dir=output_root))
    video = temporary / "reconstruction.mp4"
    encoder = subprocess.Popen(
        encode_command(video, size, fps, codec, cq),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    blank = np.zeros((size, size, 3), np.uint8)
    availability = []
    started = time.perf_counter()
    try:
        for frame_index, (timestamp, calibration) in enumerate(
            zip(timestamps, calibrations)
        ):
            frame = blank
            if frame_index in eligible:
                try:
                    frame = render_frame(
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
                    if reason not in {"missing MANO or headset pose", "MANO raster is empty"}:
                        raise
                    availability.append((frame_index, 0, reason))
                    frame = blank
            try:
                encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
            except BrokenPipeError as exc:
                stderr = encoder.stderr.read().decode("utf-8", errors="replace")
                encoder.wait()
                raise RuntimeError(f"ffmpeg pipe failed: {stderr}") from exc
        encoder.stdin.close()
        stderr = encoder.stderr.read().decode("utf-8", errors="replace")
        return_code = encoder.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr}")
        validate_video(video, total_frames, size)
        with (temporary / "mano_availability.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("frame_index", "mano_valid", "reason"))
            writer.writerows(availability)
        valid_count = sum(row[1] for row in availability)
        metadata = {
            "format_version": 2,
            "sequence": sequence,
            "render_kind": "mano",
            "frame_alignment": "output_frame_index_equals_source_frame_index",
            "total_frames": total_frames,
            "eligible_frames": len(eligible),
            "mano_valid_frames": valid_count,
            "mano_invalid_eligible_frames": len(eligible) - valid_count,
            "blank_frames": total_frames - valid_count,
            "output_size": size,
            "fps": fps,
            "codec": codec,
            "cq": cq,
            "manifest_sha256": manifest_hash,
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
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = arguments()
    manifest = args.manifest.resolve()
    grouped = load_manifest(manifest)
    sequences = sorted(grouped)
    if args.sequence:
        requested = set(args.sequence)
        missing = sorted(requested - set(sequences))
        if missing:
            raise KeyError(f"Sequences are absent from manifest: {missing}")
        sequences = [sequence for sequence in sequences if sequence in requested]
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise RuntimeError("No sequences selected")
    manifest_hash = sha256(manifest)
    tasks = [
        (
            sequence,
            grouped[sequence],
            str(args.pinhole_root.resolve()),
            str(args.output_root.resolve()),
            str(args.mano_model_dir.resolve()),
            args.size,
            args.fps,
            args.codec,
            args.cq,
            args.overwrite,
            manifest_hash,
            args.torch_threads,
        )
        for sequence in sequences
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process_sequence, task) for task in tasks]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
