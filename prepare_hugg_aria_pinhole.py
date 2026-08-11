#!/usr/bin/env python3
"""Create a frame-aligned official HOT3D/Aria pinhole video for one sequence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi
from projectaria_tools.core.calibration import (
    FISHEYE624,
    LINEAR,
    distort_by_calibration,
)
from projectaria_tools.core.stream_id import StreamId


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "hot3d/hot3d"))
from data_loaders.AriaDataProvider import AriaDataProvider


SUPPORT_FILES = (
    "headset_trajectory.csv",
    "mano_hand_pose_trajectory.jsonl",
    "umetrack_hand_pose_trajectory.jsonl",
    "umetrack_hand_user_profile.json",
    "timecode_devicetime_mapping.csv",
    "license.txt",
)

QA_FILES = (
    "mask_hand_pose_available.csv",
    "mask_headset_pose_available.csv",
    "mask_hand_visible.csv",
    "mask_good_exposure.csv",
    "mask_qa_pass.csv",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/HUGG_ARIA_PINHOLE",
    )
    parser.add_argument("--stream-id", default="214-1")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--codec", default="h264_nvenc")
    parser.add_argument("--cq", type=int, default=20)
    parser.add_argument("--upload-repo")
    parser.add_argument("--source-repo", default="LIDAR-GT/HUGG_ARIA")
    parser.add_argument("--source-revision", default="main")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-show_entries",
            "format=duration,size,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def as_float_list(values) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def calibration_record(frame_index, timestamp, linear, t_device_camera) -> dict:
    width, height = [int(value) for value in linear.get_image_size()]
    return {
        "frame_index": frame_index,
        "timestamp_ns": int(timestamp),
        "time_domain": "TIME_CODE",
        "camera_model": "LINEAR",
        "image_width": width,
        "image_height": height,
        "projection_params": as_float_list(linear.projection_params()),
        "focal_lengths": as_float_list(linear.get_focal_lengths()),
        "principal_point": as_float_list(linear.get_principal_point()),
        "T_device_camera": np.asarray(t_device_camera.to_matrix(), dtype=float).tolist(),
    }


def copy_support_files(sequence: Path, destination: Path) -> list[str]:
    copied = []
    for name in SUPPORT_FILES:
        source = sequence / name
        if source.exists():
            shutil.copy2(source, destination / name)
            copied.append(name)
    source_metadata = sequence / "metadata.json"
    if source_metadata.exists():
        shutil.copy2(source_metadata, destination / "source_metadata.json")
        copied.append("source_metadata.json")
    source_masks = sequence / "masks"
    if source_masks.exists():
        target_masks = destination / "masks"
        target_masks.mkdir()
        for name in QA_FILES:
            source = source_masks / name
            if source.exists():
                shutil.copy2(source, target_masks / name)
                copied.append(f"masks/{name}")
    return copied


def encode_command(path: Path, width: int, height: int, args) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(args.video_fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        args.codec,
        "-preset",
        "p5",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        str(args.cq),
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(round(args.video_fps)),
        "-movflags",
        "+faststart",
        str(path),
    ]


def process_sequence(args: argparse.Namespace) -> Path:
    sequence = args.sequence.resolve()
    vrs = sequence / "recording.vrs"
    online_calibration = sequence / "mps/slam/online_calibration.jsonl"
    for required in (vrs, online_calibration):
        if not required.exists():
            raise FileNotFoundError(required)

    args.output_root.mkdir(parents=True, exist_ok=True)
    final = args.output_root / sequence.name
    success = final / "_SUCCESS.json"
    if success.exists() and not args.overwrite:
        print(json.dumps({"sequence": sequence.name, "status": "already_complete"}))
        return final
    if final.exists():
        if not args.overwrite:
            raise FileExistsError(f"Incomplete output exists: {final}")
        shutil.rmtree(final)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{sequence.name}.partial.", dir=args.output_root)
    )
    video = temporary / "rgb_214_1_pinhole.mp4"
    timestamps_csv = temporary / "frame_timestamps_214_1.csv"
    calibration_jsonl = temporary / "frame_pinhole_calibration_214_1.jsonl"
    provider = AriaDataProvider(str(vrs), str(sequence / "mps"))
    stream = StreamId(args.stream_id)
    timestamps = provider.get_sequence_timestamps(stream)
    if not timestamps:
        raise RuntimeError(f"No frames for {args.stream_id}")

    sample_indices = {0, len(timestamps) // 2, len(timestamps) - 1}
    sample_differences = []
    encoder = None
    started = time.perf_counter()
    try:
        with timestamps_csv.open("w", newline="") as timestamp_handle, calibration_jsonl.open(
            "w"
        ) as calibration_handle:
            timestamp_writer = csv.writer(timestamp_handle)
            timestamp_writer.writerow(["frame_index", "timestamp_ns", "time_domain"])
            for frame_index, timestamp in enumerate(timestamps):
                raw = provider.get_image(timestamp, stream)
                if raw is None:
                    raise RuntimeError(f"Missing frame {frame_index} at {timestamp}")
                t_device_camera, native = provider.get_online_camera_calibration(
                    stream, timestamp, camera_model=FISHEYE624
                )
                _, linear = provider.get_online_camera_calibration(
                    stream, timestamp, camera_model=LINEAR
                )
                frame = distort_by_calibration(raw, linear, native)
                if encoder is None:
                    height, width = frame.shape[:2]
                    encoder = subprocess.Popen(
                        encode_command(video, width, height, args),
                        stdin=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                if frame_index in sample_indices:
                    official = provider.get_undistorted_image(timestamp, stream)
                    difference = np.abs(
                        frame.astype(np.int16) - official.astype(np.int16)
                    )
                    sample_differences.append(
                        {
                            "frame_index": frame_index,
                            "max_abs_difference": int(difference.max()),
                            "mean_abs_difference": float(difference.mean()),
                        }
                    )
                encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
                timestamp_writer.writerow([frame_index, int(timestamp), "TIME_CODE"])
                calibration_handle.write(
                    json.dumps(
                        calibration_record(
                            frame_index, timestamp, linear, t_device_camera
                        ),
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        encoder.stdin.close()
        stderr = encoder.stderr.read().decode("utf-8", errors="replace")
        return_code = encoder.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr}")

        probe = ffprobe(video)
        stream_info = probe["streams"][0]
        encoded_frames = int(stream_info["nb_read_frames"])
        if encoded_frames != len(timestamps):
            raise RuntimeError(
                f"Frame mismatch: expected {len(timestamps)}, encoded {encoded_frames}"
            )
        if any(item["max_abs_difference"] != 0 for item in sample_differences):
            raise RuntimeError(
                f"Official undistort mismatch: {sample_differences}"
            )

        copied = copy_support_files(sequence, temporary)
        elapsed = time.perf_counter() - started
        metadata = {
            "format_version": 1,
            "sequence": sequence.name,
            "source_repo": args.source_repo,
            "source_revision": args.source_revision,
            "source_stream_id": args.stream_id,
            "output_camera_model": "LINEAR",
            "method": (
                "Project Aria per-frame online FISHEYE624 calibration warped to the "
                "per-frame LINEAR calibration returned by the official HOT3D API"
            ),
            "frames": len(timestamps),
            "first_timestamp_ns": int(timestamps[0]),
            "last_timestamp_ns": int(timestamps[-1]),
            "duration_ns": int(timestamps[-1] - timestamps[0]),
            "video_fps": args.video_fps,
            "codec": args.codec,
            "cq": args.cq,
            "processing_seconds": elapsed,
            "processing_fps": len(timestamps) / elapsed,
            "official_pixel_equivalence_samples": sample_differences,
            "ffprobe": probe,
            "support_files": copied,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        hashes = {
            path.name: sha256_file(path)
            for path in (video, timestamps_csv, calibration_jsonl)
        }
        success_payload = {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "frames": len(timestamps),
            "hashes_sha256": hashes,
        }
        (temporary / "_SUCCESS.json").write_text(
            json.dumps(success_payload, indent=2) + "\n"
        )
        temporary.rename(final)
        print(json.dumps({**metadata, "output": str(final)}, indent=2))
        return final
    except BaseException:
        if encoder is not None and encoder.poll() is None:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.kill()
            encoder.wait()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def upload_sequence(folder: Path, repo_id: str) -> None:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo=folder.name,
        commit_message=f"Add official pinhole sequence {folder.name}",
    )
    print(json.dumps({"sequence": folder.name, "uploaded_to": repo_id}))


def main() -> None:
    args = arguments()
    folder = process_sequence(args)
    if args.upload_repo:
        upload_sequence(folder, args.upload_repo)


if __name__ == "__main__":
    main()
