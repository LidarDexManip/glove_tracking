#!/usr/bin/env python3
"""Segment one derived HUGG Aria pinhole sequence with chunked SAM2 video propagation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

# chumpy/MANO compatibility with NumPy >= 2.
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

from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from sam2.build_sam import build_sam2_video_predictor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "hot3d/hot3d"))
from data_loaders.HeadsetPose3dProvider import load_headset_pose_provider_from_csv
from data_loaders.ManoHandDataProvider import MANOHandDataProvider
from data_loaders.loader_hand_poses import Handedness
from data_loaders.mano_layer import MANOHandModel


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/sam2_hugg_aria_masks")
    parser.add_argument("--chunk-frames", type=int, default=300)
    parser.add_argument("--prompt-stride", type=int, default=5)
    parser.add_argument("--box-padding", type=float, default=0.08)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-chunks", type=int, default=0, help="Pilot/debug limit; zero processes all chunks.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model/sam2/sam2.1_hiera_large.pt")
    return parser.parse_args()


def load_timestamps(path: Path) -> list[int]:
    with path.open(newline="") as handle:
        return [int(row["timestamp_ns"]) for row in csv.DictReader(handle)]


def load_calibrations(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def project_mano_boxes(
    timestamp: int,
    calibration: dict,
    hand_provider,
    headset_provider,
    padding: float,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    poses = hand_provider.get_pose_at_timestamp(
        timestamp, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE, 1_000_000
    )
    headset = headset_provider.get_pose_at_timestamp(
        timestamp, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE, 1_000_000
    )
    if poses is None or headset is None:
        return {}, {}

    t_world_device = np.asarray(headset.pose3d.T_world_device.to_matrix(), dtype=np.float64)
    t_device_camera = np.asarray(calibration["T_device_camera"], dtype=np.float64)
    t_camera_world = np.linalg.inv(t_world_device @ t_device_camera)
    fx, fy = (float(value) for value in calibration["focal_lengths"])
    cx, cy = (float(value) for value in calibration["principal_point"])
    width, height = int(calibration["image_width"]), int(calibration["image_height"])

    boxes: dict[int, np.ndarray] = {}
    visible_counts: dict[int, int] = {}
    for handedness, pose in poses.pose3d_collection.poses.items():
        vertices = hand_provider.get_hand_mesh_vertices(pose)
        if vertices is None:
            continue
        vertices = vertices.detach().cpu().numpy().astype(np.float64)
        camera = (t_camera_world[:3, :3] @ vertices.T + t_camera_world[:3, 3:4]).T
        positive = camera[:, 2] > 0.01
        uv = np.empty((len(camera), 2), dtype=np.float64)
        uv[:, 0] = fx * camera[:, 0] / np.maximum(camera[:, 2], 1e-8) + cx
        uv[:, 1] = fy * camera[:, 1] / np.maximum(camera[:, 2], 1e-8) + cy
        inside = (
            positive
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] <= height - 1)
        )
        count = int(inside.sum())
        if count < 40:
            continue
        xy = uv[inside]
        x1, y1 = xy.min(axis=0)
        x2, y2 = xy.max(axis=0)
        if x2 - x1 < 16 or y2 - y1 < 16 or (x2 - x1) * (y2 - y1) < 800:
            continue
        dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
        box = np.array(
            [max(0, x1 - dx), max(0, y1 - dy), min(width - 1, x2 + dx), min(height - 1, y2 + dy)],
            dtype=np.float32,
        )
        hand = int(handedness.value)
        boxes[hand] = box
        visible_counts[hand] = count
    return boxes, visible_counts


def candidate_order(length: int, stride: int) -> list[int]:
    candidates = list(range(0, length, max(1, stride)))
    if length - 1 not in candidates:
        candidates.append(length - 1)
    center = (length - 1) / 2.0
    return sorted(candidates, key=lambda index: abs(index - center))


def choose_prompt(
    start: int,
    stop: int,
    timestamps: list[int],
    calibrations: list[dict],
    hand_provider,
    headset_provider,
    stride: int,
    padding: float,
) -> tuple[int | None, dict[int, np.ndarray], dict[int, int]]:
    best = (0, -1.0, None, {}, {})
    for local in candidate_order(stop - start, stride):
        frame = start + local
        boxes, visible = project_mano_boxes(
            timestamps[frame], calibrations[frame], hand_provider, headset_provider, padding
        )
        if not boxes:
            continue
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes.values()]
        score = (len(boxes), float(min(areas)))
        if score > best[:2]:
            best = (score[0], score[1], local, boxes, visible)
        if len(boxes) == 2 and min(visible.values()) >= 250:
            return local, boxes, visible
    return best[2], best[3], best[4]


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_index INTEGER PRIMARY KEY,
            start_frame INTEGER NOT NULL,
            stop_frame_exclusive INTEGER NOT NULL,
            prompt_frame INTEGER,
            prompt_boxes_json TEXT NOT NULL,
            visible_vertices_json TEXT NOT NULL,
            status TEXT NOT NULL,
            inference_seconds REAL NOT NULL,
            completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS frames (
            frame_index INTEGER PRIMARY KEY,
            timestamp_ns INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            labels_zlib BLOB NOT NULL,
            area_left INTEGER NOT NULL,
            area_right INTEGER NOT NULL,
            FOREIGN KEY(chunk_index) REFERENCES chunks(chunk_index)
        );
        CREATE INDEX IF NOT EXISTS frames_chunk_index ON frames(chunk_index);
        """
    )


def set_metadata(connection: sqlite3.Connection, key: str, value) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value_json) VALUES(?,?)",
        (key, json.dumps(value, separators=(",", ":"))),
    )


def completed_chunks(connection: sqlite3.Connection) -> set[int]:
    return {
        int(row[0])
        for row in connection.execute("SELECT chunk_index FROM chunks WHERE status IN ('complete','no_prompt')")
    }


def labels_from_logits(object_ids, mask_logits: torch.Tensor) -> np.ndarray:
    logits = mask_logits[:, 0].detach().float().cpu().numpy()
    winner = np.argmax(logits, axis=0)
    confidence = np.max(logits, axis=0)
    labels = np.zeros(confidence.shape, dtype=np.uint8)
    ids = [int(value) for value in object_ids]
    for output_index, object_id in enumerate(ids):
        labels[(winner == output_index) & (confidence > 0.0)] = object_id + 1
    return labels


def insert_frame(
    connection: sqlite3.Connection,
    frame_index: int,
    timestamp: int,
    chunk_index: int,
    labels: np.ndarray,
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO frames VALUES(?,?,?,?,?,?)",
        (
            frame_index,
            timestamp,
            chunk_index,
            sqlite3.Binary(zlib.compress(labels.tobytes(), 3)),
            int((labels == 1).sum()),
            int((labels == 2).sum()),
        ),
    )


def draw_prompt(path: Path, frame: np.ndarray, chunk: int, source_frame: int, boxes) -> None:
    image = frame.copy()
    colors = {0: (255, 180, 30), 1: (30, 140, 255)}
    for hand, box in boxes.items():
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[hand], 3)
        cv2.putText(image, "left" if hand == 0 else "right", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .7, colors[hand], 2, cv2.LINE_AA)
    cv2.putText(image, f"chunk={chunk} frame={source_frame}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, f"chunk={chunk} frame={source_frame}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write {path}")


def main() -> None:
    args = arguments()
    sequence = args.sequence.resolve()
    required = (
        sequence / "_SUCCESS.json",
        sequence / "rgb_214_1_pinhole.mp4",
        sequence / "frame_timestamps_214_1.csv",
        sequence / "frame_pinhole_calibration_214_1.jsonl",
        sequence / "mano_hand_pose_trajectory.jsonl",
        sequence / "headset_trajectory.csv",
        args.checkpoint,
        ROOT / "mano_v1_2/models/MANO_LEFT.pkl",
        ROOT / "mano_v1_2/models/MANO_RIGHT.pkl",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    success = json.loads((sequence / "_SUCCESS.json").read_text())
    timestamps = load_timestamps(sequence / "frame_timestamps_214_1.csv")
    calibrations = load_calibrations(sequence / "frame_pinhole_calibration_214_1.jsonl")
    total_frames = int(success["frames"])
    if len(timestamps) != total_frames or len(calibrations) != total_frames:
        raise RuntimeError("Frame/timestamp/calibration count mismatch")

    output = args.output_root.resolve() / sequence.name
    final_db = output / "masks.sqlite"
    partial_db = output / "masks.sqlite.partial"
    success_path = output / "_SUCCESS.json"
    if success_path.exists() and final_db.exists() and not args.overwrite:
        print(json.dumps({"sequence": sequence.name, "status": "already_complete"}))
        return
    if args.overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    prompts_dir = output / "prompt_frames"
    prompts_dir.mkdir(exist_ok=True)

    connection = sqlite3.connect(partial_db)
    create_schema(connection)
    for key, value in {
        "format_version": 1,
        "sequence": sequence.name,
        "height": int(calibrations[0]["image_height"]),
        "width": int(calibrations[0]["image_width"]),
        "total_frames": total_frames,
        "chunk_frames": args.chunk_frames,
        "labels": {"0": "background", "1": "left_hand", "2": "right_hand"},
        "compression": "zlib level 3 over row-major uint8 label image",
        "model": "sam2.1_hiera_large",
        "checkpoint": str(args.checkpoint.resolve()),
        "source_sequence": str(sequence),
    }.items():
        set_metadata(connection, key, value)
    connection.commit()

    mano_model = MANOHandModel(str(ROOT / "mano_v1_2/models"))
    hand_provider = MANOHandDataProvider(str(sequence / "mano_hand_pose_trajectory.jsonl"), mano_model)
    headset_provider = load_headset_pose_provider_from_csv(str(sequence / "headset_trajectory.csv"))
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(args.checkpoint),
        device="cuda",
        apply_postprocessing=True,
    )
    torch.cuda.reset_peak_memory_stats()
    capture = cv2.VideoCapture(str(sequence / "rgb_214_1_pinhole.mp4"))
    if not capture.isOpened():
        raise RuntimeError("Could not open pinhole video")

    done = completed_chunks(connection)
    chunk_count = (total_frames + args.chunk_frames - 1) // args.chunk_frames
    run_chunks = 0
    run_started = time.perf_counter()
    try:
        for chunk_index in range(chunk_count):
            if chunk_index in done:
                continue
            if args.max_chunks and run_chunks >= args.max_chunks:
                break
            start = chunk_index * args.chunk_frames
            stop = min(total_frames, start + args.chunk_frames)
            prompt_local, boxes, visible = choose_prompt(
                start,
                stop,
                timestamps,
                calibrations,
                hand_provider,
                headset_provider,
                args.prompt_stride,
                args.box_padding,
            )

            temporary = Path(tempfile.mkdtemp(prefix=f"sam2_{sequence.name}_{chunk_index:04d}_", dir="/tmp"))
            frames_dir = temporary / "frames"
            frames_dir.mkdir()
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            decoded = []
            for local in range(stop - start):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode frame {start + local}")
                decoded.append(frame if local == prompt_local else None)
                path = frames_dir / f"{local:06d}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                    raise RuntimeError(f"Could not write {path}")

            chunk_started = time.perf_counter()
            connection.execute("BEGIN")
            try:
                if prompt_local is None or not boxes:
                    zero = np.zeros(
                        (int(calibrations[0]["image_height"]), int(calibrations[0]["image_width"])),
                        dtype=np.uint8,
                    )
                    for frame in range(start, stop):
                        insert_frame(connection, frame, timestamps[frame], chunk_index, zero)
                    status = "no_prompt"
                else:
                    draw_prompt(
                        prompts_dir / f"chunk_{chunk_index:04d}_frame_{start + prompt_local:06d}.jpg",
                        decoded[prompt_local],
                        chunk_index,
                        start + prompt_local,
                        boxes,
                    )
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        state = predictor.init_state(
                            str(frames_dir),
                            offload_video_to_cpu=True,
                            offload_state_to_cpu=False,
                            async_loading_frames=True,
                        )
                        for hand, box in boxes.items():
                            predictor.add_new_points_or_box(
                                state, frame_idx=prompt_local, obj_id=hand, box=box
                            )
                        written = set()
                        for reverse in (False, True):
                            if reverse and prompt_local == 0:
                                continue
                            for local, object_ids, logits in predictor.propagate_in_video(
                                state,
                                start_frame_idx=prompt_local,
                                max_frame_num_to_track=stop - start,
                                reverse=reverse,
                            ):
                                labels = labels_from_logits(object_ids, logits)
                                frame = start + int(local)
                                insert_frame(connection, frame, timestamps[frame], chunk_index, labels)
                                written.add(frame)
                        missing = sorted(set(range(start, stop)) - written)
                        if missing:
                            raise RuntimeError(f"Chunk {chunk_index} missing frames: {missing[:20]}")
                        del state
                    status = "complete"

                elapsed = time.perf_counter() - chunk_started
                connection.execute(
                    "INSERT OR REPLACE INTO chunks(chunk_index,start_frame,stop_frame_exclusive,prompt_frame,prompt_boxes_json,visible_vertices_json,status,inference_seconds) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        chunk_index,
                        start,
                        stop,
                        None if prompt_local is None else start + prompt_local,
                        json.dumps({str(k): [float(x) for x in v] for k, v in boxes.items()}, separators=(",", ":")),
                        json.dumps({str(k): int(v) for k, v in visible.items()}, separators=(",", ":")),
                        status,
                        elapsed,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
            run_chunks += 1
            completed = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            print(
                json.dumps(
                    {
                        "sequence": sequence.name,
                        "chunk": chunk_index,
                        "chunks_complete": completed,
                        "chunks_total": chunk_count,
                        "frames": stop - start,
                        "prompt_frame": None if prompt_local is None else start + prompt_local,
                        "hands": sorted(boxes),
                        "seconds": elapsed,
                    }
                ),
                flush=True,
            )
    finally:
        capture.release()

    completed = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    frame_rows = int(connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    if completed == chunk_count and frame_rows == total_frames:
        os.replace(partial_db, final_db)
        result = {
            "status": "complete",
            "sequence": sequence.name,
            "frames": total_frames,
            "chunks": chunk_count,
            "model": "sam2.1_hiera_large",
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "run_seconds": time.perf_counter() - run_started,
            "database_bytes": final_db.stat().st_size,
        }
        success_path.write_text(json.dumps(result, indent=2) + "\n")
    else:
        result = {
            "status": "partial",
            "sequence": sequence.name,
            "frames_in_database": frame_rows,
            "chunks_complete": completed,
            "chunks_total": chunk_count,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
