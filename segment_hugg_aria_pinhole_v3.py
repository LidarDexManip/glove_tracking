#!/usr/bin/env python3
"""SAM2 for HUGG Aria: propagate first, then filter training frames.

Propagation is interrupted only when the per-hand presence state changes.  A
new visibility episode is prompted on its first frame, preferring a MANO mesh
box when the hand-pose QA mask allows it and otherwise using the original HOT3D
box reprojected from FISHEYE624 into the derived LINEAR frame.  Frames are not
removed for training QA until after SAM output has been written.
"""
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

for _name, _value in {
    "bool": bool, "int": int, "float": float, "complex": complex,
    "object": object, "unicode": str, "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

from sam2.build_sam import build_sam2_video_predictor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "hot3d/hot3d"))
from data_loaders.HeadsetPose3dProvider import load_headset_pose_provider_from_csv
from data_loaders.ManoHandDataProvider import MANOHandDataProvider
from data_loaders.mano_layer import MANOHandModel

import hugg_aria_sam_policy as policy


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--support-root", type=Path,
                        default=ROOT / "data/HUGG_ARIA_SAM_SUPPORT")
    parser.add_argument("--mapping-root", type=Path,
                        default=ROOT / "outputs/gaussian_frame_mapping")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/sam2_hugg_aria_masks_v3")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--box-padding", type=float, default=0.04)
    parser.add_argument("--min-box-side", type=float, default=12.0)
    parser.add_argument("--min-box-area", type=float, default=500.0)
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="Pilot/resume limit; zero runs every pending episode.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "model/sam2/sam2.1_hiera_large.pt")
    return parser.parse_args()


def load_timestamps(path: Path) -> list[int]:
    with path.open(newline="") as handle:
        return [int(row["timestamp_ns"]) for row in csv.DictReader(handle)]


def load_calibrations(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS episodes (
            episode_index INTEGER PRIMARY KEY,
            start_frame INTEGER NOT NULL,
            stop_frame_exclusive INTEGER NOT NULL,
            active_hands INTEGER NOT NULL,
            prompts_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            inference_seconds REAL NOT NULL DEFAULT 0,
            completed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS frames (
            frame_index INTEGER PRIMARY KEY,
            timestamp_ns INTEGER NOT NULL,
            episode_index INTEGER,
            labels_zlib BLOB NOT NULL,
            area_left INTEGER NOT NULL DEFAULT 0,
            area_right INTEGER NOT NULL DEFAULT 0,
            sam_complete INTEGER NOT NULL DEFAULT 0,
            training_candidate INTEGER NOT NULL,
            training_eligible INTEGER NOT NULL DEFAULT 0,
            qa_pass INTEGER NOT NULL,
            mano_pose_qa_available INTEGER NOT NULL,
            hand_visible INTEGER NOT NULL,
            good_exposure INTEGER NOT NULL,
            gaussian_valid INTEGER NOT NULL,
            gaussian_left_valid INTEGER NOT NULL,
            gaussian_right_valid INTEGER NOT NULL,
            active_hands INTEGER NOT NULL,
            left_prompt_source TEXT,
            right_prompt_source TEXT,
            training_filter_reason TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS frames_episode_index ON frames(episode_index);
        CREATE INDEX IF NOT EXISTS frames_training_eligible ON frames(training_eligible);
        """
    )


def set_metadata(connection: sqlite3.Connection, key: str, value) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value_json) VALUES(?,?)",
        (key, json.dumps(value, separators=(",", ":"))),
    )


def initialize_database(connection: sqlite3.Connection, plans: list[policy.FramePlan],
                        episodes: list[policy.Episode], zero_blob: bytes) -> None:
    if connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]:
        return
    episode_by_frame = {}
    for episode in episodes:
        connection.execute(
            "INSERT INTO episodes(episode_index,start_frame,stop_frame_exclusive,"
            "active_hands,prompts_json) VALUES(?,?,?,?,?)",
            (episode.episode_index, episode.start_frame, episode.stop_frame_exclusive,
             episode.active_hands, policy.prompt_json(episode.prompts)),
        )
        for frame_index in range(episode.start_frame, episode.stop_frame_exclusive):
            episode_by_frame[frame_index] = episode.episode_index
    connection.executemany(
        """
        INSERT INTO frames(
            frame_index,timestamp_ns,episode_index,labels_zlib,training_candidate,
            qa_pass,mano_pose_qa_available,hand_visible,good_exposure,
            gaussian_valid,gaussian_left_valid,gaussian_right_valid,active_hands,
            left_prompt_source,right_prompt_source,training_filter_reason,status
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                item.frame_index, item.timestamp_ns,
                episode_by_frame.get(item.frame_index), sqlite3.Binary(zero_blob),
                int(item.training_candidate), int(item.qa_pass),
                int(item.mano_pose_qa_available), int(item.hand_visible),
                int(item.good_exposure), int(item.gaussian_valid),
                int(item.gaussian_left_valid), int(item.gaussian_right_valid),
                item.active_hands,
                None if item.left_prompt is None else item.left_prompt.source,
                None if item.right_prompt is None else item.right_prompt.source,
                item.training_filter_reason,
                "pending" if item.active_hands else "no_prompt_available",
            )
            for item in plans
        ],
    )
    connection.commit()


def labels_from_logits(object_ids, mask_logits: torch.Tensor) -> np.ndarray:
    logits = mask_logits[:, 0].detach().float().cpu().numpy()
    winner = np.argmax(logits, axis=0)
    confidence = np.max(logits, axis=0)
    labels = np.zeros(confidence.shape, np.uint8)
    for output_index, object_id in enumerate(int(value) for value in object_ids):
        labels[(winner == output_index) & (confidence > 0.0)] = object_id + 1
    return labels


def write_sam_frame(connection: sqlite3.Connection, frame_index: int,
                    labels: np.ndarray) -> None:
    area_left = int((labels == 1).sum())
    area_right = int((labels == 2).sum())
    candidate, gaussian_left, gaussian_right, filter_reason = connection.execute(
        "SELECT training_candidate,gaussian_left_valid,gaussian_right_valid,"
        "training_filter_reason FROM frames WHERE frame_index=?", (frame_index,)
    ).fetchone()
    candidate = bool(candidate)
    matched_hand = ((bool(gaussian_left) and area_left > 0)
                    or (bool(gaussian_right) and area_right > 0))
    eligible = candidate and matched_hand
    if candidate and not matched_hand:
        filter_reason = (
            "sam_empty" if area_left + area_right == 0
            else "no_matching_gaussian_sam_hand"
        )
    status = "sam_complete_training" if eligible else (
        "sam_empty_filtered" if area_left + area_right == 0 else "sam_complete_filtered"
    )
    connection.execute(
        """
        UPDATE frames SET labels_zlib=?,area_left=?,area_right=?,sam_complete=1,
        training_eligible=?,training_filter_reason=?,status=? WHERE frame_index=?
        """,
        (sqlite3.Binary(zlib.compress(labels.tobytes(), 3)), area_left, area_right,
         int(eligible), filter_reason, status, frame_index),
    )


def draw_prompt(path: Path, frame: np.ndarray, episode: policy.Episode) -> None:
    image = frame.copy()
    colors = {policy.LEFT: (70, 210, 70), policy.RIGHT: (40, 140, 255)}
    names = {policy.LEFT: "left", policy.RIGHT: "right"}
    for hand, prompt in episode.prompts.items():
        x1, y1, x2, y2 = np.rint(prompt.box).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[hand], 3)
        cv2.putText(image, f"{names[hand]} {prompt.source}",
                    (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, colors[hand], 2, cv2.LINE_AA)
    text = (f"episode={episode.episode_index} source={episode.start_frame} "
            f"length={episode.stop_frame_exclusive - episode.start_frame}")
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .72,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .72,
                (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write {path}")


def main() -> None:
    args = arguments()
    sequence = args.sequence.resolve()
    support = args.support_root.resolve() / sequence.name
    mapping = args.mapping_root.resolve() / f"{sequence.name}.csv"
    required = (
        sequence / "_SUCCESS.json", sequence / "rgb_214_1_pinhole.mp4",
        sequence / "frame_timestamps_214_1.csv",
        sequence / "frame_pinhole_calibration_214_1.jsonl",
        sequence / "timecode_devicetime_mapping.csv",
        sequence / "mano_hand_pose_trajectory.jsonl",
        sequence / "headset_trajectory.csv",
        sequence / "masks/mask_hand_pose_available.csv",
        sequence / "masks/mask_qa_pass.csv",
        sequence / "masks/mask_hand_visible.csv",
        sequence / "masks/mask_good_exposure.csv",
        support / "box2d_hands.csv",
        support / "mps/slam/online_calibration.jsonl", mapping, args.checkpoint,
        ROOT / "mano_v1_2/models/MANO_LEFT.pkl",
        ROOT / "mano_v1_2/models/MANO_RIGHT.pkl",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    timestamps = load_timestamps(sequence / "frame_timestamps_214_1.csv")
    calibrations = load_calibrations(sequence / "frame_pinhole_calibration_214_1.jsonl")
    total_frames = int(json.loads((sequence / "_SUCCESS.json").read_text())["frames"])
    if len(timestamps) != total_frames or len(calibrations) != total_frames:
        raise RuntimeError("Frame/timestamp/calibration count mismatch")

    output = args.output_root.resolve() / sequence.name
    final_db, partial_db = output / "masks.sqlite", output / "masks.sqlite.partial"
    success_path = output / "_SUCCESS.json"
    if success_path.exists() and final_db.exists() and not args.overwrite:
        print(json.dumps({"sequence": sequence.name, "status": "already_complete"}))
        return
    if args.overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    prompts_dir = output / "prompt_frames"
    prompts_dir.mkdir(exist_ok=True)

    mano_model = MANOHandModel(str(ROOT / "mano_v1_2/models"))
    hand_provider = MANOHandDataProvider(
        str(sequence / "mano_hand_pose_trajectory.jsonl"), mano_model
    )
    headset_provider = load_headset_pose_provider_from_csv(
        str(sequence / "headset_trajectory.csv")
    )
    planning_started = time.perf_counter()
    plans = policy.build_frame_plan(
        timestamps, calibrations, sequence / "masks", support / "box2d_hands.csv",
        support / "mps", sequence / "timecode_devicetime_mapping.csv", mapping,
        hand_provider, headset_provider, args.box_padding,
        args.min_box_side, args.min_box_area,
    )
    episodes = policy.build_episodes(plans)
    planning_seconds = time.perf_counter() - planning_started

    height, width = int(calibrations[0]["image_height"]), int(calibrations[0]["image_width"])
    zero_blob = zlib.compress(np.zeros((height, width), np.uint8).tobytes(), 3)
    connection = sqlite3.connect(partial_db)
    create_schema(connection)
    for key, value in {
        "format_version": 3, "sequence": sequence.name, "height": height,
        "width": width, "total_frames": total_frames,
        "labels": {"0": "background", "1": "left_hand", "2": "right_hand"},
        "prompt_policy": (
            "episode-start only; MANO bbox when mask_hand_pose_available, otherwise "
            "original FISHEYE624 bbox reprojected to LINEAR"
        ),
        "episode_policy": "split only when per-hand presence state changes",
        "filter_policy": (
            "post-inference: Gaussian mapping valid AND mask_qa_pass AND "
            "mask_hand_pose_available AND mask_hand_visible AND mask_good_exposure "
            "AND a nonempty SAM label for the corresponding Gaussian-valid hand; "
            "no area ceiling"
        ),
        "support_source": str(support), "gaussian_mapping": str(mapping),
        "planning_seconds": planning_seconds, "box_padding": args.box_padding,
    }.items():
        set_metadata(connection, key, value)
    initialize_database(connection, plans, episodes, zero_blob)
    connection.commit()

    pending = [int(row[0]) for row in connection.execute(
        "SELECT episode_index FROM episodes WHERE status='pending' ORDER BY episode_index"
    )]
    if args.max_episodes:
        pending = pending[:args.max_episodes]
    pending_set = set(pending)
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml", str(args.checkpoint),
        device="cuda", apply_postprocessing=True,
    )
    torch.cuda.reset_peak_memory_stats()
    capture = cv2.VideoCapture(str(sequence / "rgb_214_1_pinhole.mp4"))
    if not capture.isOpened():
        raise RuntimeError("Could not open pinhole video")
    run_started = time.perf_counter()
    try:
        for episode in episodes:
            if episode.episode_index not in pending_set:
                continue
            episode_started = time.perf_counter()
            temporary = Path(tempfile.mkdtemp(
                prefix=f"sam2v3_{sequence.name}_{episode.episode_index:04d}_", dir="/tmp"
            ))
            frames_dir = temporary / "frames"
            frames_dir.mkdir()
            first_frame = None
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, episode.start_frame)
                for local, source_frame in enumerate(
                    range(episode.start_frame, episode.stop_frame_exclusive)
                ):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Could not decode frame {source_frame}")
                    if local == 0:
                        first_frame = frame.copy()
                    frame_path = frames_dir / f"{local:06d}.jpg"
                    if not cv2.imwrite(str(frame_path), frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                        raise RuntimeError(f"Could not write {frame_path}")
                assert first_frame is not None
                draw_prompt(
                    prompts_dir / f"episode_{episode.episode_index:04d}_frame_"
                    f"{episode.start_frame:06d}.jpg", first_frame, episode,
                )
                connection.execute("BEGIN")
                written = set()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    state = predictor.init_state(
                        str(frames_dir), offload_video_to_cpu=True,
                        offload_state_to_cpu=False, async_loading_frames=True,
                    )
                    for hand, prompt in episode.prompts.items():
                        predictor.add_new_points_or_box(
                            state, frame_idx=0, obj_id=hand, box=prompt.box,
                        )
                    for local, object_ids, logits in predictor.propagate_in_video(
                        state, start_frame_idx=0,
                        max_frame_num_to_track=(
                            episode.stop_frame_exclusive - episode.start_frame
                        ), reverse=False,
                    ):
                        source_frame = episode.start_frame + int(local)
                        write_sam_frame(
                            connection, source_frame,
                            labels_from_logits(object_ids, logits),
                        )
                        written.add(source_frame)
                    del state
                expected = set(range(episode.start_frame, episode.stop_frame_exclusive))
                if written != expected:
                    raise RuntimeError(
                        f"Episode {episode.episode_index} missing "
                        f"{sorted(expected - written)[:20]}"
                    )
                elapsed = time.perf_counter() - episode_started
                connection.execute(
                    "UPDATE episodes SET status='complete',inference_seconds=?,"
                    "completed_at=CURRENT_TIMESTAMP WHERE episode_index=?",
                    (elapsed, episode.episode_index),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
            print(json.dumps({
                "sequence": sequence.name, "episode": episode.episode_index,
                "episodes_total": len(episodes), "start": episode.start_frame,
                "stop": episode.stop_frame_exclusive,
                "frames": episode.stop_frame_exclusive - episode.start_frame,
                "prompts": json.loads(policy.prompt_json(episode.prompts)),
                "seconds": elapsed,
            }), flush=True)
    finally:
        capture.release()

    connection.execute(
        "UPDATE frames SET training_filter_reason=\"no_sam_output\" "
        "WHERE training_candidate=1 AND sam_complete=0"
    )
    pending_count = int(connection.execute(
        "SELECT COUNT(*) FROM episodes WHERE status='pending'"
    ).fetchone()[0])
    training_frames = int(connection.execute(
        "SELECT COUNT(*) FROM frames WHERE training_eligible=1"
    ).fetchone()[0])
    sam_frames = int(connection.execute(
        "SELECT COUNT(*) FROM frames WHERE sam_complete=1"
    ).fetchone()[0])
    source_counts = dict(connection.execute(
        "SELECT COALESCE(left_prompt_source,'none')||'/'||"
        "COALESCE(right_prompt_source,'none'),COUNT(*) FROM frames GROUP BY 1"
    ))
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    if pending_count == 0:
        os.replace(partial_db, final_db)
        result = {
            "status": "complete", "sequence": sequence.name,
            "total_frames": total_frames, "sam_frames": sam_frames,
            "training_eligible_frames": training_frames,
            "episodes": len(episodes), "prompt_source_frame_counts": source_counts,
            "planning_seconds": planning_seconds,
            "inference_seconds": time.perf_counter() - run_started,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "database_bytes": final_db.stat().st_size,
        }
        success_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({
            "status": "partial", "sequence": sequence.name,
            "pending_episodes": pending_count, "sam_frames": sam_frames,
        }, indent=2))


if __name__ == "__main__":
    main()
