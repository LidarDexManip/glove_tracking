#!/usr/bin/env python3
"""Audit one HOT3D subject/hand and emit a geometry-complete frame manifest.

This pass does not decode RGB video. It combines the official HOT3D MANO pose and
headset calibration with the existing visible-hand SAM labels, then rejects
truncated, occluded, tiny, and temporally redundant frames.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import sqlite3
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

for _name, _value in {
    "bool": bool, "int": int, "float": float, "complex": complex,
    "object": object, "unicode": str, "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hot3d/hot3d"))

from data_loaders.HandDataProviderBase import Handedness  # noqa: E402
from data_loaders.HeadsetPose3dProvider import load_headset_pose_provider_from_csv  # noqa: E402
from data_loaders.ManoHandDataProvider import MANOHandDataProvider  # noqa: E402
from data_loaders.mano_layer import MANOHandModel  # noqa: E402
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # noqa: E402

from hugg_aria_sam_policy import RawBoxProjector  # noqa: E402

from gaussian_hand.identity import SubjectBinding

from gaussian_hand.data.geometry import (  # noqa: E402
    crop_intrinsics, fraction_inside_box, project_points, rasterize_silhouette,
    square_crop, visible_mano_ratio,
)

POSE_TOLERANCE_NS = 10_000_000


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path,
                        default=ROOT / "outputs/sam2_hugg_aria_masks_v3_pilot")
    parser.add_argument("--mano-model-dir", type=Path,
                        default=ROOT / "mano_v1_2/models")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def nearest_pose(records: list[dict], timestamps: list[int], timestamp: int) -> dict | None:
    index = bisect.bisect_left(timestamps, timestamp)
    choices = [i for i in (index - 1, index) if 0 <= i < len(timestamps)]
    if not choices:
        return None
    best = min(choices, key=lambda i: abs(timestamps[i] - timestamp))
    return records[best] if abs(timestamps[best] - timestamp) <= POSE_TOLERANCE_NS else None


def get_collection(provider, timestamp: int):
    result = provider.get_pose_at_timestamp(
        timestamp_ns=timestamp, time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE, acceptable_time_delta=POSE_TOLERANCE_NS,
    )
    return None if result is None else result.pose3d_collection


def world_to_camera(calibration: dict, provider, timestamp: int) -> np.ndarray | None:
    result = provider.get_pose_at_timestamp(
        timestamp_ns=timestamp, time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE, acceptable_time_delta=POSE_TOLERANCE_NS,
    )
    if result is None:
        return None
    world_device = np.asarray(result.pose3d.T_world_device.to_matrix(), np.float64)
    device_camera = np.asarray(calibration["T_device_camera"], np.float64)
    return np.linalg.inv(world_device @ device_camera)


def intrinsic_matrix(calibration: dict) -> np.ndarray:
    fx, fy = calibration["focal_lengths"]
    cx, cy = calibration["principal_point"]
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float64)


def load_object_boxes(path: Path) -> dict[int, list[np.ndarray]]:
    result: dict[int, list[np.ndarray]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["stream_id"] != "214-1" or not row["x_min[pixel]"].strip():
                continue
            timestamp = int(row["timestamp[ns]"])
            result.setdefault(timestamp, []).append(np.asarray([
                float(row["x_min[pixel]"]), float(row["y_min[pixel]"]),
                float(row["x_max[pixel]"]), float(row["y_max[pixel]"]),
            ], np.float64))
    return result


def box_overlap_fraction(hand_box: np.ndarray, object_box: np.ndarray) -> float:
    lo = np.maximum(hand_box[:2], object_box[:2])
    hi = np.minimum(hand_box[2:], object_box[2:])
    intersection = float(np.maximum(hi - lo, 0.0).prod())
    hand_area = float(np.maximum(hand_box[2:] - hand_box[:2], 0.0).prod())
    return intersection / max(hand_area, 1e-8)


def audit_sequence(sequence: str, source: Path, raw_source: Path,
                   database: Path, model, policy: dict, crop_policy: dict,
                   sampling: dict,
                   max_frames: int, progress_every: int, handedness: str = "right") -> tuple[list[dict], dict]:
    meta = json.loads((source / "source_metadata.json").read_text())
    if meta.get("have_hand_object_pose_gt") is not True:
        raise ValueError(f"No GT pose: {sequence}")
    hand_enum = Handedness.Right if handedness == "right" else Handedness.Left
    hand_key = "1" if handedness == "right" else "0"
    sam_label = 2 if handedness == "right" else 1
    required = [
        source / "frame_pinhole_calibration_214_1.jsonl",
        source / "mano_hand_pose_trajectory.jsonl",
        source / "headset_trajectory.csv",
        database,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    calibrations = read_jsonl(required[0])
    pose_records = read_jsonl(required[1])
    pose_timestamps = [int(item["timestamp_ns"]) for item in pose_records]
    hand_provider = MANOHandDataProvider(str(required[1]), model)
    headset_provider = load_headset_pose_provider_from_csv(str(required[2]))
    faces = np.asarray(getattr(model, "mano_layer_" + handedness).faces, np.int64)
    downsample = int(sampling["audit_mask_downsample"])
    gap = int(sampling["minimum_frame_gap"])
    object_boxes = {}
    projector = None
    if policy.get("max_raw_object_bbox_overlap_proxy") is not None:
        object_boxes = load_object_boxes(raw_source / "box2d_objects.csv")
        projector = RawBoxProjector(
            raw_source / "mps", source / "timecode_devicetime_mapping.csv",
            int(calibrations[0]["image_width"]), int(calibrations[0]["image_height"]))

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        metadata = {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key,value_json FROM metadata")
        }
        rows = connection.execute(
            f"SELECT frame_index,timestamp_ns,labels_zlib,area_{handedness} "
            f"FROM frames WHERE training_eligible=1 AND area_{handedness}>0 ORDER BY frame_index"
        ).fetchall()
    finally:
        connection.close()
    if max_frames:
        rows = rows[:max_frames]

    width, height = int(metadata["width"]), int(metadata["height"])
    counts: Counter[str] = Counter()
    accepted: list[dict] = []
    metrics: list[tuple[float, float, float]] = []
    last_accepted = -gap
    started = time.perf_counter()

    for ordinal, (frame_index, timestamp, blob, sam_area) in enumerate(rows, 1):
        reason = "accepted"
        collection = get_collection(hand_provider, int(timestamp))
        pose = None if collection is None else collection.poses.get(hand_enum)
        t_camera_world = world_to_camera(calibrations[frame_index], headset_provider,
                                         int(timestamp))
        if pose is None or t_camera_world is None:
            reason = "missing_pose_or_camera"
        else:
            vertices = hand_provider.get_hand_mesh_vertices(pose)
            vertices = None if vertices is None else vertices.detach().cpu().numpy()
            if vertices is None:
                reason = "missing_vertices"

        if reason != "accepted":
            counts[reason] += 1
            continue

        k = intrinsic_matrix(calibrations[frame_index])
        uv, z = project_points(vertices, t_camera_world, k)
        in_image = ((z > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < width) &
                    (uv[:, 1] >= 0) & (uv[:, 1] < height))
        inside_image = float(in_image.mean())
        mano_mask = rasterize_silhouette(uv, z, faces, width, height, downsample)
        labels = np.frombuffer(zlib.decompress(blob), np.uint8).reshape(height, width)
        sam_right = cv2.resize((labels == sam_label).astype(np.uint8),
                               (mano_mask.shape[1], mano_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        completeness = visible_mano_ratio(sam_right, mano_mask)
        area_fraction = float(mano_mask.mean())
        valid_uv = uv[z > 0.01]
        box = square_crop(valid_uv, width, height,
                          float(crop_policy["scale_around_projected_mano"]))
        inside_crop = 0.0 if box is None else fraction_inside_box(valid_uv, box)
        metrics.append((completeness, inside_image, area_fraction))

        object_overlap = 0.0
        if (inside_image >= float(policy["min_mano_inside_image"]) and
                completeness >= float(policy["min_visible_mano_silhouette_ratio"]) and
                area_fraction >= float(policy["min_projected_hand_area_fraction"]) and
                box is not None and
                inside_crop >= float(policy["min_mano_inside_crop"])):
            hand_box = np.r_[valid_uv.min(axis=0), valid_uv.max(axis=0)]
            for raw_box in object_boxes.get(int(timestamp), []):
                projected = projector.project(
                    raw_box, int(timestamp), calibrations[frame_index],
                    padding=0.0, min_side=1.0, min_area=1.0)
                if projected is not None:
                    object_overlap = max(
                        object_overlap, box_overlap_fraction(hand_box, projected))

        if inside_image < float(policy["min_mano_inside_image"]):
            reason = "mano_outside_image"
        elif completeness < float(policy["min_visible_mano_silhouette_ratio"]):
            reason = "visible_completeness"
        elif area_fraction < float(policy["min_projected_hand_area_fraction"]):
            reason = "hand_too_small"
        elif box is None or inside_crop < float(policy["min_mano_inside_crop"]):
            reason = "mano_outside_crop"
        elif object_overlap > float(policy.get("max_raw_object_bbox_overlap_proxy", 1.0)):
            reason = "object_bbox_overlap"
        elif frame_index - last_accepted < gap:
            reason = "temporal_gap"

        counts[reason] += 1
        if reason != "accepted":
            continue
        raw_pose = nearest_pose(pose_records, pose_timestamps, int(timestamp))
        hand = None if raw_pose is None else raw_pose["hand_poses"].get(hand_key)
        if hand is None:
            counts["accepted"] -= 1
            counts["missing_raw_hand_pose"] += 1
            continue
        crop_k = crop_intrinsics(k, box, int(crop_policy["output_size"]))
        accepted.append({
            "schema_version": 1,
            "sequence": sequence,
            "frame_index": int(frame_index),
            "timestamp_ns": int(timestamp),
            "handedness": handedness,
            "mano_pca": [float(v) for v in hand["pose"]],
            "mano_betas": [float(v) for v in hand["betas"]],
            "wrist_q_wxyz": [float(v) for v in hand["wrist_xform"]["q_wxyz"]],
            "wrist_t_world": [float(v) for v in hand["wrist_xform"]["t_xyz"]],
            "world_to_camera": t_camera_world.tolist(),
            "source_intrinsics": k.tolist(),
            "crop_xyxy": list(box),
            "crop_intrinsics": crop_k.tolist(),
            "output_size": int(crop_policy["output_size"]),
            "visible_mano_ratio": completeness,
            "mano_inside_image": inside_image,
            "mano_inside_crop": inside_crop,
            "projected_hand_area_fraction": area_fraction,
            "sam_hand_area": int(sam_area),
            "raw_object_bbox_overlap_proxy": object_overlap,
        })
        last_accepted = int(frame_index)
        if progress_every and ordinal % progress_every == 0:
            print(json.dumps({"sequence": sequence, "audited": ordinal,
                              "accepted": len(accepted)}), flush=True)

    values = np.asarray(metrics, np.float64)
    percentiles = {}
    if len(values):
        for column, name in enumerate(("visible_mano_ratio", "mano_inside_image",
                                       "projected_hand_area_fraction")):
            percentiles[name] = {
                str(q): float(v) for q, v in zip(
                    (0, 10, 25, 50, 75, 90, 100),
                    np.percentile(values[:, column], (0, 10, 25, 50, 75, 90, 100)))
            }
    report = {
        "sequence": sequence,
        "input_candidates": len(rows),
        "accepted_frames": len(accepted),
        "acceptance_rate": len(accepted) / max(len(rows), 1),
        "reason_counts": dict(counts),
        "metric_percentiles": percentiles,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return accepted, report


def main() -> None:
    args = arguments()
    config_path = args.config.resolve()
    root = ROOT
    config = json.loads(config_path.read_text())
    binding = config["binding"]
    SubjectBinding.from_dict(config)
    sequences = args.sequence or config["sequences"]
    if any(not name.startswith(binding["subject_id"] + "_") for name in sequences):
        raise ValueError("Sequence/subject binding mismatch")

    pinhole_root = resolve(root, config["pinhole_root"])
    raw_root = resolve(root, config["raw_root"])
    output = args.output or (root / "outputs/gaussian_hand" /
                             f'{binding["subject_id"]}_{binding["handedness"]}' / "audit")
    output.mkdir(parents=True, exist_ok=True)
    model = MANOHandModel(str(args.mano_model_dir))
    all_records, reports = [], []
    for sequence in sequences:
        records, report = audit_sequence(
            sequence, pinhole_root / sequence, raw_root / sequence,
            args.mask_root / sequence / "masks.sqlite", model,
            config["selection_policy"], config["crop"], config["sampling"],
            args.max_frames_per_sequence, args.progress_every, binding["handedness"],
        )
        all_records.extend(records)
        reports.append(report)
        print(json.dumps(report), flush=True)

    manifest = output / "frames.jsonl"
    with manifest.open("w") as handle:
        for record in all_records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary = {
        "schema_version": 1,
        "binding": binding,
        "config": str(config_path),
        "sequences": sequences,
        "accepted_frames": len(all_records),
        "manifest": str(manifest),
        "manifest_sha256": digest,
        "reports": reports,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "binding", "accepted_frames", "manifest", "manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
