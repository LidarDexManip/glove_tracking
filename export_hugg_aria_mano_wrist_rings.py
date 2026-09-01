#!/usr/bin/env python3
"""Project MANO's open wrist boundary without rerendering hand images."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from render_hugg_aria_mano_aligned import (
    Handedness,
    MANOHandDataProvider,
    MANOHandModel,
    ROOT,
    load_headset_pose_provider_from_csv,
    load_training_eligible,
    policy,
    read_calibrations,
    read_timestamps,
    scaled_camera,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=ROOT / "outputs/sam2_hugg_aria_masks_v3_pilot",
    )
    parser.add_argument(
        "--pinhole-root", type=Path, default=ROOT / "data/HUGG_ARIA_PINHOLE"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/HUGG_ARIA_MANO_WRIST_RINGS",
    )
    parser.add_argument(
        "--mano-model-dir", type=Path, default=ROOT / "mano_v1_2/models"
    )
    parser.add_argument("--expected-sequences", type=int, default=136)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def boundary_vertex_indices(faces: np.ndarray) -> np.ndarray:
    edges = np.sort(
        np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
        ),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique_edges[counts == 1]).astype(np.int64)


def project_frame(
    timestamp: int,
    calibration: dict,
    size: int,
    model,
    hand_provider,
    headset_provider,
    ring_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.full((2, ring_indices.size, 2), np.nan, dtype=np.float32)
    palm_points = np.full((2, 2), np.nan, dtype=np.float32)
    valid = np.zeros(2, dtype=bool)
    collection = policy.pose_collection(hand_provider, timestamp)
    t_camera_world = policy.world_to_camera(
        calibration, headset_provider, timestamp
    )
    if collection is None or t_camera_world is None:
        return points, palm_points, valid
    camera = scaled_camera(calibration, t_camera_world, size)
    for hand_index, handedness in enumerate(
        (Handedness.Left, Handedness.Right)
    ):
        pose = collection.poses.get(handedness)
        if pose is None:
            continue
        vertices = hand_provider.get_hand_mesh_vertices(pose)
        if vertices is None:
            continue
        landmarks = hand_provider.get_hand_landmarks(pose)
        if landmarks is None:
            continue
        ring = vertices.detach().cpu().numpy()[ring_indices]
        hand_landmarks = landmarks.detach().cpu().numpy()
        palm = hand_landmarks[[8, 11, 14, 17]].mean(
            axis=0, keepdims=True
        )
        projected = camera.world_to_window3(ring).astype(np.float32)
        projected_palm = camera.world_to_window3(palm).astype(np.float32)
        if (
            not np.isfinite(projected).all()
            or not np.all(projected[:, 2] > 0)
            or not np.isfinite(projected_palm).all()
            or projected_palm[0, 2] <= 0
        ):
            continue
        points[hand_index] = projected[:, :2]
        palm_points[hand_index] = projected_palm[0, :2]
        valid[hand_index] = True
    return points, palm_points, valid


def process_sequence(task: tuple) -> dict:
    (
        sequence,
        eligible,
        pinhole_root,
        output_root,
        mano_model_dir,
        size,
        overwrite,
        torch_threads,
    ) = task
    import torch

    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    source = Path(pinhole_root) / sequence
    final = Path(output_root) / sequence
    success = final / "_SUCCESS.json"
    if success.is_file() and not overwrite:
        metadata = json.loads(success.read_text(encoding="utf-8"))
        if (
            metadata.get("format_version") == 2
            and metadata.get("output_size") == size
            and metadata.get("eligible_frames") == len(eligible)
            and (final / "wrist_ring_points.npz").is_file()
        ):
            return {"sequence": sequence, "status": "already_complete", **metadata}
        raise RuntimeError(f"Stale wrist-ring output: {success}")
    if final.exists() and not overwrite:
        raise FileExistsError(f"Incomplete wrist-ring output exists: {final}")

    timestamps = read_timestamps(source / "frame_timestamps_214_1.csv")
    calibrations = read_calibrations(
        source / "frame_pinhole_calibration_214_1.jsonl"
    )
    if len(timestamps) != len(calibrations):
        raise RuntimeError(f"Frame sidecar mismatch for {sequence}")
    total_frames = len(timestamps)
    model = MANOHandModel(str(mano_model_dir))
    hand_provider = MANOHandDataProvider(
        str(source / "mano_hand_pose_trajectory.jsonl"), model
    )
    headset_provider = load_headset_pose_provider_from_csv(
        str(source / "headset_trajectory.csv")
    )
    left_ring = boundary_vertex_indices(model.mano_layer_left.faces)
    right_ring = boundary_vertex_indices(model.mano_layer_right.faces)
    if not np.array_equal(left_ring, right_ring) or left_ring.size != 16:
        raise RuntimeError("Unexpected MANO wrist boundary topology")

    points = np.full(
        (total_frames, 2, left_ring.size, 2), np.nan, dtype=np.float32
    )
    palm_points = np.full((total_frames, 2, 2), np.nan, dtype=np.float32)
    valid = np.zeros((total_frames, 2), dtype=bool)
    started = time.perf_counter()
    for frame_index in sorted(eligible):
        frame_points, frame_palm_points, frame_valid = project_frame(
            timestamps[frame_index],
            calibrations[frame_index],
            size,
            model,
            hand_provider,
            headset_provider,
            left_ring,
        )
        points[frame_index] = frame_points
        palm_points[frame_index] = frame_palm_points
        valid[frame_index] = frame_valid

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{sequence}.partial.", dir=output_root)
    )
    try:
        np.savez_compressed(
            temporary / "wrist_ring_points.npz",
            points=points,
            palm_points=palm_points,
            valid=valid,
            ring_vertex_indices=left_ring,
        )
        metadata = {
            "format_version": 2,
            "sequence": sequence,
            "source": "projected_mano_wrist_ring_and_proximal_palm_anchor",
            "palm_anchor_landmark_indices": [8, 11, 14, 17],
            "ring_vertex_indices": left_ring.tolist(),
            "total_frames": total_frames,
            "eligible_frames": len(eligible),
            "valid_left_frames": int(valid[:, 0].sum()),
            "valid_right_frames": int(valid[:, 1].sum()),
            "output_size": size,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (temporary / "_SUCCESS.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        if final.exists():
            shutil.rmtree(final)
        temporary.replace(final)
        return {"sequence": sequence, "status": "projected", **metadata}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = arguments()
    grouped = load_training_eligible(
        args.mask_root.resolve(), args.expected_sequences
    )
    sequences = sorted(grouped)
    if args.sequence:
        requested = set(args.sequence)
        missing = sorted(requested - set(sequences))
        if missing:
            raise KeyError(f"Sequences are absent from SAM assets: {missing}")
        sequences = [sequence for sequence in sequences if sequence in requested]
    tasks = [
        (
            sequence,
            grouped[sequence],
            args.pinhole_root.resolve(),
            args.output_root.resolve(),
            args.mano_model_dir.resolve(),
            args.size,
            args.overwrite,
            args.torch_threads,
        )
        for sequence in sequences
    ]
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sequence, task): task[0] for task in tasks}
        for future in as_completed(futures):
            sequence = futures[future]
            try:
                print(json.dumps(future.result()), flush=True)
            except BaseException as error:
                failures.append((sequence, repr(error)))
                print(json.dumps({"sequence": sequence, "error": repr(error)}), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} wrist-ring exports failed: {failures[:5]}")


if __name__ == "__main__":
    main()
