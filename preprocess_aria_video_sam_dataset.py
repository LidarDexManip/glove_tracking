#!/usr/bin/env python3
"""Build the corrected-MANO Aria dataset from continuous SAM2 video tracking."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from aria_mano_dataset_utils import (
    DEFAULT_HF_REPO,
    DEFAULT_HF_REVISION,
    add_bytes,
    configure_imports,
    encode_mask,
    encode_rgb,
    load_corrected_mano_models,
    rasterize_meshes,
    read_jsonl,
    resized_camera,
    write_jsonl,
)
from pilot_aria_sam2_video_tracking import (
    component_count,
    configure_hot3d,
    mask_box_metrics,
    mask_metrics,
    nearest_hand_annotation,
    read_boxes,
    stage_extract,
    transformed_box,
)


ROOT = Path(__file__).resolve().parent
STREAM_ID = "214-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split-json", type=Path, default=(
        ROOT / "configs/hand_restoration/splits/train_aria_sequence_seed7.json"))
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/hot3d_aria")
    parser.add_argument("--work-root", type=Path, default=ROOT / "data/.aria_video_sam_work")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/derived/train_aria_mano_video_sam")
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=240)
    parser.add_argument("--prompt-stride", type=int, default=60)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=180)
    parser.add_argument("--sequence", action="append", default=[])
    return parser.parse_args()


def split_work(args: argparse.Namespace) -> tuple[dict, list[tuple[str, str]]]:
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    if split["hf_revision"] != DEFAULT_HF_REVISION:
        raise ValueError("Unexpected HOT3D revision")
    work = [(name, sequence) for name in ("train", "holdout")
            for sequence in split[name]]
    if args.sequence:
        wanted = set(args.sequence)
        work = [item for item in work if item[1] in wanted]
        missing = wanted - {item[1] for item in work}
        if missing:
            raise ValueError(f"Unknown sequences: {sorted(missing)}")
    work = [item for index, item in enumerate(work)
            if index % args.num_workers == args.worker_index]
    return split, work


def raw_sequence_complete(path: Path) -> bool:
    required = (
        "recording.vrs", "box2d_hands.csv", "box2d_objects.csv",
        "dynamic_objects.csv", "camera_models.json",
        "headset_trajectory.csv", "mano_hand_pose_trajectory.jsonl", "metadata.json",
        "mps/slam/online_calibration.jsonl", "mps/slam/summary.json",
        "masks/mask_hand_visible.csv", "masks/mask_qa_pass.csv",
    )
    return all((path / name).is_file() for name in required)


def download_sequence(args: argparse.Namespace, sequence: str, split: dict) -> Path:
    destination = args.raw_root / sequence
    if raw_sequence_complete(destination):
        return destination
    from huggingface_hub import snapshot_download

    args.raw_root.mkdir(parents=True, exist_ok=True)
    patterns = [
        f"{sequence}/recording.vrs", f"{sequence}/box2d_hands.csv",
        f"{sequence}/box2d_objects.csv", f"{sequence}/dynamic_objects.csv",
        f"{sequence}/camera_models.json", f"{sequence}/headset_trajectory.csv",
        f"{sequence}/mano_hand_pose_trajectory.jsonl", f"{sequence}/metadata.json",
        f"{sequence}/mps/slam/online_calibration.jsonl",
        f"{sequence}/mps/slam/summary.json", f"{sequence}/masks/*.csv",
    ]
    print(f"DOWNLOAD_START {sequence}", flush=True)
    snapshot_download(
        repo_id=split.get("hf_repo", DEFAULT_HF_REPO), repo_type="dataset",
        revision=split["hf_revision"], local_dir=args.raw_root,
        allow_patterns=patterns, max_workers=4,
    )
    if not raw_sequence_complete(destination):
        raise RuntimeError(f"Incomplete persistent download: {destination}")
    print(f"DOWNLOAD_DONE {sequence}", flush=True)
    return destination


def sequence_output_valid(args: argparse.Namespace, split_name: str, sequence: str) -> bool:
    root = args.output_dir / split_name
    tar_path = root / f"{sequence}.tar"
    samples = root / f"{sequence}.samples.jsonl"
    availability = root / f"{sequence}.availability.jsonl"
    if not all(path.is_file() for path in (tar_path, samples, availability)):
        return False
    try:
        keys = [row["key"] for row in read_jsonl(samples)]
        with tarfile.open(tar_path) as archive:
            names = set(archive.getnames())
        suffixes = ("target.jpg", "mano.png", "mask.png", "visible_mask.png",
                    "sam_mask.png", "json")
        return all(all(f"{key}.{suffix}" in names for suffix in suffixes) for key in keys)
    except (OSError, tarfile.TarError, json.JSONDecodeError):
        return False


def extract_video(args: argparse.Namespace, sequence: str) -> Path:
    root = args.work_root / sequence
    metadata = root / "chunks.json"
    if metadata.is_file():
        parsed = json.loads(metadata.read_text())
        existing = sum(len(list(Path(chunk["directory"]).glob("*.jpg")))
                       for chunk in parsed["chunks"])
        if existing == parsed["frame_count"]:
            return root
    extract_args = SimpleNamespace(
        sequence=sequence, raw_root=args.raw_root, output=args.work_root,
        size=args.size, chunk_size=args.chunk_size, prompt_stride=args.prompt_stride,
        prompt_window=12, prompt_min_visibility=0.2, max_frames=None,
    )
    stage_extract(extract_args)
    return root


def load_predictor(args: argparse.Namespace):
    sys.path.insert(0, str(ROOT / "third_party/sam2"))
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    return SAM2VideoPredictor.from_pretrained(args.model_id, device=args.device)


def candidate_indices(frame_count: int, stride: int, maximum: int) -> list[int]:
    return list(range(0, frame_count, stride))[:maximum]


def tracking_valid(root: Path, sequence: str, indices: list[int]) -> bool:
    records = root / "tracking_records.jsonl"
    masks = root / "candidate_masks"
    if not records.is_file():
        return False
    return all(all((masks / f"{sequence}_{index // 20:06d}.{name}.png").is_file()
                       for name in ("left", "right", "union")) for index in indices)


def track_video(args: argparse.Namespace, sequence: str, predictor) -> None:
    root = args.work_root / sequence
    metadata = json.loads((root / "chunks.json").read_text())
    candidates = set(candidate_indices(metadata["frame_count"], args.frame_stride,
                                       args.max_candidates))
    if tracking_valid(root, sequence, sorted(candidates)):
        return
    mask_root = root / "candidate_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    records = []
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for chunk in metadata["chunks"]:
        state = predictor.init_state(video_path=chunk["directory"], offload_video_to_cpu=True,
                                     offload_state_to_cpu=False, async_loading_frames=False)
        entries = [(hand, prompt) for hand in (0, 1)
                   for prompt in chunk["prompt_frames"][str(hand)]]
        masks_by_frame = {}
        if entries:
            propagation_seed = min(int(prompt["local_index"]) for _, prompt in entries)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for hand, prompt in entries:
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=int(prompt["local_index"]),
                        obj_id=hand + 1, box=np.asarray(prompt["box"], np.float32),
                    )
                for reverse in (False, True):
                    for local_index, object_ids, logits in predictor.propagate_in_video(
                        state, start_frame_idx=propagation_seed,
                        max_frame_num_to_track=chunk["frame_count"], reverse=reverse,
                    ):
                        hands = masks_by_frame.setdefault(int(local_index), {})
                        for object_index, object_id in enumerate(object_ids):
                            hands[int(object_id) - 1] = (
                                logits[object_index, 0] > 0).detach().cpu().numpy()
        for local_index in range(chunk["frame_count"]):
            global_index = int(chunk["global_start"]) + local_index
            hands = masks_by_frame.get(local_index, {})
            left = hands.get(0, np.zeros((args.size, args.size), bool))
            right = hands.get(1, np.zeros((args.size, args.size), bool))
            union = left | right
            records.append({
                "global_frame_index": global_index,
                "timestamp_ns": int(chunk["timestamps"][local_index]),
                "chunk_index": int(chunk["chunk_index"]),
                "left_pixels": int(left.sum()), "right_pixels": int(right.sum()),
                "union_pixels": int(union.sum()),
                "union_components": component_count(union, 16),
            })
            if global_index in candidates:
                key = f"{sequence}_{global_index // args.frame_stride:06d}"
                for name, mask in (("left", left), ("right", right), ("union", union)):
                    path = mask_root / f"{key}.{name}.png"
                    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
                        raise OSError(path)
        predictor.reset_state(state)
        del state, masks_by_frame
        gc.collect()
        torch.cuda.empty_cache()
        print(f"TRACK {sequence} chunk={chunk['chunk_index']}", flush=True)
    write_jsonl(root / "tracking_records.jsonl", records)


def hand_quality(provider, boxes, hand_timestamps, timestamp: int,
                 hand: int, mask: np.ndarray, size: int) -> dict:
    annotation_timestamp, annotation = nearest_hand_annotation(
        boxes, hand_timestamps[hand], hand, timestamp)
    quality = {
        "visibility": annotation["visibility"],
        "annotation_delta_ms": abs(annotation_timestamp - timestamp) / 1e6,
        "pixels": int(mask.sum()),
    }
    try:
        box = transformed_box(provider, annotation_timestamp, annotation["box"],
                              (1408, 1408), size)
        quality["reference_box"] = box
        quality.update(mask_box_metrics(mask, box))
    except RuntimeError:
        quality.update({"reference_box": None, "bbox_iou": 0.0,
                        "pixel_inside_box_fraction": 0.0})
    quality["required"] = (quality["visibility"] >= 0.2 and
                           quality["annotation_delta_ms"] <= 100.0)
    quality["passed"] = (not quality["required"] or
                         (quality["pixels"] >= 64 and quality["bbox_iou"] >= 0.25 and
                          quality["pixel_inside_box_fraction"] >= 0.6))
    return quality


def read_frame_mask(path: Path) -> dict[int, bool]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            int(row["timestamp[ns]"]): row["mask"].strip().lower() == "true"
            for row in csv.DictReader(stream)
            if row["stream_id"] == STREAM_ID
        }


def local_camera(provider, headset_provider, stream_id, timestamp: int, image):
    from aria_mano_dataset_utils import Camera
    from projectaria_tools.core.calibration import LINEAR
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

    pose = headset_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp, time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE, acceptable_time_delta=10_000_000,
    )
    if pose is None:
        return None
    device_camera, calibration = provider.get_online_camera_calibration(
        stream_id, timestamp_ns=timestamp, camera_model=LINEAR)
    world_camera = pose.pose3d.T_world_device @ device_camera
    matrix = world_camera.to_matrix()
    rotation = matrix[:3, :3].T
    translation = -rotation @ matrix[:3, 3]
    focal = calibration.get_focal_lengths()
    principal = calibration.get_principal_point()
    return Camera(
        width=int(image.shape[1]), height=int(image.shape[0]),
        fx=float(focal[0]), fy=float(focal[1]),
        cx=float(principal[0]), cy=float(principal[1]),
        R=rotation, t=translation,
    )


def build_sequence_shard(args: argparse.Namespace, split_name: str, sequence_name: str,
                         raw: Path, models) -> None:
    configure_imports()
    configure_hot3d()
    from data_loaders.AriaDataProvider import AriaDataProvider
    from data_loaders.HeadsetPose3dProvider import load_headset_pose_provider_from_csv
    from data_loaders.ManoHandDataProvider import MANOHandDataProvider
    from data_loaders.loader_hand_poses import Handedness
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from projectaria_tools.core.stream_id import StreamId

    output_root = args.output_dir / split_name
    output_root.mkdir(parents=True, exist_ok=True)
    final_tar = output_root / f"{sequence_name}.tar"
    incomplete = output_root / f".{sequence_name}.tar.inprogress"
    work = args.work_root / sequence_name
    mask_root = work / "candidate_masks"
    tracking = {row["global_frame_index"]: row
                for row in read_jsonl(work / "tracking_records.jsonl")}
    boxes = read_boxes(raw / "box2d_hands.csv")
    hand_timestamps = {hand: sorted(ts for ts, hands in boxes.items() if hand in hands)
                       for hand in (0, 1)}
    provider = AriaDataProvider(str(raw / "recording.vrs"), str(raw / "mps"))
    headset_provider = load_headset_pose_provider_from_csv(str(raw / "headset_trajectory.csv"))
    hand_provider = MANOHandDataProvider(str(raw / "mano_hand_pose_trajectory.jsonl"), models)
    stream_id = StreamId(STREAM_ID)
    hand_visible = read_frame_mask(raw / "masks/mask_hand_visible.csv")
    qa_pass = read_frame_mask(raw / "masks/mask_qa_pass.csv")
    candidate_count = min(len(tracking), args.max_candidates)
    samples, availability = [], []
    try:
        with tarfile.open(incomplete, "w") as archive:
            for frame_index in range(candidate_count):
                global_index = frame_index * args.frame_stride
                key = f"{sequence_name}_{frame_index:06d}"
                status = {"key": key, "source_sequence_id": sequence_name,
                          "source_frame_index": global_index}
                if global_index not in tracking:
                    availability.append({**status, "accepted": False,
                                         "reason": "missing_tracking_mask"})
                    continue
                timestamp = int(tracking[global_index]["timestamp_ns"])
                if not hand_visible.get(timestamp, False) or not qa_pass.get(timestamp, False):
                    availability.append({**status, "accepted": False,
                                         "reason": "not_visible_or_qa"})
                    continue
                meshes, hands, mano_valid = [], [], {"left": False, "right": False}
                hand_data = hand_provider.get_pose_at_timestamp(
                    timestamp_ns=timestamp, time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE, acceptable_time_delta=10_000_000,
                )
                if hand_data is not None:
                    pose_values = hand_data.pose3d_collection.poses.values()
                else:
                    pose_values = []
                for hand_pose in pose_values:
                    is_right = hand_pose.handedness == Handedness.Right
                    name = "right" if is_right else "left"
                    mano_valid[name] = True
                    color = np.asarray([214, 154, 118] if is_right else
                                       [184, 124, 96], np.float32)
                    vertices = hand_provider.get_hand_mesh_vertices(hand_pose)
                    faces = (models.mano_layer_right.faces if is_right
                             else models.mano_layer_left.faces)
                    meshes.append((vertices.detach().cpu().numpy().astype(np.float32),
                                   faces, color))
                    hands.append(name)
                if not meshes:
                    availability.append({**status, "accepted": False,
                                         "reason": "missing_mano"})
                    continue
                frame = provider.get_undistorted_image(timestamp, stream_id)
                camera = local_camera(provider, headset_provider, stream_id, timestamp, frame)
                if camera is None:
                    availability.append({**status, "accepted": False,
                                         "reason": "missing_headset"})
                    continue
                camera = resized_camera(camera, args.size)
                mano, full_mask = rasterize_meshes(meshes, camera)
                if int(full_mask.sum()) < 64:
                    availability.append({**status, "accepted": False,
                                         "reason": "mano_mask_too_small"})
                    continue
                hand_masks = {}
                missing_mask = False
                for hand, name in ((0, "left"), (1, "right")):
                    path = mask_root / f"{key}.{name}.png"
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        missing_mask = True
                        break
                    hand_masks[name] = image > 0
                if missing_mask or global_index not in tracking:
                    availability.append({**status, "accepted": False,
                                         "reason": "missing_tracking_mask"})
                    continue
                qualities = {
                    name: hand_quality(provider, boxes, hand_timestamps, timestamp,
                                       hand, hand_masks[name], args.size)
                    for hand, name in ((0, "left"), (1, "right"))
                }
                for name, quality in qualities.items():
                    quality["mano_valid"] = mano_valid[name]
                    if quality["required"] and not mano_valid[name]:
                        quality["passed"] = False
                raw_sam = hand_masks["left"] | hand_masks["right"]
                visible = raw_sam & full_mask
                topology = mask_metrics(visible)
                accepted = (
                    topology["pixels"] >= 64 and topology["components"] <= 5 and
                    topology["tiny_component_pixel_fraction"] <= 0.05 and
                    all(quality["passed"] for quality in qualities.values())
                )
                record = {
                    "schema_version": 4, "key": key, "clip_id": sequence_name,
                    "frame_id": f"{frame_index:06d}", "source_frame_index": global_index,
                    "participant_id": sequence_name.split("_", 1)[0],
                    "source_sequence_id": sequence_name, "camera_id": STREAM_ID,
                    "device": "Aria", "handedness": "+".join(hands),
                    "canonical_image_size": [args.size, args.size],
                    "image_geometry": "undistorted_linear",
                    "condition_source": "corrected_raw_mano_video_sam",
                    "left_shapedirs_fix": True, "mano_mask_pixels": int(full_mask.sum()),
                    "visible_mask_pixels": topology["pixels"],
                    "video_sam_mask_pixels": int(raw_sam.sum()),
                    "video_sam_model": args.model_id, "hand_quality": qualities,
                    "visible_topology": topology, "accepted": accepted,
                }
                availability.append(record)
                if not accepted:
                    continue
                target = cv2.resize(frame, (args.size, args.size),
                                    interpolation=cv2.INTER_AREA).astype(np.uint8)
                add_bytes(archive, f"{key}.target.jpg", encode_rgb(
                    target, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]))
                add_bytes(archive, f"{key}.mano.png", encode_rgb(mano, ".png"))
                add_bytes(archive, f"{key}.mask.png", encode_mask(full_mask.astype(np.uint8) * 255))
                add_bytes(archive, f"{key}.visible_mask.png",
                          encode_mask(visible.astype(np.uint8) * 255))
                add_bytes(archive, f"{key}.sam_mask.png",
                          encode_mask(raw_sam.astype(np.uint8) * 255))
                add_bytes(archive, f"{key}.json",
                          (json.dumps(record, separators=(",", ":")) + "\n").encode())
                samples.append(record)
        relative = str(final_tar.relative_to(args.output_dir))
        for record in samples:
            record["shard"] = relative
        incomplete.replace(final_tar)
        write_jsonl(output_root / f"{sequence_name}.samples.jsonl", samples)
        write_jsonl(output_root / f"{sequence_name}.availability.jsonl", availability)
        print(f"SHARD_DONE {sequence_name} accepted={len(samples)} candidates={len(availability)}",
              flush=True)
    finally:
        incomplete.unlink(missing_ok=True)
        del hand_provider, headset_provider, provider
        gc.collect()


def finalize(args: argparse.Namespace) -> None:
    split = json.loads(args.split_json.read_text())
    summary = {"schema_version": 4, "source": "HOT3D Aria + corrected MANO + SAM2 video",
               "video_tracking": True, "left_shapedirs_fix": True, "splits": {}}
    missing = []
    for split_name in ("train", "holdout"):
        samples, availability = [], []
        for sequence in split[split_name]:
            root = args.output_dir / split_name
            sample_path = root / f"{sequence}.samples.jsonl"
            availability_path = root / f"{sequence}.availability.jsonl"
            if not sample_path.is_file() or not availability_path.is_file():
                missing.append(sequence)
                continue
            samples.extend(read_jsonl(sample_path))
            availability.extend(read_jsonl(availability_path))
        write_jsonl(args.output_dir / f"{split_name}_manifest.jsonl", samples)
        write_jsonl(args.output_dir / f"{split_name}_availability.jsonl", availability)
        summary["splits"][split_name] = {
            "sequences": len(split[split_name]), "accepted": len(samples),
            "rejected": len(availability) - len(samples),
        }
    if missing:
        raise RuntimeError(f"Cannot finalize; missing {len(missing)} sequences: {missing}")
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.finalize:
        finalize(args)
        return
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index must be in [0, num-workers)")
    configure_imports()
    split, work = split_work(args)
    predictor = load_predictor(args)
    models = load_corrected_mano_models()
    print(f"WORKER_START index={args.worker_index} sequences={len(work)} device={args.device}",
          flush=True)
    for ordinal, (split_name, sequence) in enumerate(work, 1):
        if sequence_output_valid(args, split_name, sequence):
            print(f"SKIP_COMPLETE {sequence}", flush=True)
            continue
        print(f"SEQUENCE_START {ordinal}/{len(work)} {split_name} {sequence}", flush=True)
        raw = download_sequence(args, sequence, split)
        extract_video(args, sequence)
        track_video(args, sequence, predictor)
        build_sequence_shard(args, split_name, sequence, raw, models)
    print(f"WORKER_DONE index={args.worker_index}", flush=True)


if __name__ == "__main__":
    main()
