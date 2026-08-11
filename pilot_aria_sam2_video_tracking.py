#!/usr/bin/env python3
"""Pilot SAM2 video tracking on one continuous HOT3D Aria sequence."""
from __future__ import annotations

import argparse
import bisect
import csv
import gc
import json
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
STREAM_ID = "214-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("extract", "track", "compare"), required=True)
    parser.add_argument("--sequence", default="P0001_f6cc0cc8")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/hot3d_aria")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/derived/train_aria_mano_sam")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/aria_sam2_video_pilot")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=240)
    parser.add_argument("--prompt-stride", type=int, default=60,
                        help="Add a fresh bbox prompt about this often (frames); 0 uses one seed")
    parser.add_argument("--prompt-window", type=int, default=12,
                        help="Search this many frames around each prompt anchor")
    parser.add_argument("--prompt-min-visibility", type=float, default=0.2)
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def configure_hot3d() -> None:
    sys.path.insert(0, str(ROOT / "hot3d/hot3d"))
    active_prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix)).resolve()
    envs_dir = active_prefix.parent if active_prefix.parent.name == "envs" else active_prefix / "envs"
    for aria_site in (envs_dir / "glove2hand/lib").glob("python*/site-packages"):
        if aria_site.is_dir() and str(aria_site) not in sys.path:
            sys.path.append(str(aria_site))


def read_boxes(path: Path) -> dict[int, dict[int, dict]]:
    result: dict[int, dict[int, dict]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["stream_id"] != STREAM_ID or not row["x_min[pixel]"]:
                continue
            timestamp = int(row["timestamp[ns]"])
            hand = int(row["hand_index"])
            result[timestamp][hand] = {
                "box": [float(row["x_min[pixel]"]), float(row["y_min[pixel]"]),
                        float(row["x_max[pixel]"]), float(row["y_max[pixel]"])],
                "visibility": float(row["visibility_ratio[%]"]),
            }
    return result


def transformed_box(provider, timestamp: int, raw_box: list[float], raw_shape: tuple[int, int],
                    output_size: int) -> list[float]:
    from projectaria_tools.core.calibration import FISHEYE624, LINEAR, distort_by_calibration
    from projectaria_tools.core.stream_id import StreamId

    stream = StreamId(STREAM_ID)
    _, native = provider.get_online_camera_calibration(stream, timestamp, camera_model=FISHEYE624)
    _, pinhole = provider.get_online_camera_calibration(stream, timestamp, camera_model=LINEAR)
    height, width = raw_shape
    rectangle = np.zeros((height, width), np.uint8)
    x0, y0, x1, y1 = raw_box
    cv2.rectangle(rectangle, (max(0, round(x0)), max(0, round(y0))),
                  (min(width - 1, round(x1)), min(height - 1, round(y1))), 255, -1)
    undistorted = distort_by_calibration(rectangle, pinhole, native)
    ys, xs = np.nonzero(undistorted > 16)
    if not len(xs):
        raise RuntimeError("Hand box vanished during undistortion")
    scale_x, scale_y = output_size / undistorted.shape[1], output_size / undistorted.shape[0]
    box = np.asarray([xs.min() * scale_x, ys.min() * scale_y,
                      xs.max() * scale_x, ys.max() * scale_y], np.float32)
    pad = max(4.0, 0.04 * max(box[2] - box[0], box[3] - box[1]))
    box += np.asarray([-pad, -pad, pad, pad], np.float32)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, output_size - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, output_size - 1)
    return [float(value) for value in box]


def select_seed(timestamps: list[int], boxes: dict[int, dict[int, dict]]) -> int:
    best = None
    for local_index, timestamp in enumerate(timestamps):
        hands = boxes.get(timestamp, {})
        if 0 not in hands or 1 not in hands:
            continue
        left, right = hands[0]["visibility"], hands[1]["visibility"]
        score = min(left, right) + 0.15 * (left + right)
        center_penalty = abs(local_index - (len(timestamps) - 1) / 2) / max(len(timestamps), 1)
        candidate = (score - 0.02 * center_penalty, local_index)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("No frame in chunk has boxes for both hands")
    return int(best[1])


def select_prompt_frames(timestamps: list[int], boxes: dict[int, dict[int, dict]], hand: int,
                         stride: int, window: int, minimum_visibility: float) -> list[int]:
    """Pick well-observed bbox corrections near regularly spaced video frames."""
    if stride <= 0:
        return []
    anchors = list(range(0, len(timestamps), stride))
    if anchors[-1] != len(timestamps) - 1:
        anchors.append(len(timestamps) - 1)
    selected = set()
    for anchor in anchors:
        start, stop = max(0, anchor - window), min(len(timestamps), anchor + window + 1)
        candidates = []
        for local_index in range(start, stop):
            annotation = boxes.get(timestamps[local_index], {}).get(hand)
            if annotation is None or annotation["visibility"] < minimum_visibility:
                continue
            candidates.append((annotation["visibility"] - 0.002 * abs(local_index - anchor),
                               local_index))
        if candidates:
            selected.add(max(candidates)[1])
    if not selected:
        fallback = [(hands[hand]["visibility"], local_index)
                    for local_index, timestamp in enumerate(timestamps)
                    if hand in (hands := boxes.get(timestamp, {}))]
        if not fallback:
            return []
        selected.add(max(fallback)[1])
    return sorted(selected)


def stage_extract(args: argparse.Namespace) -> None:
    configure_hot3d()
    from data_loaders.AriaDataProvider import AriaDataProvider
    from projectaria_tools.core.sensor_data import TimeDomain
    from projectaria_tools.core.stream_id import StreamId

    sequence_root = args.raw_root / args.sequence
    provider = AriaDataProvider(str(sequence_root / "recording.vrs"), str(sequence_root / "mps"))
    stream = StreamId(STREAM_ID)
    timestamps = [value for value in provider.get_sequence_timestamps(stream, TimeDomain.TIME_CODE)
                  if value > 0]
    if args.max_frames:
        timestamps = timestamps[:args.max_frames]
    boxes = read_boxes(sequence_root / "box2d_hands.csv")
    work = args.output / args.sequence / "work"
    work.mkdir(parents=True, exist_ok=True)
    chunks = []
    for chunk_index, start in enumerate(range(0, len(timestamps), args.chunk_size)):
        chunk_timestamps = timestamps[start:start + args.chunk_size]
        chunk_dir = work / f"chunk_{chunk_index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        raw_seed = provider.get_image(chunk_timestamps[0], stream)
        prompt_frames = {}
        for hand in (0, 1):
            local_indices = select_prompt_frames(
                chunk_timestamps, boxes, hand, args.prompt_stride, args.prompt_window,
                args.prompt_min_visibility,
            )
            prompt_frames[str(hand)] = []
            for local_index in local_indices:
                try:
                    box = transformed_box(
                        provider, chunk_timestamps[local_index],
                        boxes[chunk_timestamps[local_index]][hand]["box"],
                        raw_seed.shape[:2], args.size,
                    )
                except RuntimeError:
                    continue
                prompt_frames[str(hand)].append({
                    "local_index": local_index,
                    "timestamp_ns": chunk_timestamps[local_index],
                    "visibility": boxes[chunk_timestamps[local_index]][hand]["visibility"],
                    "box": box,
                })
        all_prompts = [prompt for values in prompt_frames.values() for prompt in values]
        seed_local = min((int(prompt["local_index"]) for prompt in all_prompts), default=0)
        seed_timestamp = chunk_timestamps[seed_local]
        seed_boxes = {hand: values[0]["box"] for hand, values in prompt_frames.items() if values}
        for local_index, timestamp in enumerate(chunk_timestamps):
            output = chunk_dir / f"{local_index:05d}.jpg"
            if output.is_file():
                continue
            image = provider.get_undistorted_image(timestamp, stream)
            image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise OSError(output)
        chunks.append({
            "chunk_index": chunk_index,
            "directory": str(chunk_dir.resolve()),
            "global_start": start,
            "frame_count": len(chunk_timestamps),
            "timestamps": chunk_timestamps,
            "seed_local": seed_local,
            "seed_global": start + seed_local,
            "seed_timestamp": seed_timestamp,
            "seed_visibility": {hand: values[0]["visibility"]
                                for hand, values in prompt_frames.items() if values},
            "seed_boxes": seed_boxes,
            "prompt_frames": prompt_frames,
        })
        print(f"extract chunk={chunk_index} frames={len(chunk_timestamps)} seed={start + seed_local}", flush=True)
    metadata = {"sequence": args.sequence, "size": args.size, "frame_count": len(timestamps),
                "chunk_size": args.chunk_size, "chunks": chunks}
    (args.output / args.sequence / "chunks.json").write_text(json.dumps(metadata, indent=2) + "\n")


def component_count(mask: np.ndarray, minimum_area: int = 1) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return int(np.count_nonzero(stats[1:count, cv2.CC_STAT_AREA] >= minimum_area))


def candidate_rows(dataset: Path, sequence: str) -> dict[int, dict]:
    result = {}
    for split in ("train", "holdout"):
        manifest = dataset / f"{split}_manifest.jsonl"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["source_sequence_id"] == sequence:
                row["split"] = split
                result[int(row["source_frame_index"])] = row
    return result


def stage_track(args: argparse.Namespace) -> None:
    import torch

    sys.path.insert(0, str(ROOT / "third_party/sam2"))
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    root = args.output / args.sequence
    metadata = json.loads((root / "chunks.json").read_text())
    candidates = candidate_rows(args.dataset, args.sequence)
    mask_root = root / "candidate_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    predictor = SAM2VideoPredictor.from_pretrained(args.model_id, device=args.device)
    records = []
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for chunk in metadata["chunks"]:
        state = predictor.init_state(video_path=chunk["directory"], offload_video_to_cpu=True,
                                     offload_state_to_cpu=False, async_loading_frames=False)
        prompts = chunk.get("prompt_frames")
        if prompts and any(prompts.values()):
            prompt_entries = [(hand, prompt) for hand in (0, 1)
                              for prompt in prompts[str(hand)]]
        else:
            seed = int(chunk["seed_local"])
            prompt_entries = [(hand, {"local_index": seed,
                                      "box": chunk["seed_boxes"][str(hand)]})
                              for hand in (0, 1)]
        propagation_seed = min(int(prompt["local_index"]) for _, prompt in prompt_entries)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for hand, prompt in prompt_entries:
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=int(prompt["local_index"]), obj_id=hand + 1,
                    box=np.asarray(prompt["box"], np.float32),
                )
            masks_by_frame: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
            for reverse in (False, True):
                for frame_index, object_ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=propagation_seed,
                    max_frame_num_to_track=chunk["frame_count"],
                    reverse=reverse,
                ):
                    for object_index, object_id in enumerate(object_ids):
                        mask = (logits[object_index, 0] > 0).detach().cpu().numpy()
                        masks_by_frame[int(frame_index)][int(object_id) - 1] = mask
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
                "seed_global": int(chunk["seed_global"]),
                "prompt_count": len(prompt_entries),
                "left_pixels": int(left.sum()),
                "right_pixels": int(right.sum()),
                "union_pixels": int(union.sum()),
                "union_components": component_count(union, 16),
            })
            if global_index in candidates:
                key = candidates[global_index]["key"]
                for name, mask in (("left", left), ("right", right), ("union", union)):
                    output = mask_root / f"{key}.{name}.png"
                    if not cv2.imwrite(str(output), mask.astype(np.uint8) * 255):
                        raise OSError(output)
        predictor.reset_state(state)
        del state, masks_by_frame
        gc.collect()
        torch.cuda.empty_cache()
        print(f"track chunk={chunk['chunk_index']} frames={chunk['frame_count']}", flush=True)
    (root / "tracking_records.jsonl").write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    )


def decode(archive: tarfile.TarFile, name: str, flag: int) -> np.ndarray:
    stream = archive.extractfile(name)
    if stream is None:
        raise KeyError(name)
    image = cv2.imdecode(np.frombuffer(stream.read(), np.uint8), flag)
    if image is None:
        raise RuntimeError(name)
    return image


def labeled(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(result, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return result


def mask_metrics(mask: np.ndarray) -> dict:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = sorted((int(area) for area in stats[1:count, cv2.CC_STAT_AREA]), reverse=True)
    pixels = int(mask.sum())
    return {
        "pixels": pixels,
        "components": len(areas),
        "tiny_component_pixel_fraction": (sum(area for area in areas if area < 64) / pixels
                                           if pixels else 0.0),
    }


def nearest_hand_annotation(boxes: dict[int, dict[int, dict]], hand_timestamps: list[int],
                            hand: int, timestamp: int) -> tuple[int, dict]:
    insertion = bisect.bisect_left(hand_timestamps, timestamp)
    candidates = hand_timestamps[max(0, insertion - 1):min(len(hand_timestamps), insertion + 1)]
    nearest = min(candidates, key=lambda value: abs(value - timestamp))
    return nearest, boxes[nearest][hand]


def mask_box_metrics(mask: np.ndarray, box: list[float]) -> dict:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"bbox_iou": 0.0, "pixel_inside_box_fraction": 0.0}
    mask_box = np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
    reference = np.asarray(box, np.float32)
    overlap_size = np.maximum(0, np.minimum(mask_box[2:], reference[2:]) -
                              np.maximum(mask_box[:2], reference[:2]))
    intersection = float(np.prod(overlap_size))
    union = (float(np.prod(mask_box[2:] - mask_box[:2])) +
             float(np.prod(reference[2:] - reference[:2])) - intersection)
    x0, y0, x1, y1 = np.rint(reference).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(mask.shape[1], x1), min(mask.shape[0], y1)
    inside = int(mask[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0
    return {
        "bbox_iou": intersection / max(union, 1.0),
        "pixel_inside_box_fraction": inside / max(int(mask.sum()), 1),
    }


def make_comparison(target: np.ndarray, mano: np.ndarray, full: np.ndarray,
                    old: np.ndarray, tracked: np.ndarray, new: np.ndarray,
                    old_count: int, new_count: int) -> np.ndarray:
    full_overlay, old_overlay, new_overlay = target.copy(), target.copy(), target.copy()
    full_overlay[full] = mano[full]
    old_overlay[old] = mano[old]
    new_overlay[new] = mano[new]
    tracked_overlay = target.copy()
    green = np.zeros_like(target); green[:] = (45, 210, 70)
    tracked_overlay[tracked] = (0.55 * target[tracked] + 0.45 * green[tracked]).astype(np.uint8)
    audit = np.zeros_like(target)
    audit[old & ~new] = (45, 45, 230)
    audit[new & ~old] = (45, 210, 70)
    audit[old & new] = (40, 210, 210)
    return np.concatenate([
        labeled(target, "TARGET"), labeled(full_overlay, "FULL MANO"),
        labeled(old_overlay, f"STATIC SAM C={old_count}"),
        labeled(tracked_overlay, "VIDEO SAM RAW"),
        labeled(new_overlay, f"VIDEO SAM intersect MANO C={new_count}"),
        labeled(audit, "YELLOW=BOTH GREEN=NEW RED=OLD"),
    ], axis=1)


def save_page(cards: list[np.ndarray], output: Path) -> None:
    thumbs = [cv2.resize(card, (1536, 256), interpolation=cv2.INTER_AREA) for card in cards]
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) % 2:
        thumbs.append(blank)
    page = np.concatenate([np.concatenate(thumbs[index:index + 2], axis=1)
                           for index in range(0, len(thumbs), 2)], axis=0)
    if not cv2.imwrite(str(output), page, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(output)


def stage_compare(args: argparse.Namespace) -> None:
    configure_hot3d()
    from data_loaders.AriaDataProvider import AriaDataProvider

    root = args.output / args.sequence
    sequence_root = args.raw_root / args.sequence
    provider = AriaDataProvider(str(sequence_root / "recording.vrs"), str(sequence_root / "mps"))
    boxes = read_boxes(sequence_root / "box2d_hands.csv")
    hand_timestamps = {hand: sorted(timestamp for timestamp, hands in boxes.items()
                                    if hand in hands) for hand in (0, 1)}
    tracking = {record["global_frame_index"]: record for record in (
        json.loads(line) for line in (root / "tracking_records.jsonl").read_text().splitlines()
    )}
    mask_root = root / "candidate_masks"
    visual_root = root / "comparisons"
    visual_root.mkdir(parents=True, exist_ok=True)
    rows = sorted(candidate_rows(args.dataset, args.sequence).values(),
                  key=lambda row: int(row["source_frame_index"]))
    if not rows:
        raise RuntimeError("Sequence is absent from finalized dataset")
    archive_path = args.dataset / rows[0]["shard"]
    records, cards = [], []
    with tarfile.open(archive_path) as archive:
        for row in rows:
            key = row["key"]
            tracked_path = mask_root / f"{key}.union.png"
            if not tracked_path.is_file():
                continue
            target = decode(archive, f"{key}.target.jpg", cv2.IMREAD_COLOR)
            mano = decode(archive, f"{key}.mano.png", cv2.IMREAD_COLOR)
            full = decode(archive, f"{key}.mask.png", cv2.IMREAD_GRAYSCALE) > 0
            old = decode(archive, f"{key}.visible_mask.png", cv2.IMREAD_GRAYSCALE) > 0
            hand_masks = {
                hand: cv2.imread(str(mask_root / f"{key}.{name}.png"), cv2.IMREAD_GRAYSCALE) > 0
                for hand, name in ((0, "left"), (1, "right"))
            }
            tracked = hand_masks[0] | hand_masks[1]
            new = tracked & full
            old_stats, new_stats = mask_metrics(old), mask_metrics(new)
            frame_index = int(row["source_frame_index"])
            timestamp = int(tracking[frame_index]["timestamp_ns"])
            hand_quality = {}
            for hand, name in ((0, "left"), (1, "right")):
                annotation_timestamp, annotation = nearest_hand_annotation(
                    boxes, hand_timestamps[hand], hand, timestamp,
                )
                quality = {
                    "visibility": annotation["visibility"],
                    "annotation_delta_ms": abs(annotation_timestamp - timestamp) / 1e6,
                    "pixels": int(hand_masks[hand].sum()),
                }
                try:
                    reference_box = transformed_box(
                        provider, annotation_timestamp, annotation["box"], (1408, 1408), args.size,
                    )
                    quality["reference_box"] = reference_box
                    quality.update(mask_box_metrics(hand_masks[hand], reference_box))
                except RuntimeError:
                    quality.update({"reference_box": None, "bbox_iou": 0.0,
                                    "pixel_inside_box_fraction": 0.0})
                quality["required"] = (quality["visibility"] >= 0.2 and
                                       quality["annotation_delta_ms"] <= 100.0)
                quality["passed"] = (not quality["required"] or
                                     (quality["pixels"] >= 64 and quality["bbox_iou"] >= 0.25 and
                                      quality["pixel_inside_box_fraction"] >= 0.6))
                hand_quality[name] = quality
            topology_passed = (new_stats["components"] <= 5 and
                               new_stats["tiny_component_pixel_fraction"] <= 0.05)
            accepted = topology_passed and all(value["passed"] for value in hand_quality.values())
            comparison = make_comparison(target, mano, full, old, tracked, new,
                                         old_stats["components"], new_stats["components"])
            output = visual_root / f"{key}_video_vs_static.jpg"
            if not cv2.imwrite(str(output), comparison, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise OSError(output)
            cards.append(comparison)
            records.append({
                "key": key,
                "source_frame_index": frame_index,
                "full_pixels": int(full.sum()),
                "old_visible_fraction": old_stats["pixels"] / int(full.sum()),
                "new_visible_fraction": new_stats["pixels"] / int(full.sum()),
                "old": old_stats,
                "new": new_stats,
                "hand_quality": hand_quality,
                "topology_passed": topology_passed,
                "accepted": accepted,
            })
    for page_index in range(0, len(cards), 20):
        save_page(cards[page_index:page_index + 20], root / f"comparison_page_{page_index // 20:02d}.jpg")
    accepted_cards = [card for card, record in zip(cards, records) if record["accepted"]]
    rejected_cards = [card for card, record in zip(cards, records) if not record["accepted"]]
    for name, selected in (("accepted", accepted_cards), ("rejected", rejected_cards)):
        for page_index in range(0, len(selected), 20):
            save_page(selected[page_index:page_index + 20],
                      root / f"{name}_page_{page_index // 20:02d}.jpg")
    severe = lambda item: (item["components"] > 5 or item["tiny_component_pixel_fraction"] > 0.05)
    summary = {
        "sequence": args.sequence,
        "compared_samples": len(records),
        "static_components_gt_5": sum(item["old"]["components"] > 5 for item in records),
        "video_components_gt_5": sum(item["new"]["components"] > 5 for item in records),
        "static_severe": sum(severe(item["old"]) for item in records),
        "video_severe": sum(severe(item["new"]) for item in records),
        "static_mean_visible_fraction": float(np.mean([item["old_visible_fraction"] for item in records])),
        "video_mean_visible_fraction": float(np.mean([item["new_visible_fraction"] for item in records])),
        "accepted_samples": sum(item["accepted"] for item in records),
        "rejected_samples": sum(not item["accepted"] for item in records),
        "filter": {
            "minimum_annotation_visibility": 0.2,
            "maximum_annotation_delta_ms": 100.0,
            "minimum_hand_mask_pixels": 64,
            "minimum_mask_bbox_iou": 0.25,
            "minimum_pixels_inside_bbox_fraction": 0.6,
            "maximum_visible_mask_components": 5,
            "maximum_tiny_component_pixel_fraction": 0.05,
        },
    }
    (root / "comparison_records.jsonl").write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    )
    (root / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for accepted, name in ((True, "selected_frames.jsonl"), (False, "rejected_frames.jsonl")):
        (root / name).write_text("".join(
            json.dumps(record, separators=(",", ":")) + "\n"
            for record in records if record["accepted"] is accepted
        ))
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.stage == "extract":
        stage_extract(args)
    elif args.stage == "track":
        stage_track(args)
    else:
        stage_compare(args)


if __name__ == "__main__":
    main()
