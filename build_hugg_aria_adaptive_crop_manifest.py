#!/usr/bin/env python3
"""Expand a filtered HUGG Aria frame manifest into adaptive hand crops."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from hand_restoration.adaptive_hand_crops import adaptive_hand_crop_instances


ROOT = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--alpha-root",
        type=Path,
        default=ROOT / "data/HUGG_ARIA_MANO_ALIGNED",
    )
    parser.add_argument(
        "--wrist-ring-root",
        type=Path,
        default=ROOT / "data/HUGG_ARIA_MANO_WRIST_RINGS",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--single-hand-scale", type=float, default=1.5)
    parser.add_argument("--close-joint-scale", type=float, default=1.2)
    parser.add_argument("--min-component-area", type=int, default=64)
    parser.add_argument("--ring-match-distance-px", type=float, default=6.0)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"Manifest is empty: {path}")
    return records


def read_alpha(
    capture: cv2.VideoCapture, frame_index: int, next_frame: int
) -> np.ndarray:
    if next_frame != frame_index:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not decode alpha frame {frame_index}")
    return frame[:, :, 0] > 0


def process_sequence(task: tuple) -> dict:
    (
        sequence_order,
        sequence,
        indexed_records,
        alpha_root,
        wrist_ring_root,
        temporary_root,
        single_hand_scale,
        close_joint_scale,
        min_component_area,
        ring_match_distance_px,
    ) = task
    alpha_path = Path(alpha_root) / sequence / "alpha.mkv"
    ring_path = Path(wrist_ring_root) / sequence / "wrist_ring_points.npz"
    capture = cv2.VideoCapture(str(alpha_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open MANO alpha: {alpha_path}")
    with np.load(ring_path) as payload:
        ring_points = payload["points"].astype(np.float32)
        palm_points = payload["palm_points"].astype(np.float32)
        ring_valid = payload["valid"].astype(bool)

    output_path = Path(temporary_root) / f"{sequence_order:04d}_{sequence}.jsonl"
    modes: Counter[str] = Counter()
    handedness: Counter[str] = Counter()
    output_records = 0
    split_source_frames = 0
    next_frame = -1
    try:
        with output_path.open("w", encoding="utf-8") as output:
            for source_index, record in indexed_records:
                frame_index = int(record["frame_index"])
                alpha = read_alpha(capture, frame_index, next_frame)
                next_frame = frame_index + 1
                if not alpha.any():
                    raise RuntimeError(
                        f"Filtered manifest contains empty MANO alpha: "
                        f"{sequence}/{frame_index}"
                    )
                frame_rings = ring_points[frame_index].copy()
                frame_palms = palm_points[frame_index].copy()
                frame_rings[~ring_valid[frame_index]] = np.nan
                frame_palms[~ring_valid[frame_index]] = np.nan
                instances = adaptive_hand_crop_instances(
                    alpha,
                    frame_rings,
                    frame_palms,
                    single_hand_scale=single_hand_scale,
                    close_joint_scale=close_joint_scale,
                    min_component_area=min_component_area,
                    ring_match_distance_px=ring_match_distance_px,
                )
                if len(instances) == 2:
                    split_source_frames += 1
                for instance_index, instance in enumerate(instances):
                    expanded = dict(record)
                    expanded.update(
                        {
                            "source_manifest_index": source_index,
                            "adaptive_crop_mode": instance.mode,
                            "adaptive_crop_scale": instance.scale,
                            "adaptive_crop_box_xyxy": instance.box.as_list(),
                            "adaptive_component_seeds_xy": (
                                None
                                if instance.component_seeds_xy is None
                                else [list(seed) for seed in instance.component_seeds_xy]
                            ),
                            "adaptive_handedness": instance.handedness,
                            "adaptive_component_area": instance.component_area,
                            "adaptive_component_count": instance.component_count,
                            "adaptive_instance_index": instance_index,
                            "adaptive_instance_count": len(instances),
                        }
                    )
                    output.write(json.dumps(expanded, separators=(",", ":")) + "\n")
                    modes[instance.mode] += 1
                    handedness[instance.handedness] += 1
                    output_records += 1
    finally:
        capture.release()
    return {
        "sequence_order": sequence_order,
        "sequence": sequence,
        "path": str(output_path),
        "source_frames": len(indexed_records),
        "output_records": output_records,
        "split_source_frames": split_source_frames,
        "modes": dict(modes),
        "handedness": dict(handedness),
    }


def main() -> None:
    args = arguments()
    source_manifest = args.source_manifest.resolve()
    output_manifest = args.output_manifest.resolve()
    summary_path = args.summary.resolve()
    records = read_manifest(source_manifest)
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    sequence_order = []
    for source_index, record in enumerate(records):
        sequence = str(record["sequence_id"])
        if sequence not in grouped:
            sequence_order.append(sequence)
        grouped[sequence].append((source_index, record))

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".adaptive_crop_manifest.", dir=output_manifest.parent)
    )
    started = time.perf_counter()
    tasks = [
        (
            order,
            sequence,
            grouped[sequence],
            args.alpha_root.resolve(),
            args.wrist_ring_root.resolve(),
            temporary_root,
            args.single_hand_scale,
            args.close_joint_scale,
            args.min_component_area,
            args.ring_match_distance_px,
        )
        for order, sequence in enumerate(sequence_order)
    ]
    results = []
    try:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(process_sequence, task) for task in tasks]
            for complete, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(
                    f"[{complete}/{len(futures)}] {result['sequence']}: "
                    f"{result['source_frames']} -> {result['output_records']}",
                    flush=True,
                )

        results.sort(key=lambda item: item["sequence_order"])
        temporary_manifest = temporary_root / "combined.jsonl"
        with temporary_manifest.open("wb") as destination:
            for result in results:
                with Path(result["path"]).open("rb") as source:
                    shutil.copyfileobj(source, destination)
        temporary_manifest.replace(output_manifest)

        mode_counts: Counter[str] = Counter()
        handedness_counts: Counter[str] = Counter()
        for result in results:
            mode_counts.update(result["modes"])
            handedness_counts.update(result["handedness"])
        output_records = sum(item["output_records"] for item in results)
        split_source_frames = sum(item["split_source_frames"] for item in results)
        summary = {
            "format_version": 1,
            "source_manifest": str(source_manifest),
            "output_manifest": str(output_manifest),
            "source_frames": len(records),
            "output_records": output_records,
            "sequence_count": len(results),
            "split_source_frames": split_source_frames,
            "split_source_fraction": split_source_frames / len(records),
            "single_hand_scale": args.single_hand_scale,
            "close_joint_scale": args.close_joint_scale,
            "split_rule": "individual_1p5_square_crops_are_disjoint",
            "min_component_area": args.min_component_area,
            "ring_match_distance_px": args.ring_match_distance_px,
            "mode_counts": dict(sorted(mode_counts.items())),
            "handedness_counts": dict(sorted(handedness_counts.items())),
            "elapsed_seconds": time.perf_counter() - started,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()
