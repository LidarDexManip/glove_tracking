#!/usr/bin/env python3
"""Build the single filtered, frame-aligned HUGG Aria training manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SEQUENCE_RE = re.compile(r"^P[0-9]+_[0-9a-f]+$")
PINHOLE_HF_REPO = "LIDAR-GT/HUGG_ARIA_PINHOLE"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=ROOT / "outputs/sam2_hugg_aria_masks_v3_pilot",
    )
    parser.add_argument(
        "--pinhole-root",
        type=Path,
        default=ROOT / "data/HUGG_ARIA_PINHOLE",
        help="Local snapshot downloaded from LIDAR-GT/HUGG_ARIA_PINHOLE.",
    )
    parser.add_argument(
        "--mano-root",
        type=Path,
        default=ROOT / "data/HUGG_ARIA_MANO_ALIGNED",
    )
    parser.add_argument(
        "--seen-eval-spec",
        type=Path,
        default=(
            ROOT / "configs/hand_restoration/splits/"
            "hugg_aria_seen_eval_seed7.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/derived/hugg_aria_diffusion",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--expected-sequences", type=int, default=136)
    parser.add_argument("--expected-sam-eligible", type=int, default=399_164)
    parser.add_argument("--expected-train-frames", type=int, default=393_684)
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict]) -> str:
    payload = "".join(
        json.dumps(record, separators=(",", ":")) + "\n"
        for record in records
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_hf_pinhole_root(path: Path) -> None:
    readme = path / "README.md"
    if not readme.is_file() or "# HUGG ARIA Pinhole" not in readme.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError(
            f"{path} is not identifiable as a snapshot of {PINHOLE_HF_REPO}"
        )


def mano_valid_frames(path: Path, expected_rows: int) -> set[int]:
    success_path = path / "_SUCCESS.json"
    alpha_path = path / "alpha.mkv"
    availability_path = path / "mano_availability.csv"
    if not success_path.is_file() or not alpha_path.is_file():
        raise FileNotFoundError(
            f"MANO RGB/alpha asset is incomplete: {path}"
        )
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if int(success.get("format_version", 0)) < 3:
        raise RuntimeError(
            f"MANO asset predates explicit lossless alpha export: {path}"
        )
    valid: set[int] = set()
    rows = 0
    with availability_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            if row["mano_valid"].strip() == "1":
                valid.add(int(row["frame_index"]))
    if rows != expected_rows:
        raise RuntimeError(
            f"MANO availability row mismatch for {path.name}: "
            f"{rows} != {expected_rows}"
        )
    if len(valid) != int(success["mano_valid_frames"]):
        raise RuntimeError(f"MANO valid count mismatch for {path.name}")
    return valid


def condition_valid_frames(
    path: Path,
    rows: list[tuple[int, int, bytes]],
    sam_shape: tuple[int, int],
) -> tuple[set[int], int, int]:
    raster_valid = mano_valid_frames(path, expected_rows=len(rows))
    capture = cv2.VideoCapture(str(path / "alpha.mkv"))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open MANO alpha: {path}")
    condition_valid: set[int] = set()
    no_overlap = 0
    next_frame = -1
    try:
        for frame_index, _, labels_zlib in rows:
            frame_index = int(frame_index)
            if frame_index not in raster_valid:
                continue
            if next_frame != frame_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, alpha_bgr = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not decode MANO alpha {path.name}/{frame_index}"
                )
            next_frame = frame_index + 1
            alpha = alpha_bgr[:, :, 0] > 0
            labels = np.frombuffer(
                zlib.decompress(labels_zlib), dtype=np.uint8
            ).reshape(sam_shape)
            sam = cv2.resize(
                (labels > 0).astype(np.uint8),
                (alpha.shape[1], alpha.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            if np.any(alpha & sam):
                condition_valid.add(frame_index)
            else:
                no_overlap += 1
    finally:
        capture.release()
    empty_raster = len(rows) - len(raster_valid)
    return condition_valid, empty_raster, no_overlap


def main() -> None:
    args = arguments()
    mask_root = args.mask_root.resolve()
    pinhole_root = args.pinhole_root.resolve()
    mano_root = args.mano_root.resolve()
    output = args.output_dir.resolve()
    validate_hf_pinhole_root(pinhole_root)

    sequence_dirs = sorted(
        path
        for path in mask_root.iterdir()
        if path.is_dir()
        and SEQUENCE_RE.fullmatch(path.name)
        and (path / "masks.sqlite").is_file()
        and (path / "_SUCCESS.json").is_file()
    )
    if len(sequence_dirs) != args.expected_sequences:
        raise RuntimeError(
            f"Expected {args.expected_sequences} complete SAM sequences, "
            f"found {len(sequence_dirs)}"
        )

    records_by_sequence: dict[str, list[dict]] = {}
    sam_eligible_counts: Counter[str] = Counter()
    mano_empty_counts: Counter[str] = Counter()
    mano_sam_no_overlap_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for sequence_dir in sequence_dirs:
        sequence = sequence_dir.name
        pinhole_video = (
            pinhole_root / sequence / "rgb_214_1_pinhole.mp4"
        )
        if not pinhole_video.is_file():
            raise FileNotFoundError(pinhole_video)

        connection = sqlite3.connect(sequence_dir / "masks.sqlite")
        try:
            rows = connection.execute(
                "SELECT frame_index,timestamp_ns,labels_zlib FROM frames "
                "WHERE training_eligible=1 ORDER BY frame_index"
            ).fetchall()
            source_counts[sequence] = int(
                connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
            )
            database_metadata = {
                key: json.loads(value)
                for key, value in connection.execute(
                    "SELECT key,value_json FROM metadata"
                )
            }
        finally:
            connection.close()

        sam_eligible_counts[sequence] = len(rows)
        valid_condition, empty_raster, no_overlap = condition_valid_frames(
            mano_root / sequence,
            rows,
            (
                int(database_metadata["height"]),
                int(database_metadata["width"]),
            ),
        )
        records = [
            {
                "sequence_id": sequence,
                "frame_index": int(frame_index),
                "timestamp_ns": int(timestamp_ns),
            }
            for frame_index, timestamp_ns, _ in rows
            if int(frame_index) in valid_condition
        ]
        mano_empty_counts[sequence] = empty_raster
        mano_sam_no_overlap_counts[sequence] = no_overlap
        if not records:
            raise RuntimeError(
                f"No jointly eligible SAM/MANO-alpha frames for {sequence}"
            )
        records_by_sequence[sequence] = records

    sam_eligible_total = sum(sam_eligible_counts.values())
    if sam_eligible_total != args.expected_sam_eligible:
        raise RuntimeError(
            f"Expected {args.expected_sam_eligible} SAM-eligible frames, "
            f"found {sam_eligible_total}"
        )

    train_records = [
        record
        for sequence in sorted(records_by_sequence)
        for record in records_by_sequence[sequence]
    ]
    if (
        args.expected_train_frames > 0
        and len(train_records) != args.expected_train_frames
    ):
        raise RuntimeError(
            f"Expected {args.expected_train_frames} jointly eligible frames, "
            f"found {len(train_records)}"
        )

    eligible = {
        (str(record["sequence_id"]), int(record["frame_index"])): record
        for record in train_records
    }
    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for record in train_records:
        by_sequence[str(record["sequence_id"])].append(record)

    eval_spec = load_jsonl(args.seen_eval_spec.resolve())
    if len(eval_spec) != args.expected_sequences:
        raise RuntimeError("Seen-eval spec must have one row per sequence")
    eval_records = []
    replacements = []
    for requested in eval_spec:
        sequence = str(requested["sequence_id"])
        frame_index = int(requested["frame_index"])
        record = eligible.get((sequence, frame_index))
        if record is None:
            record = min(
                by_sequence[sequence],
                key=lambda item: (
                    abs(int(item["frame_index"]) - frame_index),
                    int(item["frame_index"]),
                ),
            )
            replacements.append({
                "sequence_id": sequence,
                "original_frame_index": frame_index,
                "replacement_frame_index": int(record["frame_index"]),
            })
        eval_records.append(record)
    if len({record["sequence_id"] for record in eval_records}) != len(
        sequence_dirs
    ):
        raise RuntimeError("Seen-eval selection did not preserve every sequence")

    output.mkdir(parents=True, exist_ok=True)
    train_hash = write_jsonl(output / "train_manifest.jsonl", train_records)
    eval_hash = write_jsonl(
        output / "seen_eval_manifest.jsonl", eval_records
    )
    summary = {
        "schema_version": 3,
        "policy": (
            "SAM training_eligible AND nonempty aligned MANO alpha AND "
            "nonempty MANO-alpha/SAM intersection; single source "
            "frame_index for every asset"
        ),
        "pinhole_hf_repo": PINHOLE_HF_REPO,
        "pinhole_root": str(pinhole_root),
        "mask_root": str(mask_root),
        "mano_root": str(mano_root),
        "sequence_count": len(records_by_sequence),
        "source_frames": sum(source_counts.values()),
        "sam_training_eligible_frames": sam_eligible_total,
        "removed_empty_mano_alpha_frames": sum(mano_empty_counts.values()),
        "removed_mano_sam_no_overlap_frames": sum(
            mano_sam_no_overlap_counts.values()
        ),
        "train_frames": len(train_records),
        "seen_eval_frames": len(eval_records),
        "seen_eval_is_subset_of_train": True,
        "seen_eval_replacements": replacements,
        "seed": args.seed,
        "train_manifest_sha256": train_hash,
        "seen_eval_manifest_sha256": eval_hash,
        "eligible_by_sequence": {
            sequence: len(records_by_sequence[sequence])
            for sequence in sorted(records_by_sequence)
        },
        "removed_empty_mano_alpha_by_sequence": {
            sequence: mano_empty_counts[sequence]
            for sequence in sorted(records_by_sequence)
            if mano_empty_counts[sequence]
        },
        "removed_mano_sam_no_overlap_by_sequence": {
            sequence: mano_sam_no_overlap_counts[sequence]
            for sequence in sorted(records_by_sequence)
            if mano_sam_no_overlap_counts[sequence]
        },
    }
    (output / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
