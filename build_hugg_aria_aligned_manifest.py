#!/usr/bin/env python3
"""Build the frame-aligned HUGG Aria training and seen-eval manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEQUENCE_RE = re.compile(r"^P[0-9]+_[0-9a-f]+$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-root", type=Path,
        default=ROOT / "outputs/sam2_hugg_aria_masks_v3_pilot",
    )
    parser.add_argument(
        "--pinhole-root", type=Path,
        default=ROOT / "data/HUGG_ARIA_PINHOLE",
    )
    parser.add_argument(
        "--gaussian-root", type=Path,
        default=ROOT / "data/HUGG_ARIA_GAUSSIANS_ALIGNED",
    )
    parser.add_argument(
        "--seen-eval-spec", type=Path,
        default=(
            ROOT / "configs/hand_restoration/splits/"
            "hugg_aria_seen_eval_seed7.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/derived/hugg_aria_diffusion_aligned",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--expected-sequences", type=int, default=136)
    parser.add_argument("--expected-train-frames", type=int, default=399_164)
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict]) -> str:
    payload = "".join(
        json.dumps(record, separators=(",", ":")) + "\n" for record in records
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_eval_spec(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = arguments()
    mask_root = args.mask_root.resolve()
    pinhole_root = args.pinhole_root.resolve()
    gaussian_root = args.gaussian_root.resolve()
    output = args.output_dir.resolve()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")

    sequence_dirs = sorted(
        path for path in mask_root.iterdir()
        if path.is_dir()
        and SEQUENCE_RE.fullmatch(path.name)
        and (path / "masks.sqlite").is_file()
        and (path / "_SUCCESS.json").is_file()
    )
    if len(sequence_dirs) != args.expected_sequences:
        raise RuntimeError(
            f"Expected {args.expected_sequences} complete SAM v3 sequences, "
            f"found {len(sequence_dirs)}"
        )

    records_by_sequence: dict[str, list[dict]] = {}
    rejected_by_sequence: dict[str, int] = {}
    for sequence_dir in sequence_dirs:
        sequence = sequence_dir.name
        pinhole_video = pinhole_root / sequence / "rgb_214_1_pinhole.mp4"
        gaussian_video = gaussian_root / sequence / "reconstruction.mp4"
        if not pinhole_video.is_file() or not gaussian_video.is_file():
            raise FileNotFoundError(
                f"Missing aligned input for {sequence}: "
                f"pinhole={pinhole_video.is_file()} gaussian={gaussian_video.is_file()}"
            )
        connection = sqlite3.connect(sequence_dir / "masks.sqlite")
        try:
            rows = connection.execute(
                "SELECT frame_index,timestamp_ns FROM frames "
                "WHERE training_eligible=1 ORDER BY frame_index"
            ).fetchall()
            total = int(connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
        finally:
            connection.close()
        records_by_sequence[sequence] = [
            {
                "sequence_id": sequence,
                "frame_index": int(frame_index),
                "gaussian_frame_index": int(frame_index),
                "timestamp_ns": int(timestamp_ns),
            }
            for frame_index, timestamp_ns in rows
        ]
        rejected_by_sequence[sequence] = total - len(rows)

    blocks: list[list[dict]] = []
    for sequence in sorted(records_by_sequence):
        records = records_by_sequence[sequence]
        blocks.extend(
            records[start:start + args.block_size]
            for start in range(0, len(records), args.block_size)
        )
    random.Random(args.seed).shuffle(blocks)
    train_records = [record for block in blocks for record in block]
    if len(train_records) != args.expected_train_frames:
        raise RuntimeError(
            f"Expected {args.expected_train_frames} eligible frames, "
            f"found {len(train_records)}"
        )

    eligible_keys = {
        (record["sequence_id"], record["frame_index"]): record
        for record in train_records
    }
    eval_spec = load_eval_spec(args.seen_eval_spec.resolve())
    eval_sequences = [str(record["sequence_id"]) for record in eval_spec]
    if len(eval_spec) != args.expected_sequences or len(set(eval_sequences)) != len(eval_spec):
        raise RuntimeError(
            "Seen-eval spec must contain exactly one record for every sequence"
        )
    seen_eval_records = []
    for requested in eval_spec:
        key = (str(requested["sequence_id"]), int(requested["frame_index"]))
        if key not in eligible_keys:
            raise RuntimeError(f"Seen-eval record is not training eligible: {key}")
        seen_eval_records.append(eligible_keys[key])
    if set(eval_sequences) != set(records_by_sequence):
        raise RuntimeError("Seen-eval spec and SAM sequence sets differ")

    output.mkdir(parents=True, exist_ok=True)
    train_hash = write_jsonl(output / "train_manifest.jsonl", train_records)
    eval_hash = write_jsonl(output / "seen_eval_manifest.jsonl", seen_eval_records)
    summary = {
        "schema_version": 2,
        "policy": (
            "SAM v3 training_eligible frames; all 136 sequences are training domain"
        ),
        "gaussian_index_policy": (
            "identity: aligned renderer preserves every source frame, including "
            "empty/invalid-hand frames"
        ),
        "sequence_count": len(records_by_sequence),
        "train_frames": len(train_records),
        "seen_eval_frames": len(seen_eval_records),
        "seen_eval_is_subset_of_train": True,
        "seed": args.seed,
        "block_size": args.block_size,
        "renderer_commit": "18d74ba1bade042335a563640d3d38407e582c1e",
        "train_manifest_sha256": train_hash,
        "seen_eval_manifest_sha256": eval_hash,
        "rejected_frames": sum(rejected_by_sequence.values()),
        "eligible_by_sequence": {
            sequence: len(records_by_sequence[sequence])
            for sequence in sorted(records_by_sequence)
        },
        "rejected_by_sequence": {
            sequence: rejected_by_sequence[sequence]
            for sequence in sorted(rejected_by_sequence)
        },
    }
    (output / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"train_frames={len(train_records)} "
        f"seen_eval_frames={len(seen_eval_records)} "
        f"sequences={len(records_by_sequence)} "
        f"train_sha256={train_hash} eval_sha256={eval_hash}"
    )


if __name__ == "__main__":
    main()
