#!/usr/bin/env python3
"""Build the exact common-frame manifest for Gaussian versus MANO ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=ROOT / "data/derived/hugg_aria_diffusion_aligned/train_manifest.jsonl",
    )
    parser.add_argument(
        "--seen-eval-manifest",
        type=Path,
        default=ROOT / "data/derived/hugg_aria_diffusion_aligned/seen_eval_manifest.jsonl",
    )
    parser.add_argument(
        "--mano-root", type=Path, default=ROOT / "data/HUGG_ARIA_MANO_ALIGNED"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/derived/hugg_aria_diffusion_ablation_common",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def records(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def availability(path: Path, expected: int) -> set[int]:
    success = json.loads((path / "_SUCCESS.json").read_text())
    if int(success.get("format_version", 0)) != 2:
        raise RuntimeError(f"MANO render does not have availability format v2: {path}")
    result = set()
    rows = 0
    with (path / "mano_availability.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            if row["mano_valid"].strip() == "1":
                result.add(int(row["frame_index"]))
    if rows != expected:
        raise RuntimeError(
            f"Availability count mismatch for {path.name}: {rows} != {expected}"
        )
    if len(result) != int(success["mano_valid_frames"]):
        raise RuntimeError(f"Availability valid count mismatch for {path.name}")
    return result


def write_jsonl(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> None:
    args = arguments()
    train_path = args.train_manifest.resolve()
    eval_path = args.seen_eval_manifest.resolve()
    train = records(train_path)
    seen_eval = records(eval_path)
    original_counts = Counter(str(record["sequence_id"]) for record in train)

    valid_by_sequence = {}
    for sequence, expected in sorted(original_counts.items()):
        valid_by_sequence[sequence] = availability(
            args.mano_root.resolve() / sequence, expected
        )

    common = [
        record
        for record in train
        if int(record["frame_index"])
        in valid_by_sequence[str(record["sequence_id"])]
    ]
    common_by_sequence: dict[str, list[dict]] = defaultdict(list)
    for record in common:
        common_by_sequence[str(record["sequence_id"])].append(record)

    missing_sequences = sorted(set(original_counts) - set(common_by_sequence))
    if missing_sequences:
        raise RuntimeError(
            "No common Gaussian/MANO frames for sequences: "
            + ", ".join(missing_sequences)
        )

    selected_eval = []
    replacements = []
    for record in seen_eval:
        sequence = str(record["sequence_id"])
        frame = int(record["frame_index"])
        if frame in valid_by_sequence[sequence]:
            selected_eval.append(record)
            continue
        replacement = min(
            common_by_sequence[sequence],
            key=lambda item: (
                abs(int(item["frame_index"]) - frame),
                int(item["frame_index"]),
            ),
        )
        selected_eval.append(replacement)
        replacements.append(
            {
                "sequence_id": sequence,
                "original_frame_index": frame,
                "replacement_frame_index": int(replacement["frame_index"]),
            }
        )

    if len({str(item["sequence_id"]) for item in selected_eval}) != len(seen_eval):
        raise RuntimeError("Seen-eval replacement did not preserve one frame per sequence")

    output = args.output_root.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial.", dir=output.parent))
    try:
        write_jsonl(temporary / "train_manifest.jsonl", common)
        write_jsonl(temporary / "seen_eval_manifest.jsonl", selected_eval)
        common_counts = Counter(str(record["sequence_id"]) for record in common)
        summary = {
            "format_version": 1,
            "policy": "original_filtered_manifest_intersect_mano_raster_valid",
            "original_train_manifest": str(train_path),
            "original_train_sha256": file_hash(train_path),
            "mano_root": str(args.mano_root.resolve()),
            "original_train_samples": len(train),
            "common_train_samples": len(common),
            "removed_mano_invalid_samples": len(train) - len(common),
            "sequences": len(common_counts),
            "seen_eval_samples": len(selected_eval),
            "seen_eval_replacements": replacements,
            "per_sequence": {
                sequence: {
                    "original": original_counts[sequence],
                    "common": common_counts[sequence],
                    "removed": original_counts[sequence] - common_counts[sequence],
                }
                for sequence in sorted(original_counts)
            },
        }
        (temporary / "manifest_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
        print(json.dumps(summary, indent=2))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
