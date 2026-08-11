#!/usr/bin/env python3
"""Verify derived HOT3D shards, manifests, decodability, and split isolation."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tarfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decode(tar: tarfile.TarFile, member: str, flag: int) -> np.ndarray:
    extracted = tar.extractfile(member)
    if extracted is None:
        raise KeyError(member)
    image = cv2.imdecode(np.frombuffer(extracted.read(), np.uint8), flag)
    if image is None:
        raise RuntimeError(f"Cannot decode {member}")
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--decode-samples", type=int, default=100)
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.dataset_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    metadata = split.get("clip_metadata", {})
    train_sequences = {value["sequence_id"] for key, value in metadata.items() if any(Path(path).stem == key for path in split["train"])}
    holdout_sequences = {value["sequence_id"] for key, value in metadata.items() if any(Path(path).stem == key for path in split["holdout"])}
    overlap = train_sequences & holdout_sequences
    if overlap:
        raise RuntimeError(f"Sequence leakage: {sorted(overlap)[:20]}")
    if not metadata:
        print("[WARN] split has no clip_metadata; sequence leakage check unavailable (clip leakage still checked)")

    all_records: dict[str, list[dict]] = {}
    for split_name in ("train", "holdout"):
        manifest = args.dataset_dir / f"{split_name}_manifest.jsonl"
        rows = records(manifest)
        if len(rows) != summary["splits"][split_name]["usable_samples"]:
            raise RuntimeError(f"{split_name} count mismatch")
        if len({row["key"] for row in rows}) != len(rows):
            raise RuntimeError(f"Duplicate keys in {manifest}")
        all_records[split_name] = rows
        grouped = Counter(row["shard"] for row in rows)
        for relative, expected_sha in summary["splits"][split_name]["sha256"].items():
            shard = args.dataset_dir / relative
            if not shard.is_file():
                raise FileNotFoundError(shard)
            if not args.skip_sha256 and digest(shard) != expected_sha:
                raise RuntimeError(f"SHA256 mismatch: {shard}")
            with tarfile.open(shard, "r") as archive:
                names = set(archive.getnames())
            shard_rows = [row for row in rows if row["shard"] == relative]
            for row in shard_rows:
                for suffix in ("target.jpg", "mano.png", "mask.png", "json"):
                    if f"{row['key']}.{suffix}" not in names:
                        raise KeyError(f"Missing {row['key']}.{suffix} in {shard}")
        print(f"[OK] {split_name}: samples={len(rows)} shards={len(grouped)}")

    if {row["clip_id"] for row in all_records["train"]} & {row["clip_id"] for row in all_records["holdout"]}:
        raise RuntimeError("Clip leakage between derived manifests")
    combined = [(name, row) for name, rows in all_records.items() for row in rows]
    rng = random.Random(7)
    chosen = rng.sample(combined, min(args.decode_samples, len(combined)))
    handles: dict[Path, tarfile.TarFile] = {}
    try:
        for _, row in chosen:
            shard = args.dataset_dir / row["shard"]
            archive = handles.setdefault(shard, tarfile.open(shard, "r"))
            target = decode(archive, f"{row['key']}.target.jpg", cv2.IMREAD_GRAYSCALE)
            mano = decode(archive, f"{row['key']}.mano.png", cv2.IMREAD_GRAYSCALE)
            mask = decode(archive, f"{row['key']}.mask.png", cv2.IMREAD_GRAYSCALE)
            if target.shape != mano.shape or target.shape != mask.shape or not np.any(mask):
                raise RuntimeError(f"Invalid alignment/mask for {row['key']}")
    finally:
        for archive in handles.values():
            archive.close()
    print(f"[OK] decoded={len(chosen)}; sequence_overlap=0; clip_overlap=0")
    print("Derived dataset verified. Raw archives may be archived/deleted only after a separate backup decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
