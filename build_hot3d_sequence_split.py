"""Build a deterministic participant-stratified, sequence-disjoint HOT3D split."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def normalize_clip_id(value: object) -> str:
    text = Path(str(value)).stem
    if text.startswith("clip-"):
        return text
    if text.isdigit():
        return f"clip-{int(text):06d}"
    raise ValueError(f"Cannot normalize clip id: {value!r}")


def extract_records(node: object, inherited_id: object | None = None) -> list[dict]:
    records: list[dict] = []
    if isinstance(node, dict):
        if "sequence_id" in node:
            record = dict(node)
            if "participant_id" not in record:
                sequence_id = str(record["sequence_id"])
                participant_id, separator, _ = sequence_id.partition("_")
                if not separator or not participant_id:
                    raise ValueError(
                        "Clip metadata has no participant_id and it cannot be "
                        f"derived from sequence_id: {sequence_id!r}"
                    )
                record["participant_id"] = participant_id
            clip_id = record.get("clip_id", record.get("id", inherited_id))
            if clip_id is None:
                raise ValueError(f"Clip metadata has no clip id: {record}")
            record["clip_id"] = normalize_clip_id(clip_id)
            records.append(record)
        else:
            for key, value in node.items():
                records.extend(extract_records(value, key))
    elif isinstance(node, list):
        for value in node:
            records.extend(extract_records(value, inherited_id))
    return records


def closest_holdout_subset(sequence_counts: dict[str, int], fraction: float, rng: random.Random) -> set[str]:
    names = list(sequence_counts)
    rng.shuffle(names)
    if len(names) < 2:
        return set()
    total = sum(sequence_counts.values())
    target = total * fraction
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for name in names:
        count = sequence_counts[name]
        for subtotal, subset in list(reachable.items())[::-1]:
            reachable.setdefault(subtotal + count, subset + (name,))
    candidates = [(abs(count - target), count, subset) for count, subset in reachable.items() if 0 < len(subset) < len(names)]
    _, _, selected = min(candidates, key=lambda item: (item[0], item[1]))
    return set(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-definitions", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, default=Path("data/train_quest3"))
    parser.add_argument("--output", type=Path, default=Path("configs/hand_restoration/splits/train_quest3_sequence_seed7.json"))
    parser.add_argument("--device", default="Quest3")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--require-existing", action="store_true", help="Only include clip tar files already present under --clips-dir.")
    args = parser.parse_args()
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("holdout-fraction must be between zero and one")

    raw = json.loads(args.clip_definitions.read_text(encoding="utf-8"))
    records = extract_records(raw)
    unique: dict[str, dict] = {}
    for record in records:
        device = str(record.get("device", record.get("device_type", ""))).lower().replace(" ", "")
        wanted = args.device.lower().replace(" ", "")
        if device and device != wanted:
            continue
        path = args.clips_dir / f"{record['clip_id']}.tar"
        if args.require_existing and not path.is_file():
            continue
        record["clip_tar"] = str(path)
        unique[record["clip_id"]] = record
    records = sorted(unique.values(), key=lambda item: item["clip_id"])
    if not records:
        raise RuntimeError("No matching clip definitions found; check metadata structure, device, and clips-dir.")

    by_participant: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_participant[str(record["participant_id"])][str(record["sequence_id"])].append(record)

    train_records: list[dict] = []
    holdout_records: list[dict] = []
    participant_stats = {}
    unsplittable = []
    for participant in sorted(by_participant):
        sequences = by_participant[participant]
        counts = {sequence: len(items) for sequence, items in sequences.items()}
        selected = closest_holdout_subset(counts, args.holdout_fraction, random.Random(f"{args.seed}:{participant}"))
        if not selected:
            unsplittable.append(participant)
        for sequence, items in sequences.items():
            (holdout_records if sequence in selected else train_records).extend(items)
        participant_stats[participant] = {
            "sequence_count": len(sequences),
            "clip_count": sum(counts.values()),
            "train_sequences": sorted(set(sequences) - selected),
            "holdout_sequences": sorted(selected),
            "train_clips": sum(counts[name] for name in set(sequences) - selected),
            "holdout_clips": sum(counts[name] for name in selected),
        }

    train_sequences = {str(item["sequence_id"]) for item in train_records}
    holdout_sequences = {str(item["sequence_id"]) for item in holdout_records}
    overlap = train_sequences & holdout_sequences
    if overlap:
        raise AssertionError(f"Sequence leakage: {sorted(overlap)}")
    train_participants = {str(item["participant_id"]) for item in train_records}
    holdout_participants = {str(item["participant_id"]) for item in holdout_records}
    result = {
        "schema_version": 2,
        "dataset": "HOT3D-Clips/train_quest3",
        "seed": args.seed,
        "split_unit": "sequence_id",
        "participant_overlap_required_when_possible": True,
        "holdout_fraction_target": args.holdout_fraction,
        "clip_definitions": str(args.clip_definitions),
        "train": [item["clip_tar"] for item in sorted(train_records, key=lambda value: value["clip_id"])],
        "holdout": [item["clip_tar"] for item in sorted(holdout_records, key=lambda value: value["clip_id"])],
        "clip_metadata": {item["clip_id"]: {"participant_id": item["participant_id"], "sequence_id": item["sequence_id"], "device": item.get("device", item.get("device_type", args.device))} for item in records},
        "statistics": {
            "total_clips": len(records),
            "train_clips": len(train_records),
            "holdout_clips": len(holdout_records),
            "actual_holdout_fraction": len(holdout_records) / len(records),
            "train_sequences": len(train_sequences),
            "holdout_sequences": len(holdout_sequences),
            "train_participants": sorted(train_participants),
            "holdout_participants": sorted(holdout_participants),
            "participants_in_both": sorted(train_participants & holdout_participants),
            "participants_with_fewer_than_two_sequences": unsplittable,
            "participants": participant_stats,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"clips train={len(train_records)} holdout={len(holdout_records)} fraction={len(holdout_records)/len(records):.4f}")
    print(f"sequences train={len(train_sequences)} holdout={len(holdout_sequences)} overlap=0")
    print(f"participants train={len(train_participants)} holdout={len(holdout_participants)} both={len(train_participants & holdout_participants)}")
    if unsplittable:
        print(f"WARNING participants with <2 sequences (train only): {unsplittable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
