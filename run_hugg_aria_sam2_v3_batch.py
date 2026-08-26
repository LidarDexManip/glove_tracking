#!/usr/bin/env python3
"""Run resumable SAM2 v3 over the 136 frame-aligned pinhole sequences."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEQUENCE_RE = re.compile(r"^P[0-9]+_[0-9a-f]+$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=ROOT / "data/HUGG_ARIA_PINHOLE")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/sam2_hugg_aria_masks_v3_pilot")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def valid_sequences(input_root: Path, selected: set[str]) -> list[Path]:
    result = []
    for path in input_root.iterdir():
        if not path.is_dir() or not SEQUENCE_RE.fullmatch(path.name):
            continue
        if selected and path.name not in selected:
            continue
        required = (
            path / "_SUCCESS.json",
            path / "rgb_214_1_pinhole.mp4",
            path / "masks/mask_hand_pose_available.csv",
            path / "masks/mask_qa_pass.csv",
            path / "masks/mask_hand_visible.csv",
            path / "masks/mask_good_exposure.csv",
            ROOT / "data/HUGG_ARIA_SAM_SUPPORT" / path.name / "box2d_hands.csv",
        )
        if all(item.exists() for item in required):
            result.append(path)
    return sorted(result)


def main() -> None:
    args = arguments()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "_batch_logs"
    logs.mkdir(exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")
    sequences = valid_sequences(input_root, set(args.sequence))
    if args.max_sequences:
        sequences = sequences[:args.max_sequences]
    if len(sequences) != 136 and not (args.sequence or args.max_sequences):
        raise RuntimeError(f"Expected 136 runnable sequences, found {len(sequences)}")

    failures_path = output_root / "batch_failures.json"
    previous_failures = (
        json.loads(failures_path.read_text(encoding="utf-8"))
        if failures_path.exists() else {}
    )
    pending = []
    for sequence in sequences:
        complete = ((output_root / sequence.name / "_SUCCESS.json").exists()
                    and (output_root / sequence.name / "masks.sqlite").exists())
        if complete:
            continue
        if sequence.name in previous_failures and not args.retry_failed:
            continue
        pending.append(sequence)

    manifest = {
        "format_version": 3,
        "created_at": utc_now(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "sequence_count": len(sequences),
        "initially_complete": len(sequences) - len(pending),
        "gpus": gpus,
        "worker": "segment_hugg_aria_pinhole_v3.py",
        "policy": "episode-start prompt; infer first; post-filter using QA/pose/visibility/exposure; render-alpha filtering belongs to the final manifest",
        "sequences": [item.name for item in sequences],
    }
    write_json_atomic(output_root / "batch_manifest.json", manifest)

    run_started = time.time()
    queue = iter(pending)
    active: dict[str, tuple[subprocess.Popen, Path, object, float]] = {}
    failures: dict[str, dict] = {}
    write_json_atomic(failures_path, failures)
    completed_this_run = 0

    def launch(gpu: str, sequence: Path) -> None:
        log_handle = (logs / f"{sequence.name}.log").open("a", buffering=1,
                                                             encoding="utf-8")
        command = [
            sys.executable, str(ROOT / "segment_hugg_aria_pinhole_v3.py"),
            str(sequence), "--output-root", str(output_root),
        ]
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": gpu,
            "TQDM_DISABLE": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        })
        process = subprocess.Popen(command, stdout=log_handle,
                                   stderr=subprocess.STDOUT, env=environment)
        active[gpu] = (process, sequence, log_handle, time.time())
        print(json.dumps({"event": "start", "gpu": gpu,
                          "sequence": sequence.name, "time": utc_now()}), flush=True)

    for gpu in gpus:
        try:
            launch(gpu, next(queue))
        except StopIteration:
            break

    while active:
        time.sleep(1)
        for gpu, (process, sequence, log_handle, started) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del active[gpu]
            success = ((output_root / sequence.name / "_SUCCESS.json").exists()
                       and (output_root / sequence.name / "masks.sqlite").exists())
            if return_code == 0 and success:
                completed_this_run += 1
                status = "complete"
            else:
                status = "failed"
                failures[sequence.name] = {
                    "return_code": return_code, "gpu": gpu,
                    "time": utc_now(),
                    "log": str(logs / f"{sequence.name}.log"),
                }
                write_json_atomic(failures_path, failures)
            completed_total = sum(
                ((output_root / item.name / "_SUCCESS.json").exists()
                 and (output_root / item.name / "masks.sqlite").exists())
                for item in sequences
            )
            progress = {
                "event": "sequence_end", "updated_at": utc_now(),
                "sequence": sequence.name, "gpu": gpu, "status": status,
                "return_code": return_code,
                "sequence_seconds": time.time() - started,
                "completed_sequences": completed_total,
                "total_sequences": len(sequences),
                "active_sequences": {key: value[1].name
                                     for key, value in active.items()},
                "failed_sequences": len(failures),
                "elapsed_seconds": time.time() - run_started,
            }
            write_json_atomic(output_root / "batch_progress.json", progress)
            print(json.dumps(progress), flush=True)
            try:
                launch(gpu, next(queue))
            except StopIteration:
                pass

    completed_total = sum(
        ((output_root / item.name / "_SUCCESS.json").exists()
         and (output_root / item.name / "masks.sqlite").exists())
        for item in sequences
    )
    final = {
        "event": "batch_end", "updated_at": utc_now(),
        "completed_sequences": completed_total,
        "completed_this_run": completed_this_run,
        "total_sequences": len(sequences),
        "failed_sequences": len(failures),
        "elapsed_seconds": time.time() - run_started,
    }
    write_json_atomic(output_root / "batch_progress.json", final)
    print(json.dumps(final), flush=True)
    if completed_total != len(sequences):
        raise RuntimeError(
            f"SAM batch incomplete: {completed_total}/{len(sequences)}, "
            f"failures={len(failures)}"
        )


if __name__ == "__main__":
    main()
