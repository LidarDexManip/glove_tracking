#!/usr/bin/env python3
"""Run the final resumable SAM2 video workflow over all HUGG Aria pinhole sequences."""
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
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=300)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def valid_sequences(input_root: Path, selected: set[str]) -> list[Path]:
    result = []
    for path in input_root.iterdir():
        if not path.is_dir() or not SEQUENCE_RE.fullmatch(path.name):
            continue
        if selected and path.name not in selected:
            continue
        required = (path / "_SUCCESS.json", path / "rgb_214_1_pinhole.mp4")
        if all(item.exists() for item in required):
            result.append(path)
    return sorted(result)


def main() -> None:
    args = arguments()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sequences = valid_sequences(input_root, set(args.sequence))
    if not sequences:
        raise RuntimeError(f"No valid pinhole sequences found under {input_root}")

    progress_path = output_root / "batch_progress.json"
    failures_path = output_root / "batch_failures.json"
    log_path = output_root / "batch.log"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else {}
    run_started = time.time()

    with log_path.open("a", buffering=1) as log:
        for ordinal, sequence in enumerate(sequences, 1):
            success_path = output_root / sequence.name / "_SUCCESS.json"
            if success_path.exists():
                status, return_code = "already_complete", 0
            elif sequence.name in failures and not args.retry_failed:
                status = "previously_failed"
                return_code = int(failures[sequence.name]["return_code"])
            else:
                start_event = {
                    "event": "sequence_start",
                    "ordinal": ordinal,
                    "total": len(sequences),
                    "sequence": sequence.name,
                    "time": utc_now(),
                }
                print(json.dumps(start_event), flush=True)
                log.write(json.dumps(start_event) + "\n")
                command = [
                    sys.executable,
                    str(ROOT / "segment_hugg_aria_pinhole.py"),
                    str(sequence),
                    "--output-root",
                    str(output_root),
                    "--chunk-frames",
                    str(args.chunk_frames),
                ]
                environment = os.environ.copy()
                environment.setdefault("TQDM_DISABLE", "1")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=environment,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                return_code = process.wait()
                status = "complete" if return_code == 0 and success_path.exists() else "failed"
                if status == "failed":
                    failures[sequence.name] = {
                        "return_code": return_code,
                        "time": utc_now(),
                    }
                    write_json_atomic(failures_path, failures)

            completed = sum(
                (output_root / item.name / "_SUCCESS.json").exists() for item in sequences
            )
            event = {
                "event": "sequence_end",
                "ordinal": ordinal,
                "total": len(sequences),
                "sequence": sequence.name,
                "status": status,
                "return_code": return_code,
                "completed_sequences": completed,
                "failed_sequences": len(failures),
                "elapsed_seconds": time.time() - run_started,
                "updated_at": utc_now(),
            }
            write_json_atomic(progress_path, event)
            print(json.dumps(event), flush=True)
            log.write(json.dumps(event) + "\n")

    final = json.loads(progress_path.read_text())
    final["event"] = "batch_end"
    write_json_atomic(progress_path, final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
