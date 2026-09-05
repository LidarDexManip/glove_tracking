"""Render all configured subject sequences across an explicit GPU list."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

from gaussian_hand.identity import SubjectBinding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subject-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--time-limit-seconds", type=int, default=1800)
    args = parser.parse_args()
    binding = SubjectBinding.from_json(args.subject_config)
    checkpoint = args.checkpoint.resolve(strict=True)
    import torch
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    binding.assert_compatible(metadata["binding"])
    del metadata
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        raise ValueError("Pass unique numeric GPU indices")
    args.output.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    for sequence in binding.sequences:
        pending.put(sequence)
    started = time.monotonic()
    deadline = started + args.time_limit_seconds

    def worker(gpu):
        results = []
        while True:
            try:
                sequence = pending.get_nowait()
            except queue.Empty:
                return results
            directory = args.output / sequence
            directory.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
            command = [sys.executable, "-m", "gaussian_hand.render_sequence",
                       "--checkpoint", str(checkpoint), "--sequence", sequence,
                       "--output", str(directory), "--size", str(args.size)]
            record = {"sequence": sequence, "gpu": int(gpu)}
            t0 = time.monotonic()
            remaining = deadline - t0
            if remaining <= 0:
                record.update(status="timeout_before_start")
                results.append(record)
                continue
            with (directory / "render.log").open("w") as log:
                try:
                    run = subprocess.run(command, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=remaining)
                    record.update(status="complete" if run.returncode == 0 else "failed",
                                  returncode=run.returncode)
                except subprocess.TimeoutExpired:
                    record.update(status="timeout")
            record["elapsed_seconds"] = time.monotonic() - t0
            if record["status"] == "complete":
                summary = json.loads((directory / "summary.json").read_text())
                counts = {}
                for name in ("render.mp4", "alpha.mkv"):
                    probe = subprocess.check_output([
                        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "json",
                        str(directory / name)], text=True)
                    counts[name] = int(json.loads(probe)["streams"][0]["nb_read_frames"])
                record.update(total_frames=summary["total_frames"],
                              rendered_mano_frames=summary["rendered_mano_frames"],
                              verified_video_frame_counts=counts)
                if any(n != summary["total_frames"] for n in counts.values()):
                    record["status"] = "frame_count_mismatch"
            print(json.dumps(record), flush=True)
            results.append(record)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        results = [record for group in executor.map(worker, gpus) for record in group]
    report = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
              "subject_id": binding.subject_id, "handedness": binding.handedness,
              "elapsed_seconds": time.monotonic() - started,
              "results": sorted(results, key=lambda x: x["sequence"])}
    (args.output / "render_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(r["status"] != "complete" for r in results):
        raise SystemExit("Some renders failed; see render_report.json and per-sequence logs")


if __name__ == "__main__":
    main()
