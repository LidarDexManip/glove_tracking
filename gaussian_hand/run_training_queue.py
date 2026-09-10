"""Sequential, fail-stop four-GPU training of the prepared 18 MANO-bound hands."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "outputs/gaussian_hand/pretrain_v2"


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight():
    ready = json.loads((PREP / "readiness.json").read_text())
    jobs = []
    for hand in ready["hands"]:
        if hand["status"] not in ("prepared", "prepared_with_warnings"):
            raise ValueError(f"Unprepared hand: {hand['hand']}")
        key = hand["hand"]
        path = ROOT / "gaussian_hand/configs/pretrain_v2" / f"{key}_train.json"
        config = json.loads(path.read_text())
        binding = config["binding"]
        if f"{binding['subject_id']}_{binding['handedness']}" != key:
            raise ValueError(f"Binding mismatch: {key}")
        manifest = Path(config["dataset_root"]) / "frames.jsonl"
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != hand["manifest_sha256"]:
            raise ValueError(f"Dataset changed after preparation: {key}")
        if not (config["strict_surface"] and config["normal_offset_max_m"] == 0
                and config["learning_rates"]["canonical_mesh"] == 0
                and config["learning_rates"]["pose_delta"] == 0
                and config["steps"] == 30000 and config["resolution"] == 512
                and config["distributed_world_size"] == 4
                and config["global_batch_size"] == 4 and config["training_split"] == "all"):
            raise ValueError(f"Unexpected formal training contract: {key}")
        jobs.append({"hand": key, "config": config, "frames": hand["frames"],
                     "manifest_sha256": hand["manifest_sha256"], "status": "pending"})
    if len(jobs) != 18 or len({j["hand"] for j in jobs}) != 18:
        raise ValueError("Expected exactly 18 unique hand models")
    return jobs


def preview_window(manifest):
    records = sorted((json.loads(line) for line in manifest.read_text().splitlines()),
                     key=lambda r: (r["sequence"], r["frame_index"]))
    best = current = []
    for record in records:
        if (current and record["sequence"] == current[-1]["sequence"]
                and record["frame_index"] == current[-1]["frame_index"] + 1):
            current.append(record)
        else:
            current = [record]
        if len(current) > len(best):
            best = current
    start = best[max(0, (len(best) - 60) // 2)]
    return start["sequence"], start["frame_index"], min(60, len(best))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    jobs = preflight()
    if args.preflight_only:
        print(json.dumps({"jobs": len(jobs), "frames": sum(j["frames"] for j in jobs),
                          "gpu_indices": [4, 5, 6, 7], "status": "preflight_passed"}))
        return
    # Shared advisory lock prevents another instance of this launcher starting.
    lock = (PREP / "training_gpu_4_7.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    memory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in memory.splitlines()}
    if any(free.get(g, 999999) > 100 for g in (4, 5, 6, 7)):
        raise RuntimeError("GPUs 4-7 are no longer idle; refusing to start")
    # Never reuse a run directory or overwrite previous checkpoints.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "configs").mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="4,5,6,7", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    state = {"status": "running", "started_at": now(), "pid": os.getpid(),
             "gpu_indices": [4, 5, 6, 7], "world_size": 4,
             "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "jobs": jobs}

    def save():
        state["updated_at"] = now()
        tmp = args.output / "status.json.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(args.output / "status.json")

    save()
    try:
        for job in jobs:
            directory = args.output / job["hand"]
            directory.mkdir()
            config_path = args.output / "configs" / f"{job['hand']}.json"
            config_path.write_text(json.dumps(job["config"], indent=2) + "\n")
            job.update(status="training", started_at=now(), output=str(directory))
            state["current_hand"] = job["hand"]
            save()
            started = time.monotonic()
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nnodes=1", "--nproc_per_node=4", "--max_restarts=0",
                       "-m", "gaussian_hand.train", "--config", str(config_path),
                       "--output", str(directory)]
            job["command"] = command
            with (directory / "training.log").open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                job["launcher_pid"] = child.pid
                save()
                code = child.wait()
            if code:
                raise RuntimeError(f"{job['hand']} training exited with {code}")
            events = []
            for line in (directory / "training.log").read_text().splitlines():
                if line.startswith("{"):
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            completed = [e for e in events if e.get("status") == "complete"]
            if not (completed and completed[-1]["steps"] == 30000
                    and any(e.get("distributed_model_checksums_match") for e in events)
                    and (directory / "step_030000.pt").is_file()):
                raise RuntimeError(f"Incomplete/unverified training: {job['hand']}")
            job.update(status="preview", training_seconds=time.monotonic() - started,
                       training_result=completed[-1])
            save()
            sequence, start, count = preview_window(Path(job["config"]["dataset_root"]) / "frames.jsonl")
            render_dir = directory / "preview_render"
            preview_commands = [
                [sys.executable, "-m", "gaussian_hand.render_sequence", "--checkpoint",
                 str(directory / "step_030000.pt"), "--sequence", sequence, "--output", str(render_dir),
                 "--start-frame", str(start), "--max-frames", str(count)],
                [sys.executable, "-m", "gaussian_hand.make_preview", "--render-directory", str(render_dir),
                 "--output", str(directory / "overlay.mp4"), "--max-frames", str(count), "--overlay"]]
            with (directory / "preview.log").open("w") as log:
                for command in preview_commands:
                    subprocess.run(command, cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": "4"},
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            job.update(status="complete", finished_at=now(), total_seconds=time.monotonic() - started)
            save()
            print(json.dumps({"hand": job["hand"], "status": "complete",
                              "seconds": job["total_seconds"]}), flush=True)
        state.update(status="complete", current_hand=None, finished_at=now())
        save()
    except BaseException as exc:
        job.update(status="failed", error=str(exc))
        state.update(status="failed", error=str(exc), finished_at=now())
        save()
        raise


if __name__ == "__main__":
    main()
