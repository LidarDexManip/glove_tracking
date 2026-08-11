#!/usr/bin/env python3
"""Download, process, validate, and upload all HUGG_ARIA pinhole sequences."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parent
SEQUENCE_RE = re.compile(r"^P[0-9]+_[0-9a-f]+$")
CORE_FILES = (
    "recording.vrs",
    "mps/slam/online_calibration.jsonl",
    "mps/slam/summary.json",
)
SUPPORT_FILES = (
    "headset_trajectory.csv",
    "mano_hand_pose_trajectory.jsonl",
    "umetrack_hand_pose_trajectory.jsonl",
    "umetrack_hand_user_profile.json",
    "timecode_devicetime_mapping.csv",
    "metadata.json",
    "license.txt",
    "masks/mask_hand_pose_available.csv",
    "masks/mask_headset_pose_available.csv",
    "masks/mask_hand_visible.csv",
    "masks/mask_good_exposure.csv",
    "masks/mask_qa_pass.csv",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", default="LIDAR-GT/HUGG_ARIA")
    parser.add_argument("--target-repo", default="LIDAR-GT/HUGG_ARIA_PINHOLE")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/HUGG_ARIA_PINHOLE",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=ROOT / "data/HUGG_ARIA_PINHOLE_STAGING",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache/huggingface",
    )
    parser.add_argument(
        "--local-source-root",
        type=Path,
        default=ROOT / "data/hot3d_aria",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def required_repo_paths(sequence: str, repo_files: set[str]) -> list[str]:
    result = []
    for relative in CORE_FILES + SUPPORT_FILES:
        path = f"{sequence}/{relative}"
        if path in repo_files:
            result.append(path)
        elif relative in CORE_FILES[:2]:
            raise FileNotFoundError(f"Required source file is missing: {path}")
    return result


def stage_sequence(
    sequence: str,
    source_repo: str,
    source_revision: str,
    repo_files: set[str],
    staging_root: Path,
    cache_dir: Path,
    local_source_root: Path,
) -> tuple[Path, bool]:
    local = local_source_root / sequence
    if (local / "recording.vrs").exists() and (
        local / "mps/slam/online_calibration.jsonl"
    ).exists():
        return local.resolve(), False

    final = staging_root / sequence
    marker = final / ".STAGED"
    if marker.exists():
        return final, True
    partial = staging_root / f".{sequence}.partial.{os.getpid()}"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        for repo_path in required_repo_paths(sequence, repo_files):
            cached = Path(
                hf_hub_download(
                    repo_id=source_repo,
                    repo_type="dataset",
                    filename=repo_path,
                    revision=source_revision,
                    cache_dir=str(cache_dir),
                )
            )
            relative = Path(repo_path).relative_to(sequence)
            destination = partial / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(cached)
        marker_payload = {
            "sequence": sequence,
            "source_repo": source_repo,
            "source_revision": source_revision,
            "staged_at": utc_now(),
        }
        (partial / ".STAGED").write_text(json.dumps(marker_payload, indent=2) + "\n")
        if final.exists():
            shutil.rmtree(final)
        partial.rename(final)
        return final, True
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def process_worker(payload: dict) -> dict:
    sequence = payload["sequence"]
    started = time.perf_counter()
    log_path = Path(payload["log_root"]) / f"{sequence}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        staged, disposable = stage_sequence(
            sequence=sequence,
            source_repo=payload["source_repo"],
            source_revision=payload["source_revision"],
            repo_files=set(payload["repo_files"]),
            staging_root=Path(payload["staging_root"]),
            cache_dir=Path(payload["cache_dir"]),
            local_source_root=Path(payload["local_source_root"]),
        )
        command = [
            sys.executable,
            str(ROOT / "prepare_hugg_aria_pinhole.py"),
            str(staged),
            "--output-root",
            payload["output_root"],
            "--source-repo",
            payload["source_repo"],
            "--source-revision",
            payload["source_revision"],
        ]
        with log_path.open("a") as log:
            log.write(f"\n[{utc_now()}] {' '.join(command)}\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        return {
            "sequence": sequence,
            "status": "processed",
            "output": str(Path(payload["output_root"]) / sequence),
            "staging": str(staged) if disposable else None,
            "seconds": time.perf_counter() - started,
            "log": str(log_path),
        }
    except BaseException as error:
        with log_path.open("a") as log:
            log.write(f"[{utc_now()}] ERROR {type(error).__name__}: {error}\n")
        return {
            "sequence": sequence,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - started,
            "log": str(log_path),
        }


def upload_sequence(api: HfApi, result: dict, target_repo: str, state_root: Path) -> None:
    sequence = result["sequence"]
    marker = state_root / "uploaded" / f"{sequence}.json"
    if marker.exists():
        result["upload_status"] = "already_uploaded"
        return
    api.upload_folder(
        repo_id=target_repo,
        repo_type="dataset",
        folder_path=result["output"],
        path_in_repo=sequence,
        commit_message=f"Add official pinhole sequence {sequence}",
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {"sequence": sequence, "target_repo": target_repo, "uploaded_at": utc_now()},
            indent=2,
        )
        + "\n"
    )
    result["upload_status"] = "uploaded"


def append_status(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({**result, "reported_at": utc_now()}) + "\n")


def main() -> None:
    args = arguments()
    for path in (args.output_root, args.staging_root, args.cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    state_root = args.output_root / "_state"
    log_root = state_root / "logs"
    status_path = state_root / "batch_status.jsonl"
    state_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    source_info = api.repo_info(args.source_repo, repo_type="dataset")
    source_revision = source_info.sha
    repo_files = set(
        api.list_repo_files(
            args.source_repo, repo_type="dataset", revision=source_revision
        )
    )
    sequences = sorted(
        {
            path.split("/", 1)[0]
            for path in repo_files
            if path.endswith("/recording.vrs")
            and SEQUENCE_RE.fullmatch(path.split("/", 1)[0])
        }
    )
    if args.sequence:
        requested = set(args.sequence)
        missing = requested - set(sequences)
        if missing:
            raise ValueError(f"Unknown sequences: {sorted(missing)}")
        sequences = [sequence for sequence in sequences if sequence in requested]
    if args.limit:
        sequences = sequences[: args.limit]
    manifest = {
        "source_repo": args.source_repo,
        "source_revision": source_revision,
        "target_repo": None if args.no_upload else args.target_repo,
        "sequences": len(sequences),
        "workers": args.workers,
        "started_at": utc_now(),
    }
    (state_root / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)

    payload_base = {
        "source_repo": args.source_repo,
        "source_revision": source_revision,
        "repo_files": sorted(repo_files),
        "staging_root": str(args.staging_root),
        "cache_dir": str(args.cache_dir),
        "local_source_root": str(args.local_source_root),
        "output_root": str(args.output_root),
        "log_root": str(log_root),
    }
    completed = failed = uploaded = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(process_worker, {**payload_base, "sequence": sequence}): sequence
            for sequence in sequences
        }
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                sequence = pending.pop(future)
                try:
                    result = future.result()
                except BaseException as error:
                    result = {
                        "sequence": sequence,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                if result["status"] == "processed":
                    completed += 1
                    if not args.no_upload:
                        try:
                            upload_sequence(api, result, args.target_repo, state_root)
                            if result.get("upload_status") in ("uploaded", "already_uploaded"):
                                uploaded += 1
                            staging = result.get("staging")
                            if staging:
                                shutil.rmtree(staging, ignore_errors=True)
                        except BaseException as error:
                            result["upload_status"] = "failed"
                            result["upload_error"] = f"{type(error).__name__}: {error}"
                else:
                    failed += 1
                result["progress"] = {
                    "processed": completed,
                    "uploaded": uploaded,
                    "failed": failed,
                    "total": len(sequences),
                }
                append_status(status_path, result)
                print(json.dumps(result), flush=True)

    final = {
        **manifest,
        "completed_at": utc_now(),
        "processed": completed,
        "uploaded": uploaded,
        "failed": failed,
    }
    (state_root / "batch_complete.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
