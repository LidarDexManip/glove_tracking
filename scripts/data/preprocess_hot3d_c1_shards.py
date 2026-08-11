#!/usr/bin/env python3
"""Offline HOT3D tar -> canonical C1 target/MANO/mask shards.

The command is deterministic and resumable at shard granularity. It never
removes source archives. Only frames with a non-empty sufficiently large right
MANO raster in the verified 1201-2/-90-degree C1 camera enter sample manifests;
every rejection is retained in availability.jsonl with an explicit reason.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from hand_tracking_toolkit.dataset import warp_image
from hot3d.hot3d.clips import clip_util

from scripts.data.export_hot3d_clip_undistorted import build_canonical_camera
from hand_restoration.hot3d_dataset import Hot3DSingleFrameDataset, _as_rgb
from hot3d_glove_torch_utils import load_mano_model_torch


def encode(image: np.ndarray, extension: str, params: list[int] | None = None) -> bytes:
    ok, data = cv2.imencode(extension, image, params or [])
    if not ok:
        raise RuntimeError(f"Failed to encode {extension} image")
    return data.tobytes()


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records), encoding="utf-8")


def validate_completed_shard(tar_path: Path, samples_path: Path, availability_path: Path) -> bool:
    if not (tar_path.is_file() and samples_path.is_file() and availability_path.is_file()):
        return False
    try:
        with tarfile.open(tar_path, "r") as tar:
            names = set(tar.getnames())
        return all(f"{record['key']}.target.jpg" in names for record in read_jsonl(samples_path))
    except (tarfile.TarError, OSError, json.JSONDecodeError):
        return False


def process_clip(
    clip_path: Path,
    metadata: dict,
    output_tar: tarfile.TarFile,
    renderer: Hot3DSingleFrameDataset,
    camera_id: str,
    min_mask_pixels: int,
    jpeg_quality: int,
) -> tuple[list[dict], list[dict]]:
    samples: list[dict] = []
    availability: list[dict] = []
    with tarfile.open(clip_path, "r") as source_tar:
        names = set(source_tar.getnames())
        if "__hand_shapes.json__" not in names:
            return [], [{"clip_id": clip_path.stem, "frame_id": None, "status": "missing_hand_shapes"}]
        shapes = json.load(source_tar.extractfile("__hand_shapes.json__"))
        betas = torch.tensor(shapes["mano"], dtype=torch.float32).view(1, -1)
        keys = sorted(name.removesuffix(".info.json") for name in names if name.endswith(".info.json"))
        for frame_key in keys:
            status = {"clip_id": clip_path.stem, "frame_id": frame_key}
            hands = clip_util.load_hand_annotations(source_tar, frame_key)
            if hands is None or "right" not in hands or "mano_pose" not in hands["right"]:
                availability.append({**status, "status": "missing_right_mano"})
                continue
            try:
                cameras, _ = clip_util.load_cameras(source_tar, frame_key)
                if camera_id not in cameras:
                    availability.append({**status, "status": "missing_camera"})
                    continue
                src_camera = cameras[camera_id]
                c1_camera = build_canonical_camera(src_camera, -90.0)
                source = _as_rgb(clip_util.load_image(source_tar, frame_key, camera_id))
                target = warp_image(src_camera=src_camera, dst_camera=c1_camera, src_image=source).astype(np.uint8)
                mano_rgb, mano_mask = renderer._render_mano(hands, betas, c1_camera)
            except RuntimeError as exc:
                reason = "mano_outside_c1" if "empty mask" in str(exc) else "render_error"
                availability.append({**status, "status": reason, "error": str(exc)})
                continue
            except (KeyError, ValueError, OSError) as exc:
                availability.append({**status, "status": "decode_or_metadata_error", "error": str(exc)})
                continue
            mask_pixels = int(np.count_nonzero(mano_mask))
            if mask_pixels < min_mask_pixels:
                availability.append({**status, "status": "mask_too_small", "mask_pixels": mask_pixels})
                continue

            target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
            mano_gray = cv2.cvtColor((mano_rgb * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            mask_u8 = (mano_mask > 0).astype(np.uint8) * 255
            key = f"{clip_path.stem}_{frame_key}"
            record = {
                "key": key,
                "clip_id": clip_path.stem,
                "frame_id": frame_key,
                "participant_id": metadata.get("participant_id", ""),
                "source_sequence_id": metadata.get("sequence_id", ""),
                "camera_id": camera_id,
                "handedness": "right",
                "canonical_image_size": [int(c1_camera.height), int(c1_camera.width)],
                "mask_pixels_canonical": mask_pixels,
                "source_tar": str(clip_path),
            }
            add_bytes(output_tar, f"{key}.target.jpg", encode(target_gray, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]))
            add_bytes(output_tar, f"{key}.mano.png", encode(mano_gray, ".png"))
            add_bytes(output_tar, f"{key}.mask.png", encode(mask_u8, ".png"))
            add_bytes(output_tar, f"{key}.json", (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8"))
            samples.append(record)
            availability.append({**status, "status": "usable", "mask_pixels": mask_pixels})
    return samples, availability


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "holdout"), default=("train", "holdout"))
    parser.add_argument("--clips-per-shard", type=int, default=8)
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-clips", type=int, default=None, help="Smoke/debug limit per split.")
    parser.add_argument("--force", action="store_true", help="Rebuild completed shards.")
    args = parser.parse_args()
    if args.clips_per_shard < 1 or args.min_mask_pixels < 1:
        raise ValueError("clips-per-shard and min-mask-pixels must be positive")

    split_manifest = json.loads(args.split_json.read_text(encoding="utf-8"))
    train_ids = {Path(path).stem for path in split_manifest.get("train", [])}
    holdout_ids = {Path(path).stem for path in split_manifest.get("holdout", [])}
    if train_ids & holdout_ids:
        raise ValueError(f"Clip leakage in split manifest: {sorted(train_ids & holdout_ids)[:20]}")
    clip_metadata = split_manifest.get("clip_metadata", {})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = Hot3DSingleFrameDataset.__new__(Hot3DSingleFrameDataset)
    renderer.hands = ["right"]
    renderer.models = {"right": load_mano_model_torch("right")}
    all_summary = {"schema_version": 1, "source_split": str(args.split_json), "camera_id": "1201-2", "hands": "right", "canonical_roll_degrees": -90.0, "min_mask_pixels": args.min_mask_pixels, "jpeg_quality": args.jpeg_quality, "splits": {}}

    for split_name in args.splits:
        clip_paths = [Path(path) for path in split_manifest[split_name]]
        if args.max_clips is not None:
            clip_paths = clip_paths[: args.max_clips]
        missing = [str(path) for path in clip_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} source clips; first entries: {missing[:10]}")
        split_dir = args.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        sample_records: list[dict] = []
        availability_records: list[dict] = []
        checksums = {}
        groups = [clip_paths[index:index + args.clips_per_shard] for index in range(0, len(clip_paths), args.clips_per_shard)]
        for shard_index, group in enumerate(groups):
            stem = f"{split_name}-{shard_index:05d}"
            tar_path = split_dir / f"{stem}.tar"
            samples_path = split_dir / f"{stem}.samples.jsonl"
            availability_path = split_dir / f"{stem}.availability.jsonl"
            if not args.force and validate_completed_shard(tar_path, samples_path, availability_path):
                shard_samples = read_jsonl(samples_path)
                shard_availability = read_jsonl(availability_path)
                print(f"SKIP complete {tar_path} samples={len(shard_samples)}")
            else:
                temporary = split_dir / f".{stem}.tar.inprogress"
                shard_samples, shard_availability = [], []
                with tarfile.open(temporary, "w") as output_tar:
                    for clip_path in group:
                        print(f"PROCESS {split_name} {clip_path.name}", flush=True)
                        samples, availability = process_clip(clip_path, clip_metadata.get(clip_path.stem, {}), output_tar, renderer, "1201-2", args.min_mask_pixels, args.jpeg_quality)
                        shard_samples.extend(samples)
                        shard_availability.extend(availability)
                temporary.replace(tar_path)
                relative_shard = str(tar_path.relative_to(args.output_dir))
                for record in shard_samples:
                    record["shard"] = relative_shard
                write_jsonl(samples_path, shard_samples)
                write_jsonl(availability_path, shard_availability)
            sample_records.extend(shard_samples)
            availability_records.extend(shard_availability)
            checksums[str(tar_path.relative_to(args.output_dir))] = sha256(tar_path)
        write_jsonl(args.output_dir / f"{split_name}_manifest.jsonl", sample_records)
        write_jsonl(args.output_dir / f"{split_name}_availability.jsonl", availability_records)
        statuses = Counter(record["status"] for record in availability_records)
        all_summary["splits"][split_name] = {"source_clips": len(clip_paths), "shards": len(groups), "usable_samples": len(sample_records), "availability": dict(sorted(statuses.items())), "sha256": checksums}
        print(f"DONE {split_name}: clips={len(clip_paths)} usable={len(sample_records)} statuses={dict(statuses)}")

    (args.output_dir / "dataset_summary.json").write_text(json.dumps(all_summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote derived dataset to {args.output_dir}")
    print("Source archives were NOT deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
