#!/usr/bin/env python3
"""Decode audited HOT3D frames into subject-bound RGB/mask crops."""
from __future__ import annotations

import argparse
import json
import sqlite3
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from gaussian_hand.data.geometry import project_points, rasterize_silhouette
from gaussian_hand.model.mano import LearnableMANO

ROOT = Path(__file__).resolve().parents[2]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--contact-count", type=int, default=96)
    return parser.parse_args()


def split_for(sequence: str, policy: dict) -> str:
    for split, key in (("train", "train_sequences"), ("val", "validation_sequences"),
                       ("test", "test_sequences")):
        if sequence in policy[key]:
            return split
    raise ValueError(f"Sequence absent from split policy: {sequence}")


def sharpness(image: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = cv2.Laplacian(gray, cv2.CV_32F)[mask > 0]
    return float(values.var()) if len(values) else 0.0


def make_sheet(examples: list[tuple[np.ndarray, np.ndarray, dict]], output: Path,
               count: int) -> None:
    if not examples:
        return
    indices = np.linspace(0, len(examples) - 1, min(count, len(examples))).astype(int)
    tiles = []
    for index in indices:
        image, mask, record = examples[index]
        display = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, (192, 192), interpolation=cv2.INTER_NEAREST)
        overlay = display.copy()
        overlay[small_mask == 0] = (overlay[small_mask == 0] * 0.28).astype(np.uint8)
        cv2.putText(overlay, f"{record['sequence'][-8:]} {record['frame_index']}",
                    (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(overlay, f"vis {record['visible_mano_ratio']:.2f}",
                    (4, 187), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 1,
                    cv2.LINE_AA)
        tiles.append(overlay)
    columns = 12
    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank)
    rows = [np.concatenate(tiles[i:i + columns], axis=1)
            for i in range(0, len(tiles), columns)]
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.concatenate(rows, axis=0)):
        raise RuntimeError(f"Failed to write {output}")


def main() -> None:
    args = arguments()
    cv2.setNumThreads(1)
    config = json.loads(args.config.read_text())
    handedness = config["binding"]["handedness"]
    mano = LearnableMANO(
        str(ROOT / "mano_v1_2/models" / f"MANO_{handedness.upper()}.pkl"),
        handedness=handedness)
    sam_label = 2 if handedness == "right" else 1
    records = [json.loads(line) for line in args.manifest.open() if line.strip()]
    if args.limit:
        indices = np.linspace(0, len(records) - 1, min(args.limit, len(records))).astype(int)
        records = [records[i] for i in indices]
    grouped = defaultdict(list)
    for record in records:
        grouped[record["sequence"]].append(record)

    output = args.output
    image_dir, mask_dir = output / "images", output / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    exported, examples = [], []
    # Keep only tiny contact-sheet thumbnails in memory, not every 512px crop.
    visual_indices = set(np.linspace(0, max(len(records)-1,0), min(args.contact_count, len(records))).astype(int))
    rejected_photo = 0
    for sequence, sequence_records in sorted(grouped.items()):
        source = ROOT / config["pinhole_root"] / sequence
        video = cv2.VideoCapture(str(source / "rgb_214_1_pinhole.mp4"))
        if not video.isOpened():
            raise RuntimeError(f"Cannot open {sequence} video")
        database_path = ROOT / "outputs/sam2_hugg_aria_masks_v3_pilot" / sequence / "masks.sqlite"
        database = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        metadata = {
            key: json.loads(value)
            for key, value in database.execute("SELECT key,value_json FROM metadata")
        }
        height, width = int(metadata["height"]), int(metadata["width"])
        by_frame = {item["frame_index"]: item for item in sequence_records}
        last = max(by_frame)
        try:
            for frame_index in range(last + 1):
                ok = video.grab()
                if not ok:
                    raise RuntimeError(f"Decode stopped at {sequence}:{frame_index}")
                record = by_frame.get(frame_index)
                if record is None:
                    continue
                ok, frame = video.retrieve()
                if not ok:
                    raise RuntimeError("Frame retrieve failed")
                row = database.execute(
                    "SELECT labels_zlib FROM frames WHERE frame_index=?", (frame_index,)
                ).fetchone()
                if row is None:
                    raise RuntimeError(f"Missing mask {sequence}:{frame_index}")
                labels = np.frombuffer(zlib.decompress(row[0]), np.uint8).reshape(height, width)
                x0, y0, x1, y1 = record["crop_xyxy"]
                size = int(record["output_size"])
                crop = cv2.resize(frame[y0:y1, x0:x1], (size, size),
                                  interpolation=cv2.INTER_LANCZOS4)
                mask = cv2.resize((labels[y0:y1, x0:x1] == sam_label).astype(np.uint8) * 255,
                                  (size, size), interpolation=cv2.INTER_NEAREST)
                if record["handedness"] != handedness:
                    raise ValueError("Manifest hand mismatch")
                with torch.no_grad():
                    verts = mano(torch.tensor(record["mano_pca"]), torch.tensor(record["wrist_q_wxyz"]),
                                 torch.tensor(record["wrist_t_world"]), torch.tensor(record["mano_betas"]))[0].numpy()
                uv,z = project_points(verts, np.asarray(record["world_to_camera"]), np.asarray(record["crop_intrinsics"]))
                support = rasterize_silhouette(uv,z,mano.faces.numpy(),size,size,1)*255
                reference = cv2.bitwise_and(mask,support)
                gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
                pixels = gray[reference>0]
                quality = sharpness(crop,reference)
                photo = config.get("photometric_filter",{})
                if (not len(pixels) or quality < photo.get("min_laplacian_variance",0)
                    or ((pixels<8)|(pixels>247)).mean() > photo.get("max_clipped_fraction",1.0)):
                    rejected_photo += 1
                    continue
                stem = f"{sequence}_{frame_index:06d}"
                image_path, mask_path = image_dir / f"{stem}.jpg", mask_dir / f"{stem}.png"
                if not cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Failed to write {image_path}")
                if not cv2.imwrite(str(mask_path), mask):
                    raise RuntimeError(f"Failed to write {mask_path}")
                item = dict(record)
                item.update({
                    "split": split_for(sequence, config["split_policy"]),
                    "image": str(image_path.relative_to(output)),
                    "mask": str(mask_path.relative_to(output)),
                    "sharpness": quality,
                    "mesh_mask": f"mesh_masks/{stem}.png",
                    "reference_mask": f"reference_masks/{stem}.png",
                    "mean_luma": float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[mask > 0].mean()),
                })
                for folder, values in (("mesh_masks",support),("reference_masks",reference)):
                    (output/folder).mkdir(exist_ok=True)
                    if not cv2.imwrite(str(output/folder/f"{stem}.png"),values):
                        raise RuntimeError("Mask write failed")
                if len(exported) in visual_indices:
                    examples.append((cv2.resize(crop,(192,192)),cv2.resize(reference,(192,192)),item))
                exported.append(item)
        finally:
            video.release()
            database.close()
        print(json.dumps({"sequence": sequence, "exported": len(sequence_records)}),
              flush=True)

    manifest = output / "frames.jsonl"
    with manifest.open("w") as handle:
        for item in exported:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    if not exported:
        raise ValueError("No usable frames after photometric filtering")
    stats = {
        "binding": config["binding"],
        "frames": len(exported),
        "photometric_rejections": rejected_photo,
        "splits": {name: sum(item["split"] == name for item in exported)
                   for name in ("train", "val", "test")},
        "sharpness_percentiles": dict(zip(
            ("0", "1", "5", "50", "95", "100"),
            map(float, np.percentile([item["sharpness"] for item in exported],
                                    (0, 1, 5, 50, 95, 100))))),
        "mean_luma_percentiles": dict(zip(
            ("0", "1", "5", "50", "95", "100"),
            map(float, np.percentile([item["mean_luma"] for item in exported],
                                    (0, 1, 5, 50, 95, 100))))),
    }
    (output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    sheet = args.contact_sheet or output / "contact_sheet.jpg"
    make_sheet(examples, sheet, args.contact_count)
    print(json.dumps({**stats, "manifest": str(manifest), "contact_sheet": str(sheet)},
                     indent=2))


if __name__ == "__main__":
    main()
