#!/usr/bin/env python3
"""Convert compact RGB + full-pose MANO sessions into Gaussian Hand crops."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gaussian_hand.data.geometry import (
    crop_intrinsics,
    project_points,
    rasterize_silhouette,
    square_crop,
)
from gaussian_hand.model.mano import LearnableMANO
from gaussian_hand.model.pose_refinement import axis_angle_to_quaternion


ROOT = Path(__file__).resolve().parents[1]


def camera_matrix(metadata: dict) -> np.ndarray:
    camera = metadata["camera"]
    return np.asarray([
        [camera["fx"], 0.0, camera["cx"]],
        [0.0, camera["fy"], camera["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def choose_indices(count: int, limit: int) -> np.ndarray:
    if limit and count > limit:
        return np.unique(np.linspace(0, count - 1, limit).round().astype(int))
    return np.arange(count)


def subject_betas(sessions: list[Path], side: str) -> tuple[np.ndarray, dict]:
    values = {}
    for session in sessions:
        metadata = json.loads((session / "metadata.json").read_text())
        if side not in metadata["hands_present"]:
            continue
        with np.load(session / "mano.npz", allow_pickle=False) as data:
            values[session.name] = data[f"{side}_betas"].astype(np.float32)
    if not values:
        raise ValueError(f"No sessions contain a {side} hand")
    canonical = np.mean(np.stack(list(values.values())), axis=0).astype(np.float32)
    deviations = {
        name: float(np.max(np.abs(value - canonical)))
        for name, value in values.items()
    }
    return canonical, deviations


@torch.no_grad()
def vertices_for_session(
    model: LearnableMANO, data: np.lib.npyio.NpzFile, side: str,
    indices: np.ndarray, canonical_betas: np.ndarray, chunk_size: int = 256,
) -> np.ndarray:
    parts = []
    betas = torch.from_numpy(canonical_betas).float()
    for begin in range(0, len(indices), chunk_size):
        selected = indices[begin:begin + chunk_size]
        pose = torch.from_numpy(data[f"{side}_hand_pose"][selected]).float()
        orient = torch.from_numpy(data[f"{side}_global_orient"][selected]).float()
        quaternion = axis_angle_to_quaternion(orient)
        translation = torch.from_numpy(data[f"{side}_transl"][selected]).float()
        batch_betas = betas.expand(len(selected), -1)
        parts.append(model(pose, quaternion, translation, batch_betas).numpy())
    return np.concatenate(parts) if parts else np.empty((0, 778, 3), np.float32)


def make_contact_sheet(
    records: list[dict], root: Path, output: Path, count: int,
) -> None:
    selected = np.linspace(
        0, len(records) - 1, min(count, len(records))).round().astype(int)
    tiles = []
    for index in selected:
        record = records[int(index)]
        image = cv2.imread(str(root / record["image"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(root / record["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(record["image"])
        tile = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
        tile_mask = cv2.resize(
            mask, (192, 192), interpolation=cv2.INTER_NEAREST)
        tile[0 == tile_mask] = (
            tile[0 == tile_mask].astype(np.float32) * 0.2).astype(np.uint8)
        cv2.putText(
            tile, f"{record['sequence'][-11:]} {record['frame_index']}",
            (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 255, 0), 1,
            cv2.LINE_AA)
        tiles.append(tile)
    columns = 12
    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank)
    rows = [np.concatenate(tiles[i:i + columns], axis=1)
            for i in range(0, len(tiles), columns)]
    if not cv2.imwrite(
        str(output), np.concatenate(rows, axis=0),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    ):
        raise RuntimeError(f"Failed to write {output}")


