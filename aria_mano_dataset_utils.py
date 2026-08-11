"""Shared utilities for the Aria corrected-MANO + SAM dataset pipeline."""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_HF_REPO = "LIDAR-GT/HUGG_ARIA"
DEFAULT_HF_REVISION = "fa144788084ba3eca30b753b50b26e31001c11f2"
ARIA_STREAM_ID = "214-1"


def configure_imports() -> None:
    """Expose the vendored HOT3D and Project Aria dependencies."""
    for path in (ROOT / "hot3d/hot3d",):
        if not path.is_dir():
            raise FileNotFoundError(path)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    active_prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix)).resolve()
    envs_dir = active_prefix.parent if active_prefix.parent.name == "envs" else active_prefix / "envs"
    for aria_site in (envs_dir / "glove2hand/lib").glob("python*/site-packages"):
        if aria_site.is_dir() and str(aria_site) not in sys.path:
            sys.path.append(str(aria_site))


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_rgb(image: np.ndarray, extension: str, params: list[int] | None = None) -> bytes:
    bgr = cv2.cvtColor(np.asarray(image, np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(extension, bgr, params or [])
    if not ok:
        raise RuntimeError(f"Failed to encode RGB {extension}")
    return encoded.tobytes()


def encode_mask(mask: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", np.asarray(mask, np.uint8))
    if not ok:
        raise RuntimeError("Failed to encode mask")
    return encoded.tobytes()


def add_bytes(archive: tarfile.TarFile, member: str, data: bytes) -> None:
    info = tarfile.TarInfo(member)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray
    t: np.ndarray

    def project(self, points: np.ndarray) -> np.ndarray:
        camera_points = np.asarray(points) @ self.R.T + self.t
        depth = camera_points[:, 2]
        return np.column_stack(
            (self.fx * camera_points[:, 0] / depth + self.cx,
             self.fy * camera_points[:, 1] / depth + self.cy)
        )


def resized_camera(camera, size: int):

    sx, sy = size / camera.width, size / camera.height
    return Camera(
        size,
        size,
        camera.fx * sx,
        camera.fy * sy,
        camera.cx * sx,
        camera.cy * sy,
        R=np.asarray(camera.R),
        t=np.asarray(camera.t),
    )


def mano_vertices(model, params, frame: int) -> np.ndarray:
    vertices, _ = model.forward(
        params.pose[frame],
        betas=params.betas,
        trans=params.trans[frame],
        use_pca=params.use_pca,
    )
    return vertices.astype(np.float32)


def load_corrected_mano_models():
    """Load HOT3D's MANO model, including its left-shapedirs sign fix."""
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    configure_imports()
    from data_loaders.mano_layer import MANOHandModel

    return MANOHandModel(str(ROOT / "mano_v1_2/models"))


def rasterize_meshes(
    meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]], camera
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize solid, lightly shaded MANO meshes for the condition image."""
    canvas = np.zeros((camera.height, camera.width, 3), np.uint8)
    mask = np.zeros((camera.height, camera.width), np.uint8)
    triangles = []
    light = np.array([0.25, -0.35, -0.90], np.float32)
    light /= np.linalg.norm(light)
    for vertices, faces, base_color in meshes:
        uv = camera.project(vertices)
        camera_vertices = vertices @ camera.R.T + camera.t
        for face in np.asarray(faces, np.int64):
            tri3 = vertices[face]
            z = camera_vertices[face, 2]
            if np.any(z <= 1e-4):
                continue
            tri2 = uv[face]
            if (
                tri2[:, 0].max() < 0
                or tri2[:, 1].max() < 0
                or tri2[:, 0].min() >= camera.width
                or tri2[:, 1].min() >= camera.height
            ):
                continue
            normal = np.cross(tri3[1] - tri3[0], tri3[2] - tri3[0])
            normal /= np.linalg.norm(normal) + 1e-8
            shade = 0.60 + 0.40 * abs(float(normal @ light))
            color = np.clip(base_color * shade, 0, 255).astype(np.uint8)
            triangles.append((float(z.mean()), np.round(tri2).astype(np.int32), color))
    for _, tri, color in sorted(triangles, key=lambda item: item[0], reverse=True):
        cv2.fillConvexPoly(canvas, tri, color.tolist(), lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, tri, 255, lineType=cv2.LINE_8)
    return canvas, mask > 0


def prompts(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    result = []
    height, width = mask.shape
    for index in range(1, count):
        x, y, box_width, box_height, area = stats[index]
        if area < 64:
            continue
        pad = max(8, round(max(box_width, box_height) * 0.08))
        box = np.array(
            [
                max(0, x - pad),
                max(0, y - pad),
                min(width - 1, x + box_width + pad),
                min(height - 1, y + box_height + pad),
            ]
        )
        result.append((box, np.asarray([centers[index]], np.float32)))
    return result


def selected_work(split: dict, args) -> list[tuple[str, str]]:
    membership = {
        sequence: split_name
        for split_name in ("train", "holdout")
        for sequence in split[split_name]
    }
    if args.sequence:
        missing = sorted(set(args.sequence) - set(membership))
        if missing:
            raise ValueError(f"Sequences are absent from split: {missing}")
        work = [(membership[sequence], sequence) for sequence in args.sequence]
    else:
        work = [
            (split_name, sequence)
            for split_name in args.splits
            for sequence in split[split_name]
        ]
    if args.max_sequences is not None:
        work = work[: args.max_sequences]
    return work
