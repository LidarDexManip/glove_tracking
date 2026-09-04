"""Small, testable geometry routines used by the HOT3D audit."""
from __future__ import annotations

import cv2
import numpy as np


def project_points(vertices: np.ndarray, world_to_camera: np.ndarray,
                   intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space vertices with OpenCV pinhole conventions."""
    vertices = np.asarray(vertices, np.float64)
    camera = vertices @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera[:, 2]
    uvw = camera @ intrinsics.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)
    return uv, z


def rasterize_silhouette(uv: np.ndarray, z: np.ndarray, faces: np.ndarray,
                         width: int, height: int, downsample: int = 4) -> np.ndarray:
    """Rasterize a conservative binary MANO silhouette for data QA."""
    if downsample < 1:
        raise ValueError("downsample must be positive")
    faces = np.asarray(faces)
    valid_faces = faces[np.all(z[faces] > 0.01, axis=1)]
    mask = np.zeros(((height + downsample - 1) // downsample,
                     (width + downsample - 1) // downsample), np.uint8)
    if len(valid_faces):
        polygons = np.rint(uv[valid_faces] / downsample).astype(np.int32)
        cv2.fillPoly(mask, polygons, 1)
    return mask


def visible_mano_ratio(sam_mask: np.ndarray, mano_mask: np.ndarray) -> float:
    """Fraction of the complete MANO silhouette explained by the visible SAM hand."""
    mano = np.asarray(mano_mask, bool)
    area = int(mano.sum())
    if area == 0:
        return 0.0
    sam = np.asarray(sam_mask, bool)
    if sam.shape != mano.shape:
        sam = cv2.resize(sam.astype(np.uint8), (mano.shape[1], mano.shape[0]),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    return float(np.logical_and(sam, mano).sum() / area)


def square_crop(uv: np.ndarray, width: int, height: int,
                scale: float = 1.35) -> tuple[int, int, int, int] | None:
    """Return an in-image square xyxy crop enclosing projected MANO points."""
    points = np.asarray(uv, np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return None
    lo, hi = points.min(axis=0), points.max(axis=0)
    side = max(float((hi - lo).max()) * scale, 2.0)
    side = min(side, float(min(width, height)))
    center = (lo + hi) / 2.0
    x0 = float(np.clip(center[0] - side / 2, 0, width - side))
    y0 = float(np.clip(center[1] - side / 2, 0, height - side))
    x1, y1 = int(np.floor(x0)), int(np.floor(y0))
    x2, y2 = int(np.ceil(x0 + side)), int(np.ceil(y0 + side))
    return x1, y1, min(x2, width), min(y2, height)


def fraction_inside_box(uv: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    inside = ((uv[:, 0] >= x0) & (uv[:, 0] < x1) &
              (uv[:, 1] >= y0) & (uv[:, 1] < y1))
    return float(inside.mean())


def crop_intrinsics(intrinsics: np.ndarray, box: tuple[int, int, int, int],
                    output_size: int) -> np.ndarray:
    """Update K for crop+resize, preserving OpenCV pixel-centre semantics."""
    x0, y0, x1, y1 = box
    sx, sy = output_size / (x1 - x0), output_size / (y1 - y0)
    result = np.asarray(intrinsics, np.float64).copy()
    result[0, 0] *= sx
    result[1, 1] *= sy
    result[0, 2] = (result[0, 2] - x0 + 0.5) * sx - 0.5
    result[1, 2] = (result[1, 2] - y0 + 0.5) * sy - 0.5
    return result
