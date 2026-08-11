from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from hand_tracking_toolkit import rasterizer
from hand_tracking_toolkit.dataset import warp_image
from hot3d.hot3d.clips import clip_util

from scripts.data.export_hot3d_clip_undistorted import build_canonical_camera
from hot3d_glove_torch_utils import load_mano_model_torch

from .conditions import ConditionBuilder, ConditionConfig


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4:
        return image[..., :3]
    return image


class Hot3DSingleFrameDataset(Dataset):
    """HOT3D clip-tar adapter in the verified upright canonical C1 convention.

    Returned image tensors are ``float32 [3,H,W]`` in ``[-1,1]``. Masks are
    ``float32 [1,H,W]`` in ``[0,1]``. ``*_np`` variants stay HWC/HW in [0,1]
    for debug visualization and are intentionally omitted by the train loader.
    """

    def __init__(
        self,
        clip_tars: Iterable[str | Path],
        mano_model_dir: str | Path,
        camera_id: str = "1201-2",
        hands: str = "right",
        output_size: int = 512,
        grayscale: bool = True,
        frame_start: int = 0,
        frame_stride: int = 1,
        max_frames_per_clip: int | None = None,
        condition: ConditionConfig | None = None,
        seed: int = 0,
        include_numpy: bool = True,
        require_mano_in_frame: bool = False,
        min_visible_mano_vertices: int = 1,
    ) -> None:
        if hands not in {"left", "right", "both"}:
            raise ValueError("hands must be left, right, or both")
        self.clip_tars = [Path(path) for path in clip_tars]
        if not self.clip_tars:
            raise ValueError("At least one HOT3D clip tar is required.")
        missing = [str(path) for path in self.clip_tars if not path.exists()]
        if missing:
            raise FileNotFoundError(f"HOT3D clip tar(s) not found: {missing}")
        self.camera_id = camera_id
        self.hands = ["left", "right"] if hands == "both" else [hands]
        self.output_size = int(output_size)
        self.grayscale = bool(grayscale)
        self.frame_start = max(0, int(frame_start))
        self.include_numpy = include_numpy
        self.require_mano_in_frame = bool(require_mano_in_frame)
        self.min_visible_mano_vertices = max(1, int(min_visible_mano_vertices))
        self.condition_builder = ConditionBuilder(condition or ConditionConfig(), seed=seed)
        self.models = {hand: load_mano_model_torch(hand) for hand in self.hands}
        self.samples: list[tuple[Path, str]] = []
        for clip_path in self.clip_tars:
            self.samples.extend(self._index_clip(clip_path, frame_stride, max_frames_per_clip))
        if not self.samples:
            raise RuntimeError("No frames with requested MANO annotations were found.")

    def _index_clip(self, path: Path, stride: int, limit: int | None) -> list[tuple[Path, str]]:
        selected: list[tuple[Path, str]] = []
        with tarfile.open(path, "r") as tar:
            betas = None
            if self.require_mano_in_frame:
                shapes = json.load(tar.extractfile("__hand_shapes.json__"))
                betas = torch.tensor(shapes["mano"], dtype=torch.float32).view(1, -1)
            keys = sorted(name.split(".info.json")[0] for name in tar.getnames() if name.endswith(".info.json"))
            keys = [key for key in keys if int(key) >= self.frame_start]
            for key in keys[::max(1, stride)]:
                hands = clip_util.load_hand_annotations(tar, key)
                if hands is not None and all(hand in hands and "mano_pose" in hands[hand] for hand in self.hands):
                    if self.require_mano_in_frame:
                        cameras, _ = clip_util.load_cameras(tar, key)
                        if self.camera_id not in cameras:
                            continue
                        c1_camera = build_canonical_camera(cameras[self.camera_id], -90.0)
                        try:
                            self._render_mano(hands, betas, c1_camera)
                        except RuntimeError as exc:
                            if "empty mask" not in str(exc):
                                raise
                            continue
                    selected.append((path, key))
                    if limit is not None and len(selected) >= limit:
                        break
        return selected

    @staticmethod
    def _resize(image: np.ndarray, size: int, interpolation: int) -> np.ndarray:
        return cv2.resize(image, (size, size), interpolation=interpolation)

    def _render_mano(self, hands_data: dict, betas: torch.Tensor, camera) -> tuple[np.ndarray, np.ndarray]:
        height, width = camera.height, camera.width
        render = np.zeros((height, width, 3), dtype=np.float32)
        combined_mask = np.zeros((height, width), dtype=np.float32)
        for hand in self.hands:
            pose = hands_data[hand]["mano_pose"]
            output = self.models[hand](
                betas=betas,
                global_orient=torch.tensor(pose["wrist_xform"][:3], dtype=torch.float32).view(1, 3),
                hand_pose=torch.tensor(pose["thetas"], dtype=torch.float32).view(1, -1),
                transl=torch.tensor(pose["wrist_xform"][3:], dtype=torch.float32).view(1, 3),
                return_verts=True,
            )
            vertices = output.vertices[0].detach().cpu().numpy()
            # Keyword arguments avoid toolkit-version differences in positional
            # argument ordering (verts/faces/normals/camera).
            shaded, mask, _ = rasterizer.rasterize_mesh(
                verts=vertices,
                faces=self.models[hand].faces.astype(np.int64),
                vert_normals=None,
                camera=camera,
                ambient=(0.22, 0.22, 0.22),
                diffuse=(0.58, 0.58, 0.58),
                specular=(0.08, 0.08, 0.08),
                shininess=24,
            )
            mask = mask.astype(bool)
            shaded = shaded.astype(np.float32) / 255.0
            render[mask] = shaded[mask]
            combined_mask[mask] = 1.0
        if not np.any(combined_mask):
            raise RuntimeError("MANO renderer produced an empty mask; check C1 camera alignment and annotations.")
        return render, combined_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        clip_path, frame_key = self.samples[index]
        with tarfile.open(clip_path, "r") as tar:
            cameras, _ = clip_util.load_cameras(tar, frame_key)
            if self.camera_id not in cameras:
                raise KeyError(f"Camera {self.camera_id} is absent from {clip_path.name}:{frame_key}")
            src_camera = cameras[self.camera_id]
            c1_camera = build_canonical_camera(src_camera, -90.0)
            source = _as_rgb(clip_util.load_image(tar, frame_key, self.camera_id))
            target = warp_image(src_camera=src_camera, dst_camera=c1_camera, src_image=source).astype(np.uint8)
            annotations = clip_util.load_hand_annotations(tar, frame_key)
            shapes = json.load(tar.extractfile("__hand_shapes.json__"))
            betas = torch.tensor(shapes["mano"], dtype=torch.float32).view(1, -1)
            mano_rgb, mano_mask = self._render_mano(annotations, betas, c1_camera)

        target = self._resize(target, self.output_size, cv2.INTER_AREA).astype(np.float32) / 255.0
        mano_rgb = self._resize(mano_rgb, self.output_size, cv2.INTER_LINEAR)
        mano_mask = self._resize(mano_mask, self.output_size, cv2.INTER_NEAREST)
        if self.grayscale:
            target_luma = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
            target = np.repeat(target_luma[..., None], 3, axis=2)
            mano_luma = cv2.cvtColor(mano_rgb, cv2.COLOR_RGB2GRAY)
            mano_rgb = np.repeat(mano_luma[..., None], 3, axis=2)
        condition, edit_mask = self.condition_builder(target, mano_rgb, mano_mask)
        if target.shape != condition.shape or target.shape[:2] != edit_mask.shape:
            raise AssertionError("Image/mask resize mismatch.")
        if not (0.0 <= target.min() <= target.max() <= 1.0 and 0.0 <= condition.min() <= condition.max() <= 1.0):
            raise AssertionError("Images must remain in [0,1].")

        result = {
            "target_rgb": torch.from_numpy(target.transpose(2, 0, 1) * 2.0 - 1.0),
            "mano_rgb": torch.from_numpy(mano_rgb.transpose(2, 0, 1) * 2.0 - 1.0),
            "mano_mask": torch.from_numpy(mano_mask[None].astype(np.float32)),
            "condition_rgb": torch.from_numpy(condition.transpose(2, 0, 1) * 2.0 - 1.0),
            "edit_mask": torch.from_numpy(edit_mask[None].astype(np.float32)),
            "metadata": {"sequence_id": clip_path.stem, "frame_id": frame_key, "camera_id": self.camera_id, "handedness": "+".join(self.hands), "original_image_size": [int(c1_camera.height), int(c1_camera.width)]},
        }
        if self.include_numpy:
            result.update({"target_rgb_np": target, "mano_rgb_np": mano_rgb, "mano_mask_np": mano_mask, "condition_rgb_np": condition, "edit_mask_np": edit_mask})
        return result
