from __future__ import annotations

import json
import sqlite3
import zlib
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .hugg_aria_conditioning import (
    CropBox,
    build_faithful_condition,
    crop_resize,
    map_crop_box,
    paste_crop_with_mask,
    square_crop_from_mask,
)


def overlay_and_loss_masks(
    render_foreground: np.ndarray,
    sam_mask: np.ndarray | None,
    loss_mask_source: str,
    mano_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if loss_mask_source not in {"overlay", "sam", "mano"}:
        raise ValueError("loss_mask_source must be overlay, sam, or mano")
    overlay_mask = (
        render_foreground
        if sam_mask is None
        else render_foreground & sam_mask
    )
    if loss_mask_source == "mano":
        if mano_mask is None:
            raise ValueError("mano_mask is required when loss_mask_source=mano")
        loss_mask = mano_mask
    elif loss_mask_source == "sam" and sam_mask is not None:
        loss_mask = sam_mask
    else:
        loss_mask = overlay_mask
    return overlay_mask, loss_mask


class HuggAriaOverlayDataset(Dataset):
    """Read frame-aligned RGB, render RGB/alpha, and SAM labels on demand."""

    def __init__(
        self,
        manifest: str | Path,
        pinhole_root: str | Path,
        render_root: str | Path,
        mask_root: str | Path,
        output_size: int = 512,
        condition_variant: str = "sam_mask",
        render_opacity: float = 1.0,
        render_kind: str = "gaussian",
        render_alpha_filename: str = "alpha.mkv",
        alpha_threshold: float = 0.0,
        loss_mask_source: str = "overlay",
        loss_mask_root: str | Path | None = None,
        loss_mask_alpha_filename: str = "alpha.mkv",
        spatial_mode: str = "full_frame",
        crop_scale: float = 1.2,
        condition_style: str = "legacy_overlay",
        hand_mask_dilation_px: int = 8,
        wrist_mask_enabled: bool = True,
        wrist_geometry_source: str = "silhouette_pca",
        wrist_ring_root: str | Path | None = None,
        wrist_length_ratio: float = 0.10,
        wrist_width_scale: float = 1.10,
        wrist_ring_transverse_scale: float = 1.30,
        wrist_sleeve_orientation: str = "ring_min_area",
        wrist_sleeve_forearm_ratio: float = 0.60,
        wrist_sleeve_hand_overlap_ratio: float = 0.25,
        condition_fill_value: float = 0.0,
        include_numpy: bool = False,
        max_open_sequences: int = 2,
    ) -> None:
        self.manifest_path = Path(manifest)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.samples = [
            json.loads(line)
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.samples:
            raise RuntimeError(f"Manifest has no samples: {self.manifest_path}")
        if condition_variant not in {"sam_mask", "direct_overlay"}:
            raise ValueError("condition_variant must be sam_mask or direct_overlay")
        self.pinhole_root = Path(pinhole_root)
        self.render_root = Path(render_root)
        self.mask_root = Path(mask_root)
        self.output_size = int(output_size)
        self.condition_variant = condition_variant
        self.render_opacity = float(render_opacity)
        if not 0.0 <= self.render_opacity <= 1.0:
            raise ValueError("render_opacity must be in [0, 1]")
        if render_kind not in {"gaussian", "mano"}:
            raise ValueError("render_kind must be gaussian or mano")
        self.render_kind = render_kind
        self.render_alpha_filename = str(render_alpha_filename)
        self.alpha_threshold = float(alpha_threshold)
        if not 0.0 <= self.alpha_threshold <= 1.0:
            raise ValueError("alpha_threshold must be in [0, 1]")
        if loss_mask_source not in {"overlay", "sam", "mano"}:
            raise ValueError("loss_mask_source must be overlay, sam, or mano")
        self.loss_mask_source = loss_mask_source
        self.loss_mask_root = Path(loss_mask_root) if loss_mask_root else None
        if spatial_mode not in {"full_frame", "mano_square_crop"}:
            raise ValueError("spatial_mode must be full_frame or mano_square_crop")
        if condition_style not in {"legacy_overlay", "glove2hand_faithful"}:
            raise ValueError(
                "condition_style must be legacy_overlay or glove2hand_faithful"
            )
        if crop_scale < 1.0:
            raise ValueError("crop_scale must be at least 1.0")
        if hand_mask_dilation_px < 0:
            raise ValueError("hand_mask_dilation_px must be nonnegative")
        if not 0.0 <= condition_fill_value <= 1.0:
            raise ValueError("condition_fill_value must be in [0,1]")
        if wrist_geometry_source not in {"silhouette_pca", "mano_wrist_ring"}:
            raise ValueError(
                "wrist_geometry_source must be silhouette_pca or mano_wrist_ring"
            )
        self.spatial_mode = spatial_mode
        self.crop_scale = float(crop_scale)
        self.condition_style = condition_style
        self.hand_mask_dilation_px = int(hand_mask_dilation_px)
        self.wrist_mask_enabled = bool(wrist_mask_enabled)
        self.wrist_geometry_source = wrist_geometry_source
        self.wrist_ring_root = (
            Path(wrist_ring_root) if wrist_ring_root is not None else None
        )
        if (
            self.wrist_mask_enabled
            and self.wrist_geometry_source == "mano_wrist_ring"
            and self.wrist_ring_root is None
        ):
            raise ValueError(
                "wrist_ring_root is required for mano_wrist_ring geometry"
            )
        self.wrist_length_ratio = float(wrist_length_ratio)
        self.wrist_width_scale = float(wrist_width_scale)
        self.wrist_ring_transverse_scale = float(
            wrist_ring_transverse_scale
        )
        if wrist_sleeve_orientation not in {"ring_min_area", "palm_axis"}:
            raise ValueError("unknown wrist sleeve orientation")
        self.wrist_sleeve_orientation = wrist_sleeve_orientation
        self.wrist_sleeve_forearm_ratio = float(
            wrist_sleeve_forearm_ratio
        )
        self.wrist_sleeve_hand_overlap_ratio = float(
            wrist_sleeve_hand_overlap_ratio
        )
        self.condition_fill_value = float(condition_fill_value)
        self._needs_mano_alpha = (
            self.loss_mask_source == "mano"
            or self.spatial_mode == "mano_square_crop"
            or self.condition_style == "glove2hand_faithful"
        )
        if self._needs_mano_alpha and self.loss_mask_root is None:
            raise ValueError(
                "loss_mask_root is required for MANO crop/mask conditioning"
            )
        self._loss_mask_reuses_render = (
            self.loss_mask_root is not None
            and self.render_root.resolve() == self.loss_mask_root.resolve()
        )
        self.loss_mask_alpha_filename = str(loss_mask_alpha_filename)
        self.include_numpy = bool(include_numpy)
        self.max_open_sequences = int(max_open_sequences)
        self._videos: OrderedDict[
            str,
            tuple[cv2.VideoCapture, cv2.VideoCapture, cv2.VideoCapture, int],
        ] = OrderedDict()
        self._loss_mask_videos: OrderedDict[str, tuple[cv2.VideoCapture, int]] = OrderedDict()
        self._databases: OrderedDict[str, sqlite3.Connection] = OrderedDict()
        self._wrist_rings: OrderedDict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = OrderedDict()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_videos"] = OrderedDict()
        state["_loss_mask_videos"] = OrderedDict()
        state["_databases"] = OrderedDict()
        state["_wrist_rings"] = OrderedDict()
        return state

    def __del__(self) -> None:
        for rgb, render, alpha, _ in getattr(self, "_videos", {}).values():
            rgb.release()
            render.release()
            alpha.release()
        for capture, _ in getattr(self, "_loss_mask_videos", {}).values():
            capture.release()
        for database in getattr(self, "_databases", {}).values():
            database.close()

    def __len__(self) -> int:
        return len(self.samples)

    def _video_triplet(
        self, sequence: str
    ) -> tuple[cv2.VideoCapture, cv2.VideoCapture, cv2.VideoCapture, int]:
        triplet = self._videos.pop(sequence, None)
        if triplet is None:
            rgb = cv2.VideoCapture(
                str(self.pinhole_root / sequence / "rgb_214_1_pinhole.mp4")
            )
            render = cv2.VideoCapture(
                str(self.render_root / sequence / "reconstruction.mp4")
            )
            alpha = cv2.VideoCapture(
                str(self.render_root / sequence / self.render_alpha_filename)
            )
            if not rgb.isOpened() or not render.isOpened() or not alpha.isOpened():
                rgb.release()
                render.release()
                alpha.release()
                raise RuntimeError(
                    f"Could not open aligned RGB/render/alpha for {sequence}"
                )
            counts = tuple(
                int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                for capture in (rgb, render, alpha)
            )
            if len(set(counts)) != 1:
                rgb.release()
                render.release()
                alpha.release()
                raise RuntimeError(
                    f"Frame-count mismatch for {sequence}: "
                    f"rgb/render/alpha={counts}"
                )
            triplet = (rgb, render, alpha, -1)
        self._videos[sequence] = triplet
        while len(self._videos) > self.max_open_sequences:
            _, (old_rgb, old_render, old_alpha, _) = self._videos.popitem(last=False)
            old_rgb.release()
            old_render.release()
            old_alpha.release()
        return triplet

    def _read_triplet(
        self, sequence: str, frame_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb_capture, render_capture, alpha_capture, next_frame = (
            self._video_triplet(sequence)
        )
        if next_frame != frame_index:
            for capture in (rgb_capture, render_capture, alpha_capture):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok_rgb, rgb = rgb_capture.read()
        ok_render, render = render_capture.read()
        ok_alpha, alpha = alpha_capture.read()
        if not ok_rgb or not ok_render or not ok_alpha:
            raise RuntimeError(
                f"Could not decode aligned RGB/render/alpha "
                f"for {sequence}/{frame_index}"
            )
        self._videos[sequence] = (
            rgb_capture, render_capture, alpha_capture, frame_index + 1
        )
        return (
            cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(render, cv2.COLOR_BGR2RGB),
            alpha[:, :, 0],
        )

    def _database(self, sequence: str) -> sqlite3.Connection:
        database = self._databases.pop(sequence, None)
        if database is None:
            path = self.mask_root / sequence / "masks.sqlite"
            uri = f"file:{path.resolve()}?mode=ro"
            # Web inference may serve serialized requests from different worker
            # threads. Callers must still serialize access to this connection.
            database = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._databases[sequence] = database
        while len(self._databases) > self.max_open_sequences:
            _, old = self._databases.popitem(last=False)
            old.close()
        return database

    def _read_loss_mask_render(self, sequence: str, frame_index: int) -> np.ndarray:
        item = self._loss_mask_videos.pop(sequence, None)
        if item is None:
            path = self.loss_mask_root / sequence / self.loss_mask_alpha_filename
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Could not open MANO loss-mask alpha: {path}")
            item = (capture, -1)
        capture, next_frame = item
        if next_frame != frame_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(
                f"Could not decode MANO loss mask: {sequence}/{frame_index}"
            )
        self._loss_mask_videos[sequence] = (capture, frame_index + 1)
        while len(self._loss_mask_videos) > self.max_open_sequences:
            _, (old_capture, _) = self._loss_mask_videos.popitem(last=False)
            old_capture.release()
        return frame[:, :, 0]

    def _sam_mask(self, sequence: str, frame_index: int, shape: tuple[int, int]) -> np.ndarray:
        row = self._database(sequence).execute(
            "SELECT labels_zlib,training_eligible FROM frames WHERE frame_index=?",
            (frame_index,),
        ).fetchone()
        if row is None or not bool(row[1]):
            raise RuntimeError(f"Manifest contains ineligible SAM frame: {sequence}/{frame_index}")
        labels = np.frombuffer(zlib.decompress(row[0]), dtype=np.uint8).reshape(shape)
        return labels > 0

    def _wrist_ring_points(
        self, sequence: str, frame_index: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.wrist_mask_enabled or self.wrist_geometry_source != "mano_wrist_ring":
            return None, None
        item = self._wrist_rings.pop(sequence, None)
        if item is None:
            path = self.wrist_ring_root / sequence / "wrist_ring_points.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as payload:
                points = payload["points"].astype(np.float32)
                palm_points = payload["palm_points"].astype(np.float32)
                valid = payload["valid"].astype(bool)
            if points.ndim != 4 or points.shape[1:] != (2, 16, 2):
                raise RuntimeError(f"Invalid wrist-ring point shape: {path}")
            if palm_points.shape != (points.shape[0], 2, 2):
                raise RuntimeError(f"Invalid wrist palm-point shape: {path}")
            if valid.shape != points.shape[:2]:
                raise RuntimeError(f"Invalid wrist-ring validity shape: {path}")
            item = (points, palm_points, valid)
        self._wrist_rings[sequence] = item
        while len(self._wrist_rings) > self.max_open_sequences:
            self._wrist_rings.popitem(last=False)
        points, palm_points, valid = item
        if not 0 <= frame_index < points.shape[0]:
            raise IndexError(f"Wrist-ring frame is out of range: {sequence}/{frame_index}")
        frame_points = points[frame_index].copy()
        frame_palm_points = palm_points[frame_index].copy()
        frame_points[~valid[frame_index]] = np.nan
        frame_palm_points[~valid[frame_index]] = np.nan
        if not np.isfinite(frame_points).any():
            raise RuntimeError(f"No valid MANO wrist ring: {sequence}/{frame_index}")
        return frame_points, frame_palm_points

    def __getitem__(self, index: int) -> dict:
        record = self.samples[index]
        sequence = str(record["sequence_id"])
        frame_index = int(record["frame_index"])
        target_native, render_native, render_alpha_native = self._read_triplet(
            sequence, frame_index
        )
        original_image_size = target_native.shape[:2]
        mano_loss_alpha = None
        if self._needs_mano_alpha:
            mano_loss_alpha = (
                render_alpha_native
                if self._loss_mask_reuses_render
                else self._read_loss_mask_render(sequence, frame_index)
            )
        sam_mask = None
        if self.condition_variant == "sam_mask":
            sam_mask = self._sam_mask(
                sequence, frame_index, target_native.shape[:2]
            )
        wrist_ring_points, wrist_palm_points = self._wrist_ring_points(
            sequence, frame_index
        )

        if self.spatial_mode == "mano_square_crop":
            mano_box = square_crop_from_mask(
                mano_loss_alpha > 0, scale=self.crop_scale
            )
            target_box = map_crop_box(
                mano_box, mano_loss_alpha.shape, target_native.shape[:2]
            )
            render_box = map_crop_box(
                mano_box, mano_loss_alpha.shape, render_native.shape[:2]
            )
            alpha_box = map_crop_box(
                mano_box, mano_loss_alpha.shape, render_alpha_native.shape[:2]
            )
            target = crop_resize(
                target_native, target_box, self.output_size, cv2.INTER_AREA
            )
            render = crop_resize(
                render_native, render_box, self.output_size, cv2.INTER_AREA
            )
            render_alpha = crop_resize(
                render_alpha_native, alpha_box, self.output_size, cv2.INTER_AREA
            )
            mano_loss_alpha = crop_resize(
                mano_loss_alpha, mano_box, self.output_size, cv2.INTER_NEAREST
            )
            if sam_mask is not None:
                sam_mask = crop_resize(
                    sam_mask.astype(np.uint8),
                    target_box,
                    self.output_size,
                    cv2.INTER_NEAREST,
                ).astype(bool)
            if wrist_ring_points is not None:
                for projected_points in (wrist_ring_points, wrist_palm_points):
                    projected_points[..., 0] = (
                        (projected_points[..., 0] - render_box.x0 + 0.5)
                        * self.output_size
                        / render_box.width
                        - 0.5
                    )
                    projected_points[..., 1] = (
                        (projected_points[..., 1] - render_box.y0 + 0.5)
                        * self.output_size
                        / render_box.height
                        - 0.5
                    )
        else:
            target_box = CropBox(
                0, 0, target_native.shape[1], target_native.shape[0]
            )
            size = (self.output_size, self.output_size)
            target = cv2.resize(target_native, size, interpolation=cv2.INTER_AREA)
            render = cv2.resize(render_native, size, interpolation=cv2.INTER_AREA)
            render_alpha = cv2.resize(
                render_alpha_native, size, interpolation=cv2.INTER_AREA
            )
            if mano_loss_alpha is not None:
                mano_loss_alpha = cv2.resize(
                    mano_loss_alpha, size, interpolation=cv2.INTER_NEAREST
                )
            if sam_mask is not None:
                sam_mask = cv2.resize(
                    sam_mask.astype(np.uint8),
                    size,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if wrist_ring_points is not None:
                for projected_points in (wrist_ring_points, wrist_palm_points):
                    projected_points[..., 0] = (
                        (projected_points[..., 0] + 0.5)
                        * self.output_size
                        / render_native.shape[1]
                        - 0.5
                    )
                    projected_points[..., 1] = (
                        (projected_points[..., 1] + 0.5)
                        * self.output_size
                        / render_native.shape[0]
                        - 0.5
                    )

        render_alpha = render_alpha.astype(np.float32) / 255.0
        render_alpha = np.where(
            render_alpha > self.alpha_threshold, render_alpha, 0.0
        ).astype(np.float32)
        render_foreground = render_alpha > self.alpha_threshold
        mano_loss_mask = None
        if mano_loss_alpha is not None:
            mano_loss_mask = mano_loss_alpha > 0

        target = target.astype(np.float32) / 255.0
        render = render.astype(np.float32) / 255.0
        if self.condition_style == "glove2hand_faithful":
            faithful = build_faithful_condition(
                target,
                render,
                render_alpha,
                mano_loss_mask,
                visible_hand_mask=sam_mask,
                wrist_ring_points=wrist_ring_points,
                wrist_palm_points=wrist_palm_points,
                dilation_px=self.hand_mask_dilation_px,
                wrist_enabled=self.wrist_mask_enabled,
                wrist_length_ratio=self.wrist_length_ratio,
                wrist_width_scale=self.wrist_width_scale,
                wrist_ring_transverse_scale=(
                    self.wrist_ring_transverse_scale
                ),
                wrist_sleeve_orientation=self.wrist_sleeve_orientation,
                wrist_sleeve_forearm_ratio=(
                    self.wrist_sleeve_forearm_ratio
                ),
                wrist_sleeve_hand_overlap_ratio=(
                    self.wrist_sleeve_hand_overlap_ratio
                ),
                fill_value=self.condition_fill_value,
                render_opacity=self.render_opacity,
            )
            condition = faithful.condition_rgb
            raw_overlay = faithful.raw_overlay_rgb
            overlay = faithful.overlay_rgb
            mano_mask = faithful.mano_mask
            visible_hand_mask = faithful.visible_hand_mask
            sam_excluded_mano_mask = faithful.sam_excluded_mano_mask
            edit_mask = faithful.edit_mask
            loss_weight_mask = faithful.mano_mask
            condition_mask = faithful.visible_hand_mask
            dilated_hand_mask = faithful.dilated_hand_mask
            wrist_mask = faithful.wrist_mask
            wrist_polygons = faithful.wrist_polygons
        else:
            overlay_mask, loss_weight_mask = overlay_and_loss_masks(
                render_foreground,
                sam_mask,
                self.loss_mask_source,
                mano_mask=mano_loss_mask,
            )
            effective_alpha = (
                render_alpha
                * overlay_mask.astype(np.float32)
                * self.render_opacity
            )[..., None]
            condition = (
                target * (1.0 - effective_alpha) + render * effective_alpha
            )
            overlay = condition
            raw_overlay = condition
            mano_mask = overlay_mask
            visible_hand_mask = overlay_mask
            sam_excluded_mano_mask = np.zeros_like(
                mano_mask, dtype=bool
            )
            edit_mask = loss_weight_mask
            condition_mask = loss_weight_mask
            dilated_hand_mask = mano_mask
            wrist_mask = np.zeros_like(mano_mask, dtype=bool)
            wrist_polygons = ()

        mano_mask_float = mano_mask.astype(np.float32)
        edit_mask_float = edit_mask.astype(np.float32)
        loss_mask_float = loss_weight_mask.astype(np.float32)
        condition_mask_float = condition_mask.astype(np.float32)
        result = {
            "target_rgb": torch.from_numpy(
                target.transpose(2, 0, 1) * 2.0 - 1.0
            ),
            "mano_rgb": torch.from_numpy(
                render.transpose(2, 0, 1) * 2.0 - 1.0
            ),
            "mano_mask": torch.from_numpy(mano_mask_float[None]),
            "condition_rgb": torch.from_numpy(
                condition.transpose(2, 0, 1) * 2.0 - 1.0
            ),
            "condition_mask": torch.from_numpy(condition_mask_float[None]),
            "edit_mask": torch.from_numpy(edit_mask_float[None]),
            "loss_mask": torch.from_numpy(loss_mask_float[None]),
            "metadata": {
                "sequence_id": sequence,
                "frame_id": frame_index,
                "camera_id": "214-1-pinhole",
                "handedness": "both",
                "participant_id": sequence.split("_", 1)[0],
                "source_sequence_id": sequence,
                "original_image_size": list(original_image_size),
                "model_input_size": [self.output_size, self.output_size],
                "spatial_mode": self.spatial_mode,
                "crop_scale": self.crop_scale,
                "crop_box_xyxy": target_box.as_list(),
                "condition_variant": self.condition_variant,
                "condition_style": self.condition_style,
                "overlay_logic": (
                    (
                        "mano_alpha_intersect_sam_visible_hand_then_"
                        "blank_wrist_preserve_interaction"
                    )
                    if self.condition_style == "glove2hand_faithful"
                    else (
                        f"{self.render_kind}_alpha_intersect_sam"
                        if self.condition_variant == "sam_mask"
                        else f"{self.render_kind}_alpha"
                    )
                ),
                "hand_mask_dilation_px": self.hand_mask_dilation_px,
                "wrist_mask_enabled": self.wrist_mask_enabled,
                "wrist_geometry_source": self.wrist_geometry_source,
                "wrist_length_ratio": self.wrist_length_ratio,
                "wrist_width_scale": self.wrist_width_scale,
                "wrist_ring_transverse_scale": (
                    self.wrist_ring_transverse_scale
                ),
                "wrist_sleeve_orientation": self.wrist_sleeve_orientation,
                "wrist_sleeve_forearm_ratio": (
                    self.wrist_sleeve_forearm_ratio
                ),
                "wrist_sleeve_hand_overlap_ratio": (
                    self.wrist_sleeve_hand_overlap_ratio
                ),
                "render_kind": self.render_kind,
                "render_root": str(self.render_root),
                "alpha_source": self.render_alpha_filename,
                "loss_mask_source": self.loss_mask_source,
                "loss_mask_root": (
                    None if self.loss_mask_root is None
                    else str(self.loss_mask_root)
                ),
            },
        }
        if self.include_numpy:
            full_target = target_native.astype(np.float32) / 255.0
            full_condition, full_edit_mask = paste_crop_with_mask(
                full_target, condition, edit_mask, target_box
            )
            result.update({
                "target_rgb_np": target,
                "mano_rgb_np": render,
                "mano_mask_np": mano_mask_float,
                "raw_overlay_rgb_np": raw_overlay,
                "overlay_rgb_np": overlay,
                "visible_hand_mask_np": visible_hand_mask.astype(np.float32),
                "sam_excluded_mano_mask_np": (
                    sam_excluded_mano_mask.astype(np.float32)
                ),
                "condition_rgb_np": condition,
                "condition_mask_np": condition_mask_float,
                "edit_mask_np": edit_mask_float,
                "loss_mask_np": loss_mask_float,
                "dilated_hand_mask_np": dilated_hand_mask.astype(np.float32),
                "wrist_mask_np": wrist_mask.astype(np.float32),
                "wrist_polygons_np": wrist_polygons,
                "wrist_ring_points_np": wrist_ring_points,
                "wrist_palm_points_np": wrist_palm_points,
                "full_target_rgb_np": full_target,
                "full_condition_rgb_np": full_condition,
                "full_edit_mask_np": full_edit_mask.astype(np.float32),
                "crop_box": target_box,
            })
        return result
