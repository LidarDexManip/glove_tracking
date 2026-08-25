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


class HuggAriaGaussianDataset(Dataset):
    """Read aligned RGB/Gaussian videos and optional SAM labels on demand."""

    def __init__(
        self,
        manifest: str | Path,
        pinhole_root: str | Path,
        gaussian_root: str | Path,
        mask_root: str | Path,
        output_size: int = 512,
        condition_variant: str = "sam_mask",
        gaussian_opacity: float = 1.0,
        gaussian_threshold: int = 16,
        render_kind: str = "gaussian",
        loss_mask_source: str = "overlay",
        loss_mask_root: str | Path | None = None,
        loss_mask_threshold: int | None = None,
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
        self.gaussian_root = Path(gaussian_root)
        self.mask_root = Path(mask_root)
        self.output_size = int(output_size)
        self.condition_variant = condition_variant
        self.gaussian_opacity = float(gaussian_opacity)
        self.gaussian_threshold = int(gaussian_threshold)
        if render_kind not in {"gaussian", "mano"}:
            raise ValueError("render_kind must be gaussian or mano")
        self.render_kind = render_kind
        if loss_mask_source not in {"overlay", "sam", "mano"}:
            raise ValueError("loss_mask_source must be overlay, sam, or mano")
        self.loss_mask_source = loss_mask_source
        self.loss_mask_root = Path(loss_mask_root) if loss_mask_root else None
        if self.loss_mask_source == "mano" and self.loss_mask_root is None:
            raise ValueError("loss_mask_root is required for MANO loss masks")
        self._loss_mask_reuses_render = (
            self.loss_mask_root is not None
            and self.gaussian_root.resolve() == self.loss_mask_root.resolve()
        )
        self.loss_mask_threshold = int(
            gaussian_threshold if loss_mask_threshold is None else loss_mask_threshold
        )
        self.include_numpy = bool(include_numpy)
        self.max_open_sequences = int(max_open_sequences)
        self._videos: OrderedDict[str, tuple[cv2.VideoCapture, cv2.VideoCapture, int, int]] = OrderedDict()
        self._loss_mask_videos: OrderedDict[str, tuple[cv2.VideoCapture, int]] = OrderedDict()
        self._databases: OrderedDict[str, sqlite3.Connection] = OrderedDict()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_videos"] = OrderedDict()
        state["_loss_mask_videos"] = OrderedDict()
        state["_databases"] = OrderedDict()
        return state

    def __del__(self) -> None:
        for rgb, gaussian, _, _ in getattr(self, "_videos", {}).values():
            rgb.release()
            gaussian.release()
        for capture, _ in getattr(self, "_loss_mask_videos", {}).values():
            capture.release()
        for database in getattr(self, "_databases", {}).values():
            database.close()

    def __len__(self) -> int:
        return len(self.samples)

    def _video_pair(self, sequence: str) -> tuple[cv2.VideoCapture, cv2.VideoCapture, int, int]:
        pair = self._videos.pop(sequence, None)
        if pair is None:
            rgb = cv2.VideoCapture(str(self.pinhole_root / sequence / "rgb_214_1_pinhole.mp4"))
            gaussian = cv2.VideoCapture(str(self.gaussian_root / sequence / "reconstruction.mp4"))
            if not rgb.isOpened() or not gaussian.isOpened():
                rgb.release()
                gaussian.release()
                raise RuntimeError(f"Could not open aligned videos for {sequence}")
            pair = (rgb, gaussian, -1, -1)
        self._videos[sequence] = pair
        while len(self._videos) > self.max_open_sequences:
            _, (old_rgb, old_gaussian, _, _) = self._videos.popitem(last=False)
            old_rgb.release()
            old_gaussian.release()
        return pair

    def _read_pair(
        self, sequence: str, frame_index: int, gaussian_frame_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rgb_capture, gaussian_capture, next_rgb, next_gaussian = self._video_pair(sequence)
        if next_rgb != frame_index:
            rgb_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        if next_gaussian != gaussian_frame_index:
            gaussian_capture.set(cv2.CAP_PROP_POS_FRAMES, gaussian_frame_index)
        ok_rgb, rgb = rgb_capture.read()
        ok_gaussian, gaussian = gaussian_capture.read()
        if not ok_rgb or not ok_gaussian:
            raise RuntimeError(
                f"Could not decode {sequence} source={frame_index} "
                f"gaussian={gaussian_frame_index}"
            )
        self._videos[sequence] = (
            rgb_capture, gaussian_capture, frame_index + 1, gaussian_frame_index + 1
        )
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)

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
            path = self.loss_mask_root / sequence / "reconstruction.mp4"
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Could not open MANO loss-mask video: {path}")
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
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _sam_mask(self, sequence: str, frame_index: int, shape: tuple[int, int]) -> np.ndarray:
        row = self._database(sequence).execute(
            "SELECT labels_zlib,training_eligible FROM frames WHERE frame_index=?",
            (frame_index,),
        ).fetchone()
        if row is None or not bool(row[1]):
            raise RuntimeError(f"Manifest contains ineligible SAM frame: {sequence}/{frame_index}")
        labels = np.frombuffer(zlib.decompress(row[0]), dtype=np.uint8).reshape(shape)
        return labels > 0

    def __getitem__(self, index: int) -> dict:
        record = self.samples[index]
        sequence = str(record["sequence_id"])
        frame_index = int(record["frame_index"])
        gaussian_frame_index = int(record["gaussian_frame_index"])
        target, gaussian = self._read_pair(sequence, frame_index, gaussian_frame_index)
        mano_loss_render = None
        if self.loss_mask_source == "mano":
            same_render = (
                self._loss_mask_reuses_render and frame_index == gaussian_frame_index
            )
            mano_loss_render = (
                gaussian
                if same_render
                else self._read_loss_mask_render(sequence, frame_index)
            )
        sam_mask = None
        if self.condition_variant == "sam_mask":
            sam_mask = self._sam_mask(sequence, frame_index, target.shape[:2])

        size = (self.output_size, self.output_size)
        target = cv2.resize(target, size, interpolation=cv2.INTER_AREA)
        gaussian = cv2.resize(gaussian, size, interpolation=cv2.INTER_AREA)
        gaussian_foreground = gaussian.max(axis=2) > self.gaussian_threshold
        mano_loss_mask = None
        if mano_loss_render is not None:
            mano_loss_render = cv2.resize(
                mano_loss_render, size, interpolation=cv2.INTER_AREA
            )
            mano_loss_mask = (
                mano_loss_render.max(axis=2) > self.loss_mask_threshold
            )
        if sam_mask is not None:
            sam_mask = cv2.resize(
                sam_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        overlay_mask, loss_mask = overlay_and_loss_masks(
            gaussian_foreground,
            sam_mask,
            self.loss_mask_source,
            mano_mask=mano_loss_mask,
        )
        target = target.astype(np.float32) / 255.0
        gaussian = gaussian.astype(np.float32) / 255.0
        condition = target.copy()
        alpha = self.gaussian_opacity
        condition[overlay_mask] = (
            (1.0 - alpha) * target[overlay_mask] + alpha * gaussian[overlay_mask]
        )
        overlay_mask_float = overlay_mask.astype(np.float32)
        loss_mask_float = loss_mask.astype(np.float32)
        result = {
            "target_rgb": torch.from_numpy(target.transpose(2, 0, 1) * 2.0 - 1.0),
            "mano_rgb": torch.from_numpy(gaussian.transpose(2, 0, 1) * 2.0 - 1.0),
            "mano_mask": torch.from_numpy(overlay_mask_float[None]),
            "condition_rgb": torch.from_numpy(condition.transpose(2, 0, 1) * 2.0 - 1.0),
            "edit_mask": torch.from_numpy(loss_mask_float[None]),
            "metadata": {
                "sequence_id": sequence,
                "frame_id": frame_index,
                "gaussian_frame_id": gaussian_frame_index,
                "camera_id": "214-1-pinhole",
                "handedness": "both",
                "participant_id": sequence.split("_", 1)[0],
                "source_sequence_id": sequence,
                "original_image_size": target.shape[:2],
                "condition_variant": self.condition_variant,
                "overlay_logic": (
                    f"{self.render_kind}_foreground_intersect_sam"
                    if self.condition_variant == "sam_mask"
                    else f"{self.render_kind}_foreground"
                ),
                "render_kind": self.render_kind,
                "loss_mask_source": self.loss_mask_source,
                "loss_mask_root": (
                    None if self.loss_mask_root is None else str(self.loss_mask_root)
                ),
            },
        }
        if self.include_numpy:
            result.update({
                "target_rgb_np": target,
                "mano_rgb_np": gaussian,
                "mano_mask_np": overlay_mask_float,
                "condition_rgb_np": condition,
                "edit_mask_np": loss_mask_float,
            })
        return result
