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
        if self.loss_mask_source == "mano" and self.loss_mask_root is None:
            raise ValueError("loss_mask_root is required for MANO loss masks")
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

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_videos"] = OrderedDict()
        state["_loss_mask_videos"] = OrderedDict()
        state["_databases"] = OrderedDict()
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

    def __getitem__(self, index: int) -> dict:
        record = self.samples[index]
        sequence = str(record["sequence_id"])
        frame_index = int(record["frame_index"])
        target, render, render_alpha = self._read_triplet(sequence, frame_index)
        mano_loss_alpha = None
        if self.loss_mask_source == "mano":
            mano_loss_alpha = (
                render_alpha
                if self._loss_mask_reuses_render
                else self._read_loss_mask_render(sequence, frame_index)
            )
        sam_mask = None
        if self.condition_variant == "sam_mask":
            sam_mask = self._sam_mask(sequence, frame_index, target.shape[:2])

        size = (self.output_size, self.output_size)
        target = cv2.resize(target, size, interpolation=cv2.INTER_AREA)
        render = cv2.resize(render, size, interpolation=cv2.INTER_AREA)
        render_alpha = cv2.resize(
            render_alpha, size, interpolation=cv2.INTER_AREA
        ).astype(np.float32) / 255.0
        render_foreground = render_alpha > self.alpha_threshold
        mano_loss_mask = None
        if mano_loss_alpha is not None:
            mano_loss_alpha = cv2.resize(
                mano_loss_alpha, size, interpolation=cv2.INTER_NEAREST
            )
            mano_loss_mask = mano_loss_alpha > 0
        if sam_mask is not None:
            sam_mask = cv2.resize(
                sam_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        overlay_mask, loss_mask = overlay_and_loss_masks(
            render_foreground,
            sam_mask,
            self.loss_mask_source,
            mano_mask=mano_loss_mask,
        )

        target = target.astype(np.float32) / 255.0
        render = render.astype(np.float32) / 255.0
        effective_alpha = (
            render_alpha
            * overlay_mask.astype(np.float32)
            * self.render_opacity
        )[..., None]
        condition = target * (1.0 - effective_alpha) + render * effective_alpha
        overlay_mask_float = overlay_mask.astype(np.float32)
        loss_mask_float = loss_mask.astype(np.float32)
        result = {
            "target_rgb": torch.from_numpy(
                target.transpose(2, 0, 1) * 2.0 - 1.0
            ),
            "mano_rgb": torch.from_numpy(
                render.transpose(2, 0, 1) * 2.0 - 1.0
            ),
            "mano_mask": torch.from_numpy(overlay_mask_float[None]),
            "condition_rgb": torch.from_numpy(
                condition.transpose(2, 0, 1) * 2.0 - 1.0
            ),
            "edit_mask": torch.from_numpy(loss_mask_float[None]),
            "metadata": {
                "sequence_id": sequence,
                "frame_id": frame_index,
                "camera_id": "214-1-pinhole",
                "handedness": "both",
                "participant_id": sequence.split("_", 1)[0],
                "source_sequence_id": sequence,
                "original_image_size": target.shape[:2],
                "condition_variant": self.condition_variant,
                "overlay_logic": (
                    f"{self.render_kind}_alpha_intersect_sam"
                    if self.condition_variant == "sam_mask"
                    else f"{self.render_kind}_alpha"
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
            result.update({
                "target_rgb_np": target,
                "mano_rgb_np": render,
                "mano_mask_np": overlay_mask_float,
                "condition_rgb_np": condition,
                "edit_mask_np": loss_mask_float,
            })
        return result
