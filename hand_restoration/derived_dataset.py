from __future__ import annotations

import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .conditions import ConditionBuilder, ConditionConfig


def _decode_image(data: bytes, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    if image is None:
        raise RuntimeError("Failed to decode derived image.")
    return image


class DerivedHandRestorationDataset(Dataset):
    """Map-style reader for offline C1 WebDataset-compatible tar shards."""

    def __init__(
        self,
        manifest: str | Path,
        output_size: int = 512,
        grayscale: bool = True,
        condition: ConditionConfig | None = None,
        seed: int = 0,
        include_numpy: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Derived manifest not found: {self.manifest_path}")
        self.root = self.manifest_path.parent
        self.samples = [json.loads(line) for line in self.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.samples:
            raise RuntimeError(f"Derived manifest has no samples: {self.manifest_path}")
        self.output_size = int(output_size)
        self.grayscale = bool(grayscale)
        if not self.grayscale:
            raise ValueError("Derived dataset currently stores the verified grayscale C1 path only.")
        self.include_numpy = include_numpy
        self.condition_builder = ConditionBuilder(condition or ConditionConfig(), seed=seed)
        self._tar_handles: dict[Path, tarfile.TarFile] = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_tar_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_tar_handles", {}).values():
            handle.close()

    def __len__(self) -> int:
        return len(self.samples)

    def _read(self, shard: Path, member: str) -> bytes:
        handle = self._tar_handles.get(shard)
        if handle is None:
            handle = tarfile.open(shard, "r")
            self._tar_handles[shard] = handle
        extracted = handle.extractfile(member)
        if extracted is None:
            raise KeyError(f"Missing {member} in {shard}")
        return extracted.read()

    def __getitem__(self, index: int) -> dict:
        record = self.samples[index]
        shard = Path(record["shard"])
        if not shard.is_absolute():
            shard = self.root / shard
        key = record["key"]
        target = _decode_image(self._read(shard, f"{key}.target.jpg"), cv2.IMREAD_GRAYSCALE)
        mano = _decode_image(self._read(shard, f"{key}.mano.png"), cv2.IMREAD_GRAYSCALE)
        mask = _decode_image(self._read(shard, f"{key}.mask.png"), cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        mano = cv2.resize(mano, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        target_rgb = np.repeat(target[..., None], 3, axis=2)
        mano_rgb = np.repeat(mano[..., None], 3, axis=2)
        condition_rgb, edit_mask = self.condition_builder(target_rgb, mano_rgb, mask)
        result = {
            "target_rgb": torch.from_numpy(target_rgb.transpose(2, 0, 1) * 2.0 - 1.0),
            "mano_rgb": torch.from_numpy(mano_rgb.transpose(2, 0, 1) * 2.0 - 1.0),
            "mano_mask": torch.from_numpy(mask[None]),
            "condition_rgb": torch.from_numpy(condition_rgb.transpose(2, 0, 1) * 2.0 - 1.0),
            "edit_mask": torch.from_numpy(edit_mask[None].astype(np.float32)),
            "metadata": {
                "sequence_id": record["clip_id"],
                "frame_id": record["frame_id"],
                "camera_id": record["camera_id"],
                "handedness": record["handedness"],
                "participant_id": record.get("participant_id", ""),
                "source_sequence_id": record.get("source_sequence_id", ""),
                "original_image_size": record["canonical_image_size"],
            },
        }
        if self.include_numpy:
            result.update({"target_rgb_np": target_rgb, "mano_rgb_np": mano_rgb, "mano_mask_np": mask, "condition_rgb_np": condition_rgb, "edit_mask_np": edit_mask})
        return result
