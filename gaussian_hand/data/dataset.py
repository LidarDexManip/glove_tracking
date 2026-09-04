"""PyTorch dataset over exported Gaussian Hand crops."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class GaussianHandDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", resolution: int | None = None):
        self.root = Path(root)
        records = [json.loads(line) for line in (self.root / "frames.jsonl").open()]
        self.records = [r for r in records if r["split"] == split]
        if not self.records:
            raise ValueError(f"No {split} records in {self.root}")
        self.resolution = resolution

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        image = cv2.imread(str(self.root / r["image"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.root / r["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(r["image"])
        source_size, size = int(r["output_size"]), self.resolution or int(r["output_size"])
        if size != source_size:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        k = np.asarray(r["crop_intrinsics"], np.float32)
        if size != source_size:
            scale = size / source_size
            k[0, 0] *= scale; k[1, 1] *= scale
            k[0, 2] = (k[0, 2] + .5) * scale - .5
            k[1, 2] = (k[1, 2] + .5) * scale - .5
        return {
            "dataset_index": index,
            "image": torch.from_numpy(image[..., ::-1].copy()).float() / 255,
            "mask": torch.from_numpy((mask > 127)[..., None].astype(np.float32)),
            "mano_pca": torch.tensor(r["mano_pca"], dtype=torch.float32),
            "wrist_q_wxyz": torch.tensor(r["wrist_q_wxyz"], dtype=torch.float32),
            "wrist_t_world": torch.tensor(r["wrist_t_world"], dtype=torch.float32),
            "world_to_camera": torch.tensor(r["world_to_camera"], dtype=torch.float32),
            "intrinsics": torch.tensor(k, dtype=torch.float32),
            "sequence": r["sequence"], "frame_index": int(r["frame_index"]),
        }
