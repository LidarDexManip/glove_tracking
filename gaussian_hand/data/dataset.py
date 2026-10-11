"""PyTorch dataset over exported Gaussian Hand crops."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class GaussianHandDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", resolution: int | None = None, require_quality: bool = False):
        self.root = Path(root)
        records = [json.loads(line) for line in (self.root / "frames.jsonl").open()]
        self.records = records if split == "all" else [
            r for r in records if r["split"] == split
        ]
        if not self.records:
            raise ValueError(f"No {split} records in {self.root}")
        if require_quality:
            summary = json.loads((self.root / 'summary.json').read_text())
            preparation = json.loads((self.root.parent / 'preparation.json').read_text())
            import hashlib
            if not preparation.get('complete') or summary['frames_sha256'] != hashlib.sha256((self.root/'frames.jsonl').read_bytes()).hexdigest():
                raise ValueError('Incomplete or changed quality-filtered dataset')
            for record in self.records:
                if (record.get('quality_pass') is not True or record.get('mask_source') != 'hf_sam3.1'
                        or record.get('source_revision') != summary['source_revision']
                        or record.get('sequence') == 'black_glove_session_040'
                        or record.get('split') != 'train' or not record.get('supervision')):
                    raise ValueError('Unqualified training record')
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
        supervision = cv2.imread(str(self.root / r['supervision']), cv2.IMREAD_GRAYSCALE) if r.get('supervision') else np.full((size,size),255,np.uint8)
        if supervision is None: raise FileNotFoundError(r['supervision'])
        if supervision.shape != (size,size): supervision = cv2.resize(supervision,(size,size),interpolation=cv2.INTER_NEAREST)
        return {
            'has_supervision': bool(r.get('supervision')),
            'supervision': torch.from_numpy((supervision > 127)[...,None].astype(np.float32)),
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
