import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hand_restoration.conditions import ConditionConfig
from hand_restoration.derived_dataset import DerivedHandRestorationDataset


class SequenceSplitAndDerivedTests(unittest.TestCase):
    def test_sequence_split_has_no_leakage_without_participant_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definitions = []
            for sequence in range(10):
                definitions.append({"clip_id": f"clip-{len(definitions):06d}", "sequence_id": f"sequence-{sequence}", "device": "Quest3"})
            source = root / "definitions.json"
            output = root / "split.json"
            source.write_text(json.dumps(definitions), encoding="utf-8")
            subprocess.run(["python", "build_hot3d_sequence_split.py", "--clip-definitions", str(source), "--clips-dir", "data/train_quest3", "--output", str(output)], check=True)
            split = json.loads(output.read_text(encoding="utf-8"))
            train_ids = {Path(path).stem for path in split["train"]}
            holdout_ids = {Path(path).stem for path in split["holdout"]}
            train_sequences = {split["clip_metadata"][key]["sequence_id"] for key in train_ids}
            holdout_sequences = {split["clip_metadata"][key]["sequence_id"] for key in holdout_ids}
            self.assertFalse(train_sequences & holdout_sequences)
            self.assertEqual(split["schema_version"], 3)
            self.assertNotIn("train_participants", split["statistics"])
            self.assertEqual(len(holdout_ids), 2)

    def test_derived_tar_reader_builds_masked_replace_at_requested_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_dir = root / "train"
            shard_dir.mkdir()
            key = "clip-000000_000001"
            target = np.full((12, 16), 160, np.uint8)
            mano = np.zeros((12, 16), np.uint8)
            mano[3:9, 5:11] = 220
            mask = np.zeros((12, 16), np.uint8)
            mask[3:9, 5:11] = 255
            with tarfile.open(shard_dir / "train-00000.tar", "w") as archive:
                for suffix, image, extension in (("target.jpg", target, ".jpg"), ("mano.png", mano, ".png"), ("mask.png", mask, ".png")):
                    ok, encoded = cv2.imencode(extension, image)
                    self.assertTrue(ok)
                    data = encoded.tobytes()
                    info = tarfile.TarInfo(f"{key}.{suffix}")
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            record = {"key": key, "shard": "train/train-00000.tar", "clip_id": "clip-000000", "frame_id": "000001", "camera_id": "1201-2", "handedness": "right", "canonical_image_size": [12, 16]}
            manifest = root / "train_manifest.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = DerivedHandRestorationDataset(manifest, output_size=32, condition=ConditionConfig(mode="masked_replace"), seed=7)
            sample = dataset[0]
            self.assertEqual(tuple(sample["target_rgb"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["condition_rgb"].shape), (3, 32, 32))
            self.assertGreater(float(sample["mano_mask"].sum()), 0)
            self.assertEqual(sample["metadata"]["camera_id"], "1201-2")


if __name__ == "__main__":
    unittest.main()
