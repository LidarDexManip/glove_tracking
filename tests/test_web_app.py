import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hand_restoration_web import InferenceApp, build_ui


class WebAppTests(unittest.TestCase):
    def test_derived_preview_and_ui_build_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_dir = root / "holdout"
            shard_dir.mkdir()
            key = "clip-000000_000001"
            gray = np.full((32, 32), 128, np.uint8)
            mask = np.zeros((32, 32), np.uint8)
            mask[8:24, 10:22] = 255
            with tarfile.open(shard_dir / "holdout-00000.tar", "w") as archive:
                for suffix, image, extension in (("target.jpg", gray, ".jpg"), ("mano.png", gray, ".png"), ("mask.png", mask, ".png")):
                    ok, encoded = cv2.imencode(extension, image)
                    self.assertTrue(ok)
                    payload = encoded.tobytes()
                    info = tarfile.TarInfo(f"{key}.{suffix}")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            record = {"key": key, "shard": "holdout/holdout-00000.tar", "clip_id": "clip-000000", "frame_id": "000001", "camera_id": "1201-2", "handedness": "right", "canonical_image_size": [32, 32]}
            for name in ("train_manifest.jsonl", "holdout_manifest.jsonl"):
                (root / name).write_text(json.dumps(record) + "\n", encoding="utf-8")
            config = {"seed": 7, "data": {"format": "derived_webdataset", "train_manifest": str(root / "train_manifest.jsonl"), "val_manifest": str(root / "holdout_manifest.jsonl"), "output_size": 32, "grayscale": True}, "condition": {"mode": "masked_replace", "background_fill": "gray"}, "model": {"base_model_id": "runwayml/stable-diffusion-v1-5", "controlnet_model_id": None}}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "controlnet_final.pt").write_bytes(b"discovery-only")

            app = InferenceApp(config_path, checkpoints, root / "outputs", "cpu")
            preview = app.preview("holdout", 0)

            self.assertEqual(preview[0].shape, (32, 32, 3))
            self.assertEqual(preview[-1]["clip_id"], "clip-000000")
            self.assertIsNotNone(build_ui(app))


if __name__ == "__main__":
    unittest.main()
