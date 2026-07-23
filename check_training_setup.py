"""Validate the local files and Python environment required for training."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from hand_restoration.data_config import resolve_clip_splits
from hot3d_glove_torch_utils import patch_legacy_dependencies


REQUIRED_IMPORTS = {
    "chumpy": "chumpy",
    "accelerate": "accelerate",
    "cv2": "opencv-python",
    "diffusers": "diffusers",
    "hand_tracking_toolkit": "hand_tracking_toolkit",
    "huggingface_hub": "huggingface_hub",
    "numpy": "numpy",
    "scipy": "scipy",
    "smplx": "smplx",
    "torch": "torch",
    "transformers": "transformers",
    "trimesh": "trimesh",
}


def fail(message: str, failures: list[str]) -> None:
    print(f"[FAIL] {message}")
    failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hand_restoration/tiny_overfit_shaded_3000.json"),
    )
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    failures: list[str] = []
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config

    if not config_path.is_file():
        fail(f"Missing config: {config_path}", failures)
        return 1

    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"[OK] config: {config_path}")

    patch_legacy_dependencies()
    for module_name, package_name in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            fail(f"Cannot import {package_name}: {exc}", failures)
        else:
            print(f"[OK] import: {package_name}")

    hot3d_clip_util = root / "hot3d" / "hot3d" / "clips" / "clip_util.py"
    if hot3d_clip_util.is_file():
        print(f"[OK] HOT3D submodule: {hot3d_clip_util.parents[2]}")
    else:
        fail("HOT3D submodule is absent; run git submodule update --init --recursive", failures)

    data_config = config["data"]
    try:
        train_clips, val_clips, split_path = resolve_clip_splits(config, root)
    except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        fail(f"Invalid data split configuration: {exc}", failures)
        train_clips, val_clips, split_path = [], [], None
    if split_path is not None:
        print(f"[OK] clip split: {split_path}")
    for clip in train_clips + val_clips:
        clip_path = (root / clip).resolve()
        if clip_path.is_file():
            print(f"[OK] HOT3D clip: {clip_path}")
        else:
            fail(f"Missing HOT3D clip: {clip_path}", failures)

    mano_dir = (root / data_config["mano_model_dir"]).resolve()
    configured_hands = str(data_config.get("hands", "right")).lower()
    hands = ("left", "right") if configured_hands in {"both", "all"} else (configured_hands,)
    for hand in hands:
        model_path = mano_dir / f"MANO_{hand.upper()}.pkl"
        if model_path.is_file():
            print(f"[OK] MANO {hand}: {model_path}")
        else:
            fail(f"Missing MANO model: {model_path}", failures)

    try:
        import torch

        print(f"[OK] torch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA: {torch.version.cuda} / {torch.cuda.get_device_name(0)}")
        elif args.require_cuda:
            fail("CUDA was required but torch.cuda.is_available() is False", failures)
        else:
            print("[WARN] CUDA is unavailable; preprocessing works, training will be impractical")
    except Exception:
        pass

    if failures:
        print(f"\nPreflight failed with {len(failures)} issue(s).")
        return 1
    print("\nPreflight passed. The workspace is ready for the smoke test and training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
