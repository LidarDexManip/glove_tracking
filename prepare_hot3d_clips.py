"""Download and validate the fixed HOT3D-Clips 000000--000019 dataset."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from collections import Counter

from huggingface_hub import hf_hub_download
from hot3d.hot3d.clips import clip_util

DEFAULT_MANIFEST = Path("configs/hand_restoration/splits/clips_000000_000019_seed7.json")


def clip_statistics(path: Path, hand: str = "right") -> dict:
    with tarfile.open(path, "r") as tar:
        names = tar.getnames()
        if "__hand_shapes.json__" not in names:
            raise RuntimeError("missing __hand_shapes.json__")
        keys = sorted(name.removesuffix(".info.json") for name in names if name.endswith(".info.json"))
        annotated = 0
        usable = 0
        for key in keys:
            hands = clip_util.load_hand_annotations(tar, key)
            has_mano = hands is not None and hand in hands and "mano_pose" in hands[hand]
            annotated += int(has_mano)
            required = [f"{key}.image_1201-2.jpg", f"{key}.cameras.json"]
            usable += int(has_mano and all(name in names for name in required))
    return {
        "bytes": path.stat().st_size,
        "total_frames": len(keys),
        "right_mano_frames": annotated,
        "final_samples": usable,
        "tar_readable": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--download", action="store_true", help="Download missing official HOT3D-Clips tar files from Hugging Face.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repo-id", default="bop-benchmark/hot3d")
    parser.add_argument("--subset", default="train_quest3")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = [Path(path) for path in manifest["train"] + manifest["holdout"]]
    failures = []
    stats = {}
    print("clip\tstatus\tMiB\ttotal\tright_mano\tannotation_candidates")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.download and (args.force or not path.exists()):
            downloaded = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=f"{args.subset}/{path.name}", force_download=args.force)
            shutil.copy2(downloaded, path)
        if not path.exists():
            stats[path.stem] = {"status": "missing", "bytes": None, "total_frames": None, "right_mano_frames": None, "final_samples": None, "tar_readable": False}
            failures.append(f"missing: {path}")
            print(f"{path.stem}\tMISSING\t-\t-\t-\t-")
            continue
        try:
            item = clip_statistics(path)
        except (tarfile.TarError, OSError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            failures.append(f"invalid: {path}: {exc}")
            print(f"{path.stem}\tINVALID\t{path.stat().st_size / 1024**2:.1f}\t-\t-\t-")
            continue
        stats[path.stem] = item
        status = "OK" if item["final_samples"] else "NO_RIGHT_HAND"
        if not item["final_samples"]:
            failures.append(f"no usable right-hand samples: {path}")
        print(f"{path.stem}\t{status}\t{item['bytes'] / 1024**2:.1f}\t{item['total_frames']}\t{item['right_mano_frames']}\t{item['final_samples']}")
    if not failures:
        from hand_restoration.config import load_json_config
        from hand_restoration.data_config import resolve_clip_splits
        from train_hand_restorer import make_dataset

        training_config = load_json_config("configs/hand_restoration/train_clips000000_000019.json")
        train_clips, holdout_clips, _ = resolve_clip_splits(training_config, Path.cwd())
        for clip_paths in (train_clips, holdout_clips):
            dataset = make_dataset(training_config, clip_paths)
            visible_counts = Counter(path.stem for path, _ in dataset.samples)
            for clip_path in clip_paths:
                stem = Path(clip_path).stem
                stats[stem]["c1_visible_right_frames"] = visible_counts[stem]
                stats[stem]["final_samples"] = visible_counts[stem]

    if not failures:
        print("\nclip\tc1_visible_final_samples")
        for path in paths:
            print(f"{path.stem}\t{stats[path.stem]['final_samples']}")

    manifest["statistics"] = {
        "status": "complete" if not failures else "incomplete",
        "camera_id": "1201-2",
        "hands": "right",
        "frame_stride": 1,
        "clips": stats,
        "train_samples": sum(stats.get(Path(path).stem, {}).get("final_samples", 0) or 0 for path in manifest["train"]),
        "holdout_samples": sum(stats.get(Path(path).stem, {}).get("final_samples", 0) or 0 for path in manifest["holdout"]),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
