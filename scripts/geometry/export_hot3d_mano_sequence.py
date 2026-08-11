import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import torch

from hot3d_glove_torch_utils import ROOT, load_mano_model_torch
from mano_utils import write_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a raw HOT3D MANO OBJ sequence for one hand."
    )
    parser.add_argument(
        "--clip-tar",
        type=Path,
        required=True,
    )
    parser.add_argument("--hand", choices=("left", "right"), required=True)
    parser.add_argument("--num-frames", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing OBJ files in the output directory before writing.",
    )
    return parser.parse_args()


def sample_frame_keys(frame_count: int, num_frames: int) -> list[str]:
    sample_count = min(frame_count, num_frames)
    positions = np.linspace(0, frame_count - 1, sample_count)
    return [f"{int(round(float(pos))):06d}" for pos in positions]


def main() -> None:
    args = parse_args()
    model = load_mano_model_torch(args.hand)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        for path in args.output_dir.glob("*.obj"):
            path.unlink()
        manifest_old = args.output_dir / "sequence_manifest.json"
        if manifest_old.exists():
            manifest_old.unlink()

    with tarfile.open(args.clip_tar, mode="r") as tar:
        shape_data = json.load(tar.extractfile("__hand_shapes.json__"))
        betas = torch.tensor(shape_data["mano"], dtype=torch.float32).view(1, -1)
        frame_count = sum(1 for name in tar.getnames() if name.endswith(".info.json"))
        frame_keys = sample_frame_keys(frame_count, args.num_frames)

        manifest = {
            "clip_tar": str(args.clip_tar.resolve()),
            "hand": args.hand,
            "frame_keys": frame_keys,
            "files": [],
        }

        for sample_idx, frame_key in enumerate(frame_keys):
            hands = json.load(tar.extractfile(f"{frame_key}.hands.json"))
            hand_entry = hands[args.hand]["mano_pose"]
            global_orient = torch.tensor(
                hand_entry["wrist_xform"][:3], dtype=torch.float32
            ).view(1, 3)
            translation = torch.tensor(
                hand_entry["wrist_xform"][3:], dtype=torch.float32
            ).view(1, 3)
            hand_pose_coeffs = torch.tensor(
                hand_entry["thetas"], dtype=torch.float32
            ).view(1, -1)

            output = model(
                betas=betas,
                global_orient=global_orient,
                hand_pose=hand_pose_coeffs,
                transl=translation,
                return_verts=True,
            )
            mano_vertices = output.vertices[0].detach().cpu().numpy()

            mano_path = args.output_dir / (
                f"mano_{args.hand}_frame{frame_key}_sample{sample_idx:02d}.obj"
            )
            write_obj(mano_path, mano_vertices, model.faces.astype(np.int64))
            manifest["files"].append(str(mano_path.resolve()))

    manifest_path = args.output_dir / "sequence_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"Saved MANO sequence to: {args.output_dir.resolve()}")
    print(f"frames: {', '.join(frame_keys)}")
    print(f"manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
