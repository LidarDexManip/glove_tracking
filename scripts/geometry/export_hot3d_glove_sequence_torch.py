import argparse
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch

from hot3d_glove_torch_utils import ROOT, glove_forward_torch, load_mano_model_torch
from mano_utils import write_obj, write_obj_from_template


def find_default_glove_rig(glove_obj: Path, hand: str) -> Path:
    search_roots = [glove_obj.parent, ROOT / "model"]
    candidates: list[Path] = []
    stem_lower = glove_obj.stem.lower()
    hand_token = f"{hand}_rig"
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*.npz"):
            name_lower = path.name.lower()
            if hand_token not in name_lower:
                continue
            score = 0
            if "posefit" in name_lower:
                score += 20
            if "k20" in name_lower:
                score += 10
            if stem_lower in name_lower:
                score += 50
            if glove_obj.parent == path.parent:
                score += 5
            candidates.append((score, len(str(path)), path))
    if not candidates:
        raise FileNotFoundError(
            f"Could not automatically find a {hand} glove rig for template: {glove_obj}"
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a glove OBJ sequence driven by HOT3D MANO annotations."
    )
    parser.add_argument(
        "--clip-id",
        default="clip-000000",
        help="Clip id under data/train_quest3, e.g. clip-000002.",
    )
    parser.add_argument("--hand", choices=("right", "left"), default="left")
    parser.add_argument(
        "--glove-obj",
        type=Path,
        required=True,
        help="Canonical glove OBJ template. Also used to auto-locate the matching rig.",
    )
    parser.add_argument(
        "--clip-tar",
        type=Path,
        default=None,
        help="Optional explicit clip tar path. Defaults to data/train_quest3/<clip-id>.tar.",
    )
    parser.add_argument(
        "--glove-rig",
        type=Path,
        default=None,
        help="Optional explicit glove rig .npz. If omitted, auto-detected from --glove-obj and --hand.",
    )
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit output directory. Defaults to "
            "data/train_quest3_processed/<clip-id>/sequences_3d/glove/<hand>."
        ),
    )
    parser.add_argument(
        "--textured-template-obj",
        type=Path,
        default=None,
        help=(
            "Optional canonical glove OBJ with UV/material information. "
            "If provided, each exported glove frame preserves the template's "
            "mtl/usemtl/vt/vn/f structure and only replaces vertex positions."
        ),
    )
    parser.add_argument(
        "--export-mano",
        action="store_true",
        help="Also export the corresponding MANO mesh from smplx.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing exported OBJ files in the output directory before writing.",
    )
    args = parser.parse_args()
    if args.clip_tar is None:
        args.clip_tar = ROOT / "data" / "train_quest3" / f"{args.clip_id}.tar"
    if args.glove_rig is None:
        args.glove_rig = find_default_glove_rig(args.glove_obj, args.hand)
    if args.output_dir is None:
        args.output_dir = (
            ROOT
            / "data"
            / "train_quest3_processed"
            / args.clip_id
            / "sequences_3d"
            / "glove"
            / args.hand
        )
    if args.textured_template_obj is None:
        args.textured_template_obj = args.glove_obj
    return args


def sample_frame_keys(frame_count: int, num_frames: int) -> list[str]:
    sample_count = min(frame_count, num_frames)
    positions = np.linspace(0, frame_count - 1, sample_count)
    return [f"{int(round(float(pos))):06d}" for pos in positions]


def copy_template_assets(template_obj: Path, output_dir: Path) -> None:
    template_dir = template_obj.parent
    for src in template_dir.iterdir():
        if src.is_file() and src.suffix.lower() != ".obj":
            shutil.copy2(src, output_dir / src.name)


def main() -> None:
    args = parse_args()
    rig = np.load(args.glove_rig, allow_pickle=False)
    mano_output_dir = (
        ROOT
        / "data"
        / "train_quest3_processed"
        / args.clip_id
        / "sequences_3d"
        / "mano"
        / args.hand
    )

    glove_template = torch.tensor(rig["glove_template"], dtype=torch.float32)
    glove_shapedirs = torch.tensor(rig["glove_shapedirs"], dtype=torch.float32)
    glove_posedirs = torch.tensor(rig["glove_posedirs"], dtype=torch.float32)
    glove_weights = torch.tensor(rig["glove_weights"], dtype=torch.float32)
    glove_faces = rig["glove_faces"]
    glove_base_betas = torch.tensor(
        rig["base_betas"] if "base_betas" in rig.files else np.zeros(10, dtype=np.float32),
        dtype=torch.float32,
    ).view(1, -1)

    mano_model = load_mano_model_torch(args.hand)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.export_mano:
        mano_output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        for path in args.output_dir.glob("*.obj"):
            path.unlink()
        for manifest_name in ("sequence_manifest.json", "mano_sequence_manifest.json"):
            manifest_old = args.output_dir / manifest_name
            if manifest_old.exists():
                manifest_old.unlink()
        if args.export_mano:
            for path in mano_output_dir.glob("*.obj"):
                path.unlink()
            manifest_old = mano_output_dir / "sequence_manifest.json"
            if manifest_old.exists():
                manifest_old.unlink()
    if args.textured_template_obj is not None:
        copy_template_assets(args.textured_template_obj, args.output_dir)

    with tarfile.open(args.clip_tar, mode="r") as tar:
        shape_data = json.load(tar.extractfile("__hand_shapes.json__"))
        betas = torch.tensor(shape_data["mano"], dtype=torch.float32).view(1, -1)
        frame_count = sum(1 for name in tar.getnames() if name.endswith(".info.json"))
        frame_keys = sample_frame_keys(frame_count, args.num_frames)

        manifest = {
            "clip_tar": str(args.clip_tar.resolve()),
            "glove_rig": str(args.glove_rig.resolve()),
            "hand": args.hand,
            "frame_keys": frame_keys,
            "files": [],
            "textured_template_obj": (
                str(args.textured_template_obj.resolve())
                if args.textured_template_obj is not None
                else None
            ),
        }
        mano_manifest = {
            "clip_tar": str(args.clip_tar.resolve()),
            "source_glove_rig": str(args.glove_rig.resolve()),
            "hand": args.hand,
            "frame_keys": frame_keys,
            "files": [],
            "output_dir": str(mano_output_dir.resolve()),
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

            glove_vertices, mano_vertices = glove_forward_torch(
                glove_template=glove_template,
                glove_shapedirs=glove_shapedirs,
                glove_posedirs=glove_posedirs,
                glove_weights=glove_weights,
                model=mano_model,
                betas=betas,
                global_orient=global_orient,
                hand_pose_coeffs=hand_pose_coeffs,
                translation=translation,
                glove_base_betas=glove_base_betas,
            )

            glove_path = args.output_dir / (
                f"glove_{args.hand}_frame{frame_key}_sample{sample_idx:02d}.obj"
            )
            if args.textured_template_obj is not None:
                write_obj_from_template(
                    glove_path, glove_vertices, args.textured_template_obj
                )
            else:
                write_obj(glove_path, glove_vertices, glove_faces)
            manifest["files"].append(str(glove_path.resolve()))

            if args.export_mano:
                mano_path = mano_output_dir / (
                    f"mano_{args.hand}_frame{frame_key}_sample{sample_idx:02d}.obj"
                )
                write_obj(mano_path, mano_vertices, mano_model.faces.astype(np.int64))
                mano_manifest["files"].append(str(mano_path.resolve()))

    manifest_path = args.output_dir / "sequence_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    if args.export_mano:
        mano_manifest_path = mano_output_dir / "sequence_manifest.json"
        with mano_manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(mano_manifest, fp, indent=2)

    print(f"Saved sequence to: {args.output_dir.resolve()}")
    if args.export_mano:
        print(f"Saved MANO sequence to: {mano_output_dir.resolve()}")
    print(f"frames: {', '.join(frame_keys)}")
    print(f"manifest: {manifest_path.resolve()}")
    if args.export_mano:
        print(f"mano manifest: {mano_manifest_path.resolve()}")
    print(f"glove rig: {args.glove_rig.resolve()}")


if __name__ == "__main__":
    main()
