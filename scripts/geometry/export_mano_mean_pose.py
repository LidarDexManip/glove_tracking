import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HOT3D_PY_ROOT = ROOT / "hot3d" / "hot3d"

for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HOT3D_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(HOT3D_PY_ROOT))

from data_loaders.hand_common import LANDMARK_INDEX_TO_NAMING, LANDMARK_CONNECTIVITY
from data_loaders.mano_layer import MANOHandModel
from mano_utils import write_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MANO mean-pose mesh and landmarks.")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output OBJ path. Defaults to average_mano_<hand>.obj in repo root.",
    )
    parser.add_argument(
        "--landmarks-output",
        type=Path,
        default=None,
        help="Optional JSON sidecar for MANO landmark positions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or ROOT / f"average_mano_{args.hand}.obj"
    landmarks_output = args.landmarks_output or output_path.with_name(output_path.stem + "_landmarks.json")

    model = MANOHandModel(str(ROOT / "mano_v1_2" / "models"))
    betas = torch.zeros(10, dtype=torch.float32)
    pose = torch.zeros(15, dtype=torch.float32)
    xform = torch.zeros(6, dtype=torch.float32)
    vertices, landmarks = model.forward_kinematics(
        shape_params=betas,
        joint_angles=pose,
        global_xfrom=xform,
        is_right_hand=torch.tensor([args.hand == "right"], dtype=torch.bool),
    )
    vertices = vertices.detach().cpu().numpy().astype(np.float64)
    landmarks = landmarks.detach().cpu().numpy().astype(np.float64)

    faces = (
        model.mano_layer_right.faces.astype(np.int64)
        if args.hand == "right"
        else model.mano_layer_left.faces.astype(np.int64)
    )
    write_obj(output_path, vertices, faces)

    landmark_names = [str(x.value) for x in LANDMARK_INDEX_TO_NAMING]
    payload = {
        "schema_version": 1,
        "source": "mano_mean_pose_export",
        "hand": args.hand,
        "mesh_obj": str(output_path),
        "landmark_names": landmark_names,
        "landmark_positions_m": landmarks.tolist(),
        "connectivity": [[int(a), int(b)] for a, b in LANDMARK_CONNECTIVITY],
    }
    landmarks_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved mean-pose {args.hand} MANO mesh to: {output_path.resolve()}")
    print(f"Saved landmarks sidecar to: {landmarks_output.resolve()}")
    print(f"Vertices: {vertices.shape[0]}")
    print(f"Faces: {faces.shape[0]}")
    print(f"Landmarks: {landmarks.shape[0]}")


if __name__ == "__main__":
    main()
