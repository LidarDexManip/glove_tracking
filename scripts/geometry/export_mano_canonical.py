import argparse
from pathlib import Path

from mano_utils import (
    ROOT,
    build_canonical_vertices,
    load_faces,
    load_mano_data,
    write_obj,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a canonical MANO mesh as an OBJ file."
    )
    parser.add_argument(
        "--hand",
        choices=("right", "left"),
        default="right",
        help="Which MANO hand model to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output OBJ path. Defaults to canonical_mano_<hand>.obj in the repo root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_data = load_mano_data(args.hand)
    vertices = build_canonical_vertices(model_data)
    faces = load_faces(model_data)

    output_path = args.output or ROOT / f"canonical_mano_{args.hand}.obj"
    write_obj(output_path, vertices, faces)

    print(f"Saved canonical {args.hand} MANO mesh to: {output_path.resolve()}")
    print(f"Vertices: {vertices.shape[0]}")
    print(f"Faces: {faces.shape[0]}")
    print("Definition: pose = 0, betas = 0 (template mesh)")


if __name__ == "__main__":
    main()
