import argparse
from pathlib import Path

import numpy as np

from glove_utils import (
    apply_uniform_alignment,
    compute_uniform_alignment,
    read_obj,
    transfer_weights_knn,
)
from mano_utils import (
    ROOT,
    build_canonical_vertices,
    load_faces,
    load_mano_data,
    load_skinning_weights,
    write_obj,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer MANO skinning weights to a canonical glove mesh using KNN."
    )
    parser.add_argument(
        "--glove-obj",
        type=Path,
        default=ROOT / "model" / "Glove_Shell_5_10_1.obj",
        help="Canonical glove OBJ path.",
    )
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="How many MANO vertices to average for each glove vertex.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "glove_rig_knn.npz",
        help="Output .npz with aligned glove template and transferred weights.",
    )
    parser.add_argument(
        "--aligned-obj",
        type=Path,
        default=ROOT / "glove_canonical_aligned.obj",
        help="Optional OBJ export for the aligned glove template.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    glove_vertices, glove_faces = read_obj(args.glove_obj)
    model_data = load_mano_data(args.hand)
    mano_vertices = build_canonical_vertices(model_data)
    mano_faces = load_faces(model_data)
    mano_weights = load_skinning_weights(model_data)

    scale, translation = compute_uniform_alignment(glove_vertices, mano_vertices)
    glove_vertices_aligned = apply_uniform_alignment(glove_vertices, scale, translation)
    glove_weights, knn_indices, knn_dists = transfer_weights_knn(
        glove_vertices_aligned,
        mano_vertices,
        mano_weights,
        k=args.k,
    )

    np.savez(
        args.output,
        glove_vertices=glove_vertices_aligned,
        glove_faces=glove_faces,
        glove_weights=glove_weights,
        glove_source_vertices=glove_vertices,
        scale=np.array(scale, dtype=np.float64),
        translation=np.asarray(translation, dtype=np.float64),
        knn_indices=knn_indices,
        knn_distances=knn_dists,
        mano_vertices=mano_vertices,
        mano_faces=mano_faces,
        hand=np.array(args.hand),
        k=np.array(args.k, dtype=np.int64),
    )
    write_obj(args.aligned_obj, glove_vertices_aligned, glove_faces)

    print(f"Saved glove rig to: {args.output.resolve()}")
    print(f"Saved aligned glove template to: {args.aligned_obj.resolve()}")
    print(f"glove vertices: {glove_vertices.shape[0]}")
    print(f"MANO vertices: {mano_vertices.shape[0]}")
    print(f"k: {args.k}")
    print(f"alignment scale: {scale:.6f}")
    print(f"mean knn distance: {float(knn_dists.mean()):.6f}")
    print(f"max knn distance: {float(knn_dists.max()):.6f}")


if __name__ == "__main__":
    main()
