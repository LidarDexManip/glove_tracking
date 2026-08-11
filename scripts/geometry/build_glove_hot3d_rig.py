import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from glove_utils import (
    apply_uniform_alignment,
    compute_uniform_alignment,
    read_obj,
)
from hot3d_glove_torch_utils import (
    ROOT,
    load_mano_model_torch,
    mano_template_data,
)
from mano_utils import rodrigues, write_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a glove rig by transferring MANO weights and blendshapes with KNN."
    )
    parser.add_argument(
        "--glove-obj",
        type=Path,
        default=ROOT / "model" / "Glove_Shell_5_10_1.obj",
    )
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--base-betas-npy",
        type=Path,
        default=None,
        help="Optional .npy file with 10 MANO betas used as the glove template base shape.",
    )
    parser.add_argument(
        "--reference-transform-json",
        type=Path,
        default=None,
        help=(
            "Optional fitted global transform JSON produced by fit_mano_betas_to_glove.py. "
            "If provided, MANO reference vertices and blend directions are rotated/scaled "
            "into glove space before KNN transfer."
        ),
    )
    parser.add_argument(
        "--reference-verts-npy",
        type=Path,
        default=None,
        help=(
            "Optional .npy file with fitted MANO reference vertices. If provided, "
            "these vertices are used for KNN correspondence instead of the canonical "
            "MANO template positions."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "glove_hot3d_rig.npz",
    )
    parser.add_argument(
        "--aligned-obj",
        type=Path,
        default=ROOT / "glove_hot3d_aligned.obj",
    )
    return parser.parse_args()


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(weights, 0.0, None)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def transform_direction_basis(
    basis: np.ndarray, rotation: np.ndarray, scale: float
) -> np.ndarray:
    flat = basis.reshape(-1, 3)
    transformed = (scale * (rotation @ flat.T)).T
    return transformed.reshape(basis.shape)


def main() -> None:
    args = parse_args()
    glove_vertices, glove_faces = read_obj(args.glove_obj)

    mano_model = load_mano_model_torch(args.hand)
    base_betas = np.zeros(10, dtype=np.float32)
    if args.base_betas_npy is not None:
        base_betas = np.load(args.base_betas_npy).astype(np.float32).reshape(-1)
    mano_data = mano_template_data(mano_model, betas=base_betas)

    if args.reference_transform_json is not None:
        transform = json.loads(args.reference_transform_json.read_text(encoding="utf-8"))
        rotation = rodrigues(
            np.asarray(transform["rotation_axis_angle"], dtype=np.float64)
        )
        scale = float(transform["scale"])
        translation = np.asarray(transform["translation"], dtype=np.float64)

        reference_vertices = (
            scale * (rotation @ mano_data["v_template"].T)
        ).T + translation[None, :]
        glove_aligned = glove_vertices
        mano_data["v_template"] = reference_vertices
        mano_data["shapedirs"] = transform_direction_basis(
            mano_data["shapedirs"], rotation, scale
        )
        mano_data["posedirs"] = transform_direction_basis(
            mano_data["posedirs"], rotation, scale
        )
    else:
        scale, translation = compute_uniform_alignment(
            glove_vertices, mano_data["v_template"]
        )
        glove_aligned = apply_uniform_alignment(glove_vertices, scale, translation)

    if args.reference_verts_npy is not None:
        mano_data["v_template"] = np.load(args.reference_verts_npy).astype(np.float64)

    tree = cKDTree(mano_data["v_template"])
    dists, indices = tree.query(glove_aligned, k=args.k)
    if args.k == 1:
        dists = dists[:, None]
        indices = indices[:, None]

    blend = 1.0 / np.maximum(dists, 1e-8)
    blend /= blend.sum(axis=1, keepdims=True)

    glove_weights = np.einsum("nk,nkj->nj", blend, mano_data["weights"][indices])
    glove_weights = normalize_weights(glove_weights)
    glove_shapedirs = np.einsum("nk,nkcd->ncd", blend, mano_data["shapedirs"][indices])
    glove_posedirs = np.einsum("nk,nkcd->ncd", blend, mano_data["posedirs"][indices])

    np.savez(
        args.output,
        glove_template=glove_aligned,
        glove_faces=glove_faces,
        glove_weights=glove_weights,
        glove_shapedirs=glove_shapedirs,
        glove_posedirs=glove_posedirs,
        glove_source_vertices=glove_vertices,
        scale=np.array(scale, dtype=np.float64),
        translation=np.asarray(translation, dtype=np.float64),
        knn_indices=indices,
        knn_distances=dists,
        base_betas=base_betas,
        reference_transform_json=np.array(
            str(args.reference_transform_json) if args.reference_transform_json else "",
            dtype=np.str_,
        ),
        reference_verts_npy=np.array(
            str(args.reference_verts_npy) if args.reference_verts_npy else "",
            dtype=np.str_,
        ),
        hand=np.array(args.hand),
        k=np.array(args.k, dtype=np.int64),
    )
    write_obj(args.aligned_obj, glove_aligned, glove_faces)

    print(f"Saved rig: {args.output.resolve()}")
    print(f"Saved aligned glove: {args.aligned_obj.resolve()}")
    print(f"glove vertices: {glove_aligned.shape[0]}")
    print(f"k: {args.k}")
    print(f"mean knn distance: {float(dists.mean()):.6f}")
    print(f"max knn distance: {float(dists.max()):.6f}")


if __name__ == "__main__":
    main()
