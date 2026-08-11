import argparse
import json
from pathlib import Path

import numpy as np
import torch

from glove_utils import read_obj
from hot3d_glove_torch_utils import ROOT, load_mano_model_torch
from mano_utils import write_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit MANO shape/pose plus a global similarity transform to a glove "
            "canonical OBJ. This is useful before KNN weight transfer when the glove "
            "shape is a person-specific scan."
        )
    )
    parser.add_argument(
        "--glove-obj",
        type=Path,
        required=True,
    )
    parser.add_argument("--hand", choices=("right", "left"), required=True)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument(
        "--pose-comps",
        type=int,
        default=15,
        help="Number of MANO PCA pose coefficients to optimize. Use 0 to disable pose fitting.",
    )
    parser.add_argument(
        "--finger-weight",
        type=float,
        default=4.0,
        help="Relative loss weight for points near finger regions.",
    )
    parser.add_argument(
        "--palm-weight",
        type=float,
        default=1.0,
        help="Relative loss weight for palm/wrist regions.",
    )
    parser.add_argument(
        "--betas-out",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mano-out",
        type=Path,
        required=True,
        help="Export the fitted MANO canonical mesh after scale+translation alignment.",
    )
    parser.add_argument(
        "--transform-out",
        type=Path,
        default=None,
        help="Optional JSON file for the fitted similarity transform.",
    )
    parser.add_argument(
        "--reference-verts-out",
        type=Path,
        default=None,
        help="Optional .npy file with fitted MANO reference vertices for later correspondence transfer.",
    )
    return parser.parse_args()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / torch.clamp(weights.sum(), min=1e-8)


def axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec) + 1e-12
    axis = rotvec / theta
    x, y, z = axis
    zero = torch.zeros((), dtype=rotvec.dtype, device=rotvec.device)
    k = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    ident = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    outer = axis[:, None] @ axis[None, :]
    return torch.cos(theta) * ident + (1.0 - torch.cos(theta)) * outer + torch.sin(theta) * k


def main() -> None:
    args = parse_args()
    glove_vertices, _ = read_obj(args.glove_obj)
    glove = torch.tensor(glove_vertices, dtype=torch.float32)

    model = load_mano_model_torch(args.hand)
    device = torch.device("cpu")
    shapedirs = model.shapedirs.detach().to(device)
    v_template = model.v_template.detach().to(device)
    lbs_weights = model.lbs_weights.detach().to(device)

    dominant_joint = torch.argmax(lbs_weights, dim=1)
    mano_vertex_importance_base = torch.where(
        dominant_joint > 0,
        torch.full_like(dominant_joint, float(args.finger_weight), dtype=torch.float32),
        torch.full_like(dominant_joint, float(args.palm_weight), dtype=torch.float32),
    )

    betas = torch.zeros((1, 10), dtype=torch.float32, device=device, requires_grad=True)
    pose_coeffs = torch.zeros(
        (1, args.pose_comps), dtype=torch.float32, device=device, requires_grad=True
    )
    rotation = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
    log_scale = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
    translation = torch.zeros((1, 3), dtype=torch.float32, device=device, requires_grad=True)

    params = [betas, rotation, log_scale, translation]
    if args.pose_comps > 0:
        params.append(pose_coeffs)
    optimizer = torch.optim.Adam(params, lr=args.lr)

    best = None
    best_loss = float("inf")

    for step in range(args.iterations):
        optimizer.zero_grad()

        model_output = model(
            betas=betas,
            global_orient=torch.zeros((1, 3), dtype=torch.float32, device=device),
            hand_pose=pose_coeffs if args.pose_comps > 0 else None,
            return_verts=True,
        )
        mano = model_output.vertices
        rotmat = axis_angle_to_matrix(rotation)
        mano = torch.einsum("ij,bmj->bmi", rotmat, mano)
        scale = torch.exp(log_scale).view(1, 1, 1)
        mano_aligned = mano * scale + translation.view(1, 1, 3)

        dists = torch.cdist(glove.unsqueeze(0), mano_aligned, p=2)[0]
        glove_to_mano, glove_to_mano_idx = dists.min(dim=1)
        mano_to_glove, _ = dists.min(dim=0)

        glove_importance = mano_vertex_importance_base[glove_to_mano_idx]
        mano_importance = mano_vertex_importance_base

        loss_glove_to_mano = weighted_mean(glove_to_mano, glove_importance)
        loss_mano_to_glove = weighted_mean(mano_to_glove, mano_importance)
        beta_reg = 1e-3 * (betas ** 2).mean()
        pose_reg = 1e-3 * (pose_coeffs ** 2).mean() if args.pose_comps > 0 else 0.0
        rotation_reg = 1e-4 * (rotation ** 2).mean()
        loss = (
            loss_glove_to_mano
            + loss_mano_to_glove
            + beta_reg
            + pose_reg
            + rotation_reg
        )
        loss.backward()
        optimizer.step()

        current_loss = float(loss.item())
        if current_loss < best_loss:
            best_loss = current_loss
            best = {
                "betas": betas.detach().cpu().numpy().reshape(-1).copy(),
                "pose_coeffs": pose_coeffs.detach().cpu().numpy().reshape(-1).copy(),
                "rotation_axis_angle": rotation.detach().cpu().numpy().reshape(-1).copy(),
                "scale": float(torch.exp(log_scale).item()),
                "translation": translation.detach().cpu().numpy().reshape(-1).copy(),
                "vertices": mano_aligned[0].detach().cpu().numpy().copy(),
            }

        if step % 100 == 0 or step == args.iterations - 1:
            print(
                f"step={step:04d} loss={current_loss:.6f} "
                f"scale={float(torch.exp(log_scale).item()):.6f}"
            )

    args.betas_out.parent.mkdir(parents=True, exist_ok=True)
    args.mano_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.betas_out, best["betas"].astype(np.float32))
    write_obj(args.mano_out, best["vertices"], model.faces.astype(np.int64))
    if args.transform_out is not None:
        args.transform_out.parent.mkdir(parents=True, exist_ok=True)
        args.transform_out.write_text(
            json.dumps(
                {
                    "rotation_axis_angle": best["rotation_axis_angle"].tolist(),
                    "pose_coeffs": best["pose_coeffs"].tolist(),
                    "scale": best["scale"],
                    "translation": best["translation"].tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.reference_verts_out is not None:
        args.reference_verts_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.reference_verts_out, best["vertices"].astype(np.float32))

    print(f"Saved fitted betas: {args.betas_out.resolve()}")
    print(f"Saved fitted aligned MANO mesh: {args.mano_out.resolve()}")
    if args.transform_out is not None:
        print(f"Saved fitted transform: {args.transform_out.resolve()}")
    if args.reference_verts_out is not None:
        print(f"Saved fitted reference verts: {args.reference_verts_out.resolve()}")
    print(f"best loss: {best_loss:.6f}")
    print("betas:", ", ".join(f"{x:.6f}" for x in best["betas"]))
    if args.pose_comps > 0:
        print("pose_coeffs:", ", ".join(f"{x:.6f}" for x in best["pose_coeffs"]))
    print(
        "rotation_axis_angle:",
        ", ".join(f"{x:.6f}" for x in best["rotation_axis_angle"]),
    )
    print(f"scale: {best['scale']:.6f}")
    print(
        "translation:",
        ", ".join(f"{x:.6f}" for x in best["translation"]),
    )


if __name__ == "__main__":
    main()
