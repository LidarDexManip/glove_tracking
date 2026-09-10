"""Joint depth-sorted rendering of subject-bound hands in one calibrated view."""
from __future__ import annotations

import numpy as np
import torch
from gsplat import rasterization_2dgs

from gaussian_hand.data.geometry import project_points, rasterize_silhouette


def scaled_intrinsics(calibration, size=512):
    sx = size / calibration["image_width"]
    sy = size / calibration["image_height"]
    fx, fy = calibration["focal_lengths"]
    cx, cy = calibration["principal_point"]
    return np.array([[fx*sx, 0, (cx+.5)*sx-.5],
                     [0, fy*sy, (cy+.5)*sy-.5], [0, 0, 1]], np.float32)


@torch.no_grad()
def render_scene(hands, world_to_camera, intrinsics, size=512):
    """hands: (model, PCA pose, wrist quaternion, wrist translation) tuples.

    All surfels enter a single rasterization call: no fixed left/right paste
    order. Output RGB is premultiplied. Only MANO, never SAM, bounds support.
    """
    support = np.zeros((size, size), np.uint8)
    attributes = []
    for model, pca, q, t in hands:
        if not model.strict_surface:
            raise ValueError("Diffusion cache requires strict MANO models")
        a = model.attributes(pca, q, t)
        uv, z = project_points(a["vertices"].cpu().numpy(),
                               world_to_camera.cpu().numpy(), intrinsics.cpu().numpy())
        support |= rasterize_silhouette(uv, z, model.mano.faces.cpu().numpy(), size, size, 1)
        attributes.append(a)
    device = world_to_camera.device
    mesh_mask = torch.as_tensor(support, device=device)[..., None].float()
    if not attributes or not support.any():
        return {"rgb": torch.zeros((size, size, 3), device=device),
                "alpha": torch.zeros_like(mesh_mask), "mesh_mask": mesh_mask}
    scene = {key: torch.cat([a[key] for a in attributes], dim=0)
             for key in ("means", "quats", "scales", "opacities", "colors")}
    rgb, alpha, *_ = rasterization_2dgs(
        means=scene["means"], quats=scene["quats"], scales=scene["scales"],
        opacities=scene["opacities"], colors=scene["colors"][None],
        viewmats=world_to_camera[None], Ks=intrinsics[None], width=size, height=size,
        render_mode="RGB+D", packed=False, distloss=False)
    return {"rgb": rgb[0, ..., :3] * mesh_mask,
            "alpha": alpha[0] * mesh_mask, "mesh_mask": mesh_mask}
