"""Differentiable gsplat 2DGS renderer."""
from __future__ import annotations

import torch
from gsplat import rasterization_2dgs


def render_gaussian_hand(model, pca: torch.Tensor, wrist_q_wxyz: torch.Tensor,
                         wrist_t_world: torch.Tensor, world_to_camera: torch.Tensor,
                         intrinsics: torch.Tensor, width: int, height: int,
                         background: torch.Tensor | None = None):
    attributes = model.attributes(pca, wrist_q_wxyz, wrist_t_world)
    if world_to_camera.ndim == 2:
        world_to_camera = world_to_camera[None]
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None]
    colors = attributes["colors"][None]
    backgrounds = None if background is None else background.reshape(1, 3)
    rgb, alpha, normals, normal_from_depth, distort, median, meta = rasterization_2dgs(
        means=attributes["means"], quats=attributes["quats"],
        scales=attributes["scales"], opacities=attributes["opacities"],
        colors=colors, viewmats=world_to_camera, Ks=intrinsics,
        width=width, height=height, backgrounds=backgrounds,
        render_mode="RGB+D", packed=False, distloss=True,
    )
    return {"rgb": rgb[0, ..., :3], "alpha": alpha[0], "normals": normals[0],
            "distortion": distort[0], "meta": meta}
