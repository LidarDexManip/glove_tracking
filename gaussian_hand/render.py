"""Differentiable gsplat 2DGS renderer."""
from __future__ import annotations

import torch
import numpy as np
from gaussian_hand.data.geometry import project_points, rasterize_silhouette
from gsplat import rasterization_2dgs


def render_gaussian_hand(model, pca: torch.Tensor, wrist_q_wxyz: torch.Tensor,
                         wrist_t_world: torch.Tensor, world_to_camera: torch.Tensor,
                         intrinsics: torch.Tensor, width: int, height: int,
                         background: torch.Tensor | None = None):
    if model.strict_surface and background is not None:
        raise ValueError("Strict surface rendering returns premultiplied RGB; composite the background separately")
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
    result = {"rgb": rgb[0, ..., :3], "alpha": alpha[0], "normals": normals[0],
              "distortion": distort[0], "meta": meta}
    if model.strict_surface:
        # Geometric support only. SAM is never an input to this renderer.
        uv, z = project_points(attributes["vertices"].detach().cpu().numpy(),
                               world_to_camera[0].detach().cpu().numpy(),
                               intrinsics[0].detach().cpu().numpy())
        support = rasterize_silhouette(uv, z, model.mano.faces.cpu().numpy(),
                                      width, height, downsample=1)
        support = torch.from_numpy(support.astype(np.float32)).to(rgb.device)[..., None]
        result["raw_alpha"] = result["alpha"]
        result["mesh_mask"] = support
        result["rgb"] = result["rgb"] * support
        result["alpha"] = result["alpha"] * support
    return result


def composite_over_background(background, rgb, alpha):
    """RGB is premultiplied by alpha by gsplat; do not multiply it twice."""
    return rgb + (1 - alpha) * background
