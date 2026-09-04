"""Differentiable MANO with a learnable canonical vertex correction."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

for _name, _value in {
    "bool": bool, "int": int, "float": float, "complex": complex,
    "object": object, "unicode": str, "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

import smplx
from smplx.lbs import lbs


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized wxyz quaternions to axis-angle."""
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    xyz, w = quaternion[..., 1:], quaternion[..., :1]
    xyz_norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, w)
    scale = torch.where(xyz_norm > 1e-8, angle / xyz_norm,
                        2.0 + xyz_norm.square() / 3.0)
    return xyz * scale


class LearnableMANO(nn.Module):
    """MANO LBS whose template correction participates in every posed mesh."""

    def __init__(self, model_path: str, num_pca_comps: int = 15) -> None:
        super().__init__()
        self.layer = smplx.MANO(
            model_path=model_path, is_rhand=True, use_pca=True,
            num_pca_comps=num_pca_comps, flat_hand_mean=False,
        )
        self.layer.requires_grad_(False)
        self.canonical_delta = nn.Parameter(torch.zeros_like(self.layer.v_template))
        self.register_buffer("faces", torch.as_tensor(
            np.asarray(self.layer.faces, np.int64), dtype=torch.long))

    def forward(self, pca: torch.Tensor, wrist_q_wxyz: torch.Tensor,
                wrist_t_world: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        if pca.ndim == 1:
            pca = pca[None]
            wrist_q_wxyz = wrist_q_wxyz[None]
            wrist_t_world = wrist_t_world[None]
            betas = betas[None]
        global_orient = quaternion_to_axis_angle(wrist_q_wxyz)
        hand_pose = torch.einsum("bi,ij->bj", pca, self.layer.hand_components)
        full_pose = torch.cat((global_orient, hand_pose), dim=-1)
        full_pose = full_pose + self.layer.pose_mean
        vertices, _ = lbs(
            betas, full_pose, self.layer.v_template + self.canonical_delta,
            self.layer.shapedirs, self.layer.posedirs, self.layer.J_regressor,
            self.layer.parents, self.layer.lbs_weights, pose2rot=True)
        return vertices + wrist_t_world[:, None, :]

    def canonical(self, betas: torch.Tensor) -> torch.Tensor:
        batch = 1 if betas.ndim == 1 else len(betas)
        device, dtype = betas.device, betas.dtype
        zeros_pose = torch.zeros((batch, 15), device=device, dtype=dtype)
        identity = torch.zeros((batch, 4), device=device, dtype=dtype)
        identity[:, 0] = 1
        zeros_t = torch.zeros((batch, 3), device=device, dtype=dtype)
        return self.forward(zeros_pose, identity, zeros_t, betas)
