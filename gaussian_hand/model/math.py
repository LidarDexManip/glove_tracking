"""Differentiable rotation and real spherical-harmonic helpers."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert (...,3,3) rotation matrices to normalized wxyz quaternions."""
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    q_abs = torch.sqrt(torch.clamp(torch.stack((
        1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
        1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1), min=1e-8))
    candidates = torch.stack((
        torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), -1),
        torch.stack((m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20), -1),
        torch.stack((m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21), -1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()), -1),
    ), -2)
    candidates = candidates / (2 * q_abs[..., None].clamp_min(0.1))
    choice = q_abs.argmax(dim=-1)
    selected = candidates.gather(
        -2, choice[..., None, None].expand(*choice.shape, 1, 4)).squeeze(-2)
    return F.normalize(selected, dim=-1)


def sh_degree2(directions: torch.Tensor) -> torch.Tensor:
    """Real SH basis through degree two, ordered like common 3DGS code."""
    x, y, z = directions.unbind(-1)
    return torch.stack((
        torch.full_like(x, 0.28209479177387814),
        -0.4886025119029199 * y,
        0.4886025119029199 * z,
        -0.4886025119029199 * x,
        1.0925484305920792 * x * y,
        -1.0925484305920792 * y * z,
        0.31539156525252005 * (3 * z.square() - 1),
        -1.0925484305920792 * x * z,
        0.5462742152960396 * (x.square() - y.square()),
    ), -1)
