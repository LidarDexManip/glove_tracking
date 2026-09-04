import torch

from gaussian_hand.model.math import matrix_to_quaternion
from gaussian_hand.model.pose_refinement import axis_angle_to_quaternion


def test_identity_matrix_quaternion_has_finite_gradient():
    matrix = torch.eye(3, requires_grad=True)
    quaternion = matrix_to_quaternion(matrix)
    quaternion.sum().backward()
    assert torch.isfinite(quaternion).all()
    assert torch.isfinite(matrix.grad).all()
    assert torch.allclose(quaternion, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_zero_axis_angle_quaternion_has_finite_gradient():
    value = torch.zeros(3, requires_grad=True)
    quaternion = axis_angle_to_quaternion(value)
    quaternion.sum().backward()
    assert torch.isfinite(value.grad).all()
    assert torch.allclose(quaternion, torch.tensor([1.0, 0.0, 0.0, 0.0]))
