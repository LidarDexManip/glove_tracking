import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gaussian_glove.render_aria_overlay_video import wrist_in_camera
from gaussian_hand.model.pose_refinement import PoseRefinement


def test_pose_refinement_supports_full_axis_angle_pose():
    module = PoseRefinement(frame_count=2, pose_dim=45)
    pose = torch.zeros(1, 45)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    translation = torch.zeros(1, 3)
    refined = module(torch.tensor([1]), pose, quaternion, translation)
    assert refined[0].shape == (1, 45)
    assert module.deltas.weight.shape == (2, 51)
    torch.testing.assert_close(refined[0], pose)
    torch.testing.assert_close(refined[1], quaternion)
    torch.testing.assert_close(refined[2], translation)


def test_wrist_camera_conversion_preserves_mano_root_pivot():
    world_rotation = Rotation.from_rotvec([0.2, -0.4, 0.1])
    camera_rotation = Rotation.from_rotvec([-0.3, 0.1, 0.25])
    wrist_translation = np.asarray([0.3, -0.2, 0.6])
    camera_translation = np.asarray([-0.1, 0.05, 0.2])
    root = np.asarray([0.09, 0.007, 0.006])
    point = np.asarray([0.04, -0.03, 0.12])
    quaternion_xyzw = world_rotation.as_quat()
    hand = {
        "wrist_xform": {
            "q_wxyz": quaternion_xyzw[[3, 0, 1, 2]].tolist(),
            "t_xyz": wrist_translation.tolist(),
        }
    }
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = camera_rotation.as_matrix()
    world_to_camera[:3, 3] = camera_translation

    orient, translation = wrist_in_camera(hand, world_to_camera, root)
    expected_world = (
        world_rotation.as_matrix() @ (point - root)
        + root + wrist_translation)
    expected_camera = (
        camera_rotation.as_matrix() @ expected_world + camera_translation)
    actual_camera = (
        Rotation.from_rotvec(orient).as_matrix() @ (point - root)
        + root + translation)

    np.testing.assert_allclose(actual_camera, expected_camera, atol=1e-7)
