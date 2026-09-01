import numpy as np

from hand_restoration.adaptive_hand_crops import (
    adaptive_hand_crop_instances,
    components_from_seeds,
)


def ring_at(x: float, y: float) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    return np.stack((x + 3.0 * np.cos(angles), y + 3.0 * np.sin(angles)), axis=1)


def test_far_hands_become_two_independent_1p5_crops() -> None:
    mask = np.zeros((256, 256), dtype=bool)
    mask[80:140, 20:70] = True
    mask[90:150, 185:235] = True
    rings = np.stack((ring_at(45, 125), ring_at(210, 135))).astype(np.float32)

    palms = np.asarray(((45.0, 110.0), (210.0, 120.0)), dtype=np.float32)
    instances = adaptive_hand_crop_instances(mask, rings, palms)

    assert len(instances) == 2
    assert {item.mode for item in instances} == {"split_far_hand"}
    assert {item.scale for item in instances} == {1.5}
    assert {item.handedness for item in instances} == {"left", "right"}
    recovered = [
        components_from_seeds(mask, item.component_seeds_xy)
        for item in instances
    ]
    assert sorted(int(item.sum()) for item in recovered) == [3000, 3000]
    assert instances[0].box.x1 <= instances[1].box.x0


def test_nearby_disconnected_hands_use_one_1p2_union_crop() -> None:
    mask = np.zeros((256, 256), dtype=bool)
    mask[80:140, 65:115] = True
    mask[90:150, 125:175] = True
    rings = np.stack((ring_at(90, 125), ring_at(150, 135))).astype(np.float32)

    palms = np.asarray(((90.0, 110.0), (150.0, 120.0)), dtype=np.float32)
    instances = adaptive_hand_crop_instances(mask, rings, palms)

    assert len(instances) == 1
    assert instances[0].mode == "joint_close_hands"
    assert instances[0].scale == 1.2
    assert instances[0].component_seeds_xy is None


def test_touching_hands_use_one_1p2_union_crop() -> None:
    mask = np.zeros((256, 256), dtype=bool)
    mask[80:145, 65:175] = True
    rings = np.stack((ring_at(90, 125), ring_at(150, 130))).astype(np.float32)

    palms = np.asarray(((90.0, 110.0), (150.0, 115.0)), dtype=np.float32)
    instances = adaptive_hand_crop_instances(mask, rings, palms)

    assert len(instances) == 1
    assert instances[0].mode == "joint_touching_hands"
    assert instances[0].scale == 1.2


def test_single_hand_preserves_baseline_1p5_scale() -> None:
    mask = np.zeros((256, 256), dtype=bool)
    mask[70:150, 80:145] = True
    rings = np.full((2, 16, 2), np.nan, dtype=np.float32)
    rings[1] = ring_at(112, 135)

    palms = np.full((2, 2), np.nan, dtype=np.float32)
    palms[1] = (112.0, 110.0)
    instances = adaptive_hand_crop_instances(mask, rings, palms)

    assert len(instances) == 1
    assert instances[0].mode == "single_hand"
    assert instances[0].scale == 1.5
    assert instances[0].handedness == "right"
