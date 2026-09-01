import cv2
import numpy as np

from hand_restoration.hugg_aria_conditioning import (
    CropBox,
    build_faithful_condition,
    wrist_sleeve_from_ring,
    paste_crop_with_mask,
    square_crop_from_mask,
    wrist_mask_from_mano,
    wrist_mask_from_ring_points,
)


def synthetic_hand() -> np.ndarray:
    mask = np.zeros((128, 128), dtype=bool)
    mask[42:92, 43:85] = True
    mask[84:127, 36:92] = True
    for x0 in (45, 54, 63, 72, 81):
        mask[12:48, x0 : x0 + 7] = True
    return mask


def test_union_square_crop_contains_both_hands_and_is_scaled() -> None:
    mask = np.zeros((150, 160), dtype=bool)
    mask[20:40, 10:30] = True
    mask[50:80, 70:100] = True
    box = square_crop_from_mask(mask, scale=1.2)
    assert box.width == box.height
    assert box.width >= 108
    assert box.x0 <= 10 and box.x1 >= 100
    assert box.y0 <= 20 and box.y1 >= 80


def test_wrist_rectangle_selects_thicker_endpoint() -> None:
    mask = synthetic_hand()
    wrist, polygons = wrist_mask_from_mano(mask)
    assert wrist.any()
    assert len(polygons) == 1
    assert float(polygons[0][:, 1].mean()) > 90.0
    polygon = polygons[0].astype(np.float32)
    assert float(polygon[:, 1].max()) >= mask.shape[0]
    assert np.allclose(polygon[1] - polygon[0], polygon[2] - polygon[3], atol=2)
    assert np.allclose(polygon[2] - polygon[1], polygon[3] - polygon[0], atol=2)


def test_wrist_sleeve_is_perpendicular_to_arm_and_not_shifted() -> None:
    ring = cv2.boxPoints(((8.0, 64.0), (40.0, 8.0), 0.0))
    palm = np.asarray((8.0, 20.0), dtype=np.float32)
    polygon = wrist_sleeve_from_ring(
        ring,
        palm,
        transverse_scale=1.5,
        forearm_length_ratio=0.75,
        hand_overlap_ratio=0.25,
    )
    # The sleeve remains anchored at x=8, so its left side naturally lies
    # outside the image instead of being translated inward.
    assert int(polygon[:, 0].min()) < 0
    forearm_edge = polygon[[0, 3]].mean(axis=0)
    hand_edge = polygon[[1, 2]].mean(axis=0)
    assert hand_edge[1] < 64 < forearm_edge[1]
    across_wrist = polygon[2] - polygon[1]
    toward_palm = palm - ring.mean(axis=0)
    assert abs(float(np.dot(across_wrist, toward_palm))) < 1e-4


def test_palm_axis_sleeve_edges_align_exactly_with_palm_direction() -> None:
    ring = cv2.boxPoints(((64.0, 64.0), (42.0, 9.0), 27.0))
    palm = np.asarray((102.0, 31.0), dtype=np.float32)
    polygon = wrist_sleeve_from_ring(
        ring, palm, orientation_mode="palm_axis"
    ).astype(np.float64)
    toward_palm = palm - np.asarray((64.0, 64.0))
    toward_palm /= np.linalg.norm(toward_palm)
    longitudinal = polygon[1] - polygon[0]
    longitudinal /= np.linalg.norm(longitudinal)
    transverse = polygon[2] - polygon[1]
    transverse /= np.linalg.norm(transverse)
    assert float(np.dot(longitudinal, toward_palm)) > 0.999
    assert abs(float(np.dot(transverse, toward_palm))) < 0.01


def test_offscreen_or_raster_mismatched_ring_is_rejected() -> None:
    mano = np.zeros((128, 128), dtype=bool)
    mano[40:90, 40:90] = True
    matching = np.repeat(
        cv2.boxPoints(((48.0, 64.0), (12.0, 5.0), 15.0)), 4, axis=0
    )
    offscreen = np.repeat(
        cv2.boxPoints(((-80.0, 64.0), (12.0, 5.0), 15.0)), 4, axis=0
    )
    mismatched = np.repeat(
        cv2.boxPoints(((110.0, 110.0), (12.0, 5.0), 15.0)), 4, axis=0
    )
    palm_points = np.asarray(((60.0, 64.0), (-60.0, 64.0)))
    wrist, polygons = wrist_mask_from_ring_points(
        np.stack((matching, offscreen)),
        palm_points,
        mano.shape,
        mano_mask=mano,
    )
    assert wrist.any()
    assert len(polygons) == 1
    wrist, polygons = wrist_mask_from_ring_points(
        np.stack((matching, mismatched)),
        np.asarray(((60.0, 64.0), (100.0, 100.0))),
        mano.shape,
        mano_mask=mano,
    )
    assert wrist.any()
    assert len(polygons) == 1


def test_faithful_condition_blanks_dilation_and_terminal_wrist() -> None:
    mano = synthetic_hand()
    target = np.full((128, 128, 3), 0.5, dtype=np.float32)
    render = np.zeros_like(target)
    render[..., 0] = 1.0
    alpha = mano.astype(np.float32)
    result = build_faithful_condition(
        target, render, alpha, mano, dilation_px=3
    )
    ring = result.dilated_hand_mask & ~result.mano_mask & ~result.wrist_mask
    visible_hand = result.mano_mask & ~result.wrist_mask
    assert ring.any() and visible_hand.any() and result.wrist_mask.any()
    assert np.all(result.condition_rgb[ring] == 0.0)
    assert np.all(result.condition_rgb[result.wrist_mask] == 0.0)
    assert np.allclose(result.condition_rgb[visible_hand], render[visible_hand])
    assert np.allclose(result.overlay_rgb[~mano], target[~mano])


def test_sam_visible_hand_intersection_preserves_occluding_object() -> None:
    mano = synthetic_hand()
    visible = mano.copy()
    object_occlusion = np.zeros_like(mano)
    object_occlusion[65:82, 52:76] = True
    visible[object_occlusion] = False
    target = np.full((128, 128, 3), 0.4, dtype=np.float32)
    render = np.zeros_like(target)
    render[..., 0] = 1.0
    result = build_faithful_condition(
        target,
        render,
        mano.astype(np.float32),
        mano,
        visible_hand_mask=visible,
        dilation_px=3,
    )
    protected = object_occlusion & mano
    assert np.all(result.sam_excluded_mano_mask[protected])
    assert not np.any(result.visible_hand_mask[protected])
    assert not np.any(result.edit_mask[protected])
    assert np.allclose(result.raw_overlay_rgb[protected], render[protected])
    assert np.allclose(result.overlay_rgb[protected], target[protected])
    assert np.allclose(result.condition_rgb[protected], target[protected])


def test_crop_paste_changes_only_masked_pixels() -> None:
    base = np.zeros((10, 10, 3), dtype=np.float32)
    crop = np.ones((4, 4, 3), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    pasted, full_mask = paste_crop_with_mask(
        base, crop, mask, CropBox(2, 2, 8, 8)
    )
    assert int(full_mask.sum()) == 9
    assert np.all(pasted[full_mask] == 1.0)
    assert np.all(pasted[~full_mask] == 0.0)
