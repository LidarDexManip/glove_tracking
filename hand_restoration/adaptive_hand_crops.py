from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .hugg_aria_conditioning import CropBox, square_crop_from_mask


HAND_NAMES = ("left", "right")


@dataclass(frozen=True)
class AdaptiveCropInstance:
    """One deterministic training crop derived from a union MANO raster."""

    box: CropBox
    mode: str
    scale: float
    component_seeds_xy: tuple[tuple[int, int], ...] | None
    handedness: str
    component_area: int
    component_count: int


def components_from_seeds(
    mask: np.ndarray,
    seeds_xy: tuple[tuple[int, int], ...] | list[list[int]],
) -> np.ndarray:
    """Recover a union of connected components from foreground seeds."""

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask must be HW")
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    selected = set()
    for seed_xy in seeds_xy:
        x, y = (int(value) for value in seed_xy)
        if not (0 <= x < binary.shape[1] and 0 <= y < binary.shape[0]):
            raise ValueError(f"component seed is outside the mask: {(x, y)}")
        label = int(labels[y, x])
        if count <= 1 or label == 0:
            raise ValueError(f"component seed is not foreground: {(x, y)}")
        selected.add(label)
    if not selected:
        raise ValueError("at least one component seed is required")
    return np.isin(labels, list(selected))


def component_from_seed(
    mask: np.ndarray, seed_xy: tuple[int, int]
) -> np.ndarray:
    """Backward-compatible single-component wrapper."""

    return components_from_seeds(mask, (seed_xy,))


def _box_intersects(first: CropBox, second: CropBox) -> bool:
    return (
        min(first.x1, second.x1) > max(first.x0, second.x0)
        and min(first.y1, second.y1) > max(first.y0, second.y0)
    )


def _component_seed(component: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(component)
    if xs.size == 0:
        raise ValueError("cannot select a seed from an empty component")
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    closest = int(np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
    return int(xs[closest]), int(ys[closest])


def _ring_matches_component(
    component: np.ndarray,
    ring_points: np.ndarray | None,
    max_distance_px: float,
) -> list[int]:
    if ring_points is None:
        return []
    points = np.asarray(ring_points, dtype=np.float32)
    if points.shape != (2, 16, 2):
        raise ValueError("wrist ring points must have shape [2,16,2]")
    distance = cv2.distanceTransform(
        (~component).astype(np.uint8), cv2.DIST_L2, 5
    )
    height, width = component.shape
    matched = []
    for hand_index, hand_points in enumerate(points):
        finite = np.isfinite(hand_points).all(axis=1)
        if not finite.any():
            continue
        visible = hand_points[finite]
        clipped_x = np.clip(np.rint(visible[:, 0]), 0, width - 1).astype(np.int32)
        clipped_y = np.clip(np.rint(visible[:, 1]), 0, height - 1).astype(np.int32)
        clipped = np.stack((clipped_x, clipped_y), axis=1).astype(np.float32)
        outside_distance = np.linalg.norm(visible - clipped, axis=1)
        raster_distance = distance[clipped_y, clipped_x] + outside_distance
        if float(raster_distance.min()) <= float(max_distance_px):
            matched.append(hand_index)
    return matched


def _raw_components(binary: np.ndarray) -> tuple[int, list[tuple[int, int, np.ndarray, np.ndarray]]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        components.append((label, area, labels == label, centroids[label]))
    return count - 1, components


def _group_components_by_palm(
    binary: np.ndarray,
    palm_points: np.ndarray | None,
    min_component_area: int,
) -> list[tuple[int, np.ndarray, tuple[tuple[int, int], ...]]]:
    """Assign every visible alpha fragment to its nearest left/right palm."""

    if palm_points is None:
        return []
    palms = np.asarray(palm_points, dtype=np.float32)
    if palms.shape != (2, 2):
        raise ValueError("wrist palm points must have shape [2,2]")
    valid_hand_indices = np.flatnonzero(np.isfinite(palms).all(axis=1))
    if valid_hand_indices.size == 0:
        return []
    _, components = _raw_components(binary)
    grouped_masks = [np.zeros_like(binary), np.zeros_like(binary)]
    grouped_seeds: list[list[tuple[int, int]]] = [[], []]
    for _, _, component, centroid in components:
        local_index = int(
            np.argmin(
                np.linalg.norm(palms[valid_hand_indices] - centroid, axis=1)
            )
        )
        hand_index = int(valid_hand_indices[local_index])
        grouped_masks[hand_index] |= component
        grouped_seeds[hand_index].append(_component_seed(component))
    groups = []
    for hand_index, (group, seeds) in enumerate(zip(grouped_masks, grouped_seeds)):
        area = int(group.sum())
        if area >= int(min_component_area):
            groups.append((hand_index, group, tuple(seeds)))
    return groups


def adaptive_hand_crop_instances(
    mano_mask: np.ndarray,
    wrist_ring_points: np.ndarray | None,
    wrist_palm_points: np.ndarray | None = None,
    *,
    single_hand_scale: float = 1.5,
    close_joint_scale: float = 1.2,
    min_component_area: int = 64,
    ring_match_distance_px: float = 6.0,
) -> list[AdaptiveCropInstance]:
    """Split far hands and retain one tighter crop for nearby hands.

    Visible alpha fragments are first assigned to the nearest projected left or
    right proximal-palm anchor. Two hands are considered far only when their
    individual 1.5x square crops are disjoint. Nearby or touching hands use one
    1.2x union crop. Single visible hands retain the baseline 1.5x crop.
    """

    binary = np.asarray(mano_mask) > 0
    if binary.ndim != 2:
        raise ValueError("mano_mask must be HW")
    if not binary.any():
        raise ValueError("cannot crop an empty MANO mask")
    if single_hand_scale < 1.0 or close_joint_scale < 1.0:
        raise ValueError("crop scales must be at least 1.0")
    if min_component_area <= 0:
        raise ValueError("min_component_area must be positive")

    raw_component_count, components = _raw_components(binary)
    hand_groups = _group_components_by_palm(
        binary, wrist_palm_points, min_component_area
    )
    if len(hand_groups) == 2:
        per_hand_boxes = [
            square_crop_from_mask(group, scale=single_hand_scale)
            for _, group, _ in hand_groups
        ]
        if not _box_intersects(per_hand_boxes[0], per_hand_boxes[1]):
            return [
                AdaptiveCropInstance(
                    box=box,
                    mode="split_far_hand",
                    scale=float(single_hand_scale),
                    component_seeds_xy=seeds,
                    handedness=HAND_NAMES[hand_index],
                    component_area=int(group.sum()),
                    component_count=raw_component_count,
                )
                for box, (hand_index, group, seeds) in zip(
                    per_hand_boxes, hand_groups
                )
            ]
        return [
            AdaptiveCropInstance(
                box=square_crop_from_mask(binary, scale=close_joint_scale),
                mode="joint_close_hands",
                scale=float(close_joint_scale),
                component_seeds_xy=None,
                handedness="both",
                component_area=int(binary.sum()),
                component_count=raw_component_count,
            )
        ]

    if len(hand_groups) == 1:
        hand_index, group, seeds = hand_groups[0]
        matching_hands = _ring_matches_component(
            binary, wrist_ring_points, ring_match_distance_px
        )
        if len(matching_hands) == 2:
            return [
                AdaptiveCropInstance(
                    box=square_crop_from_mask(binary, scale=close_joint_scale),
                    mode="joint_touching_hands",
                    scale=float(close_joint_scale),
                    component_seeds_xy=None,
                    handedness="both",
                    component_area=int(binary.sum()),
                    component_count=raw_component_count,
                )
            ]
        return [
            AdaptiveCropInstance(
                box=square_crop_from_mask(group, scale=single_hand_scale),
                mode="single_hand",
                scale=float(single_hand_scale),
                component_seeds_xy=seeds,
                handedness=HAND_NAMES[hand_index],
                component_area=int(group.sum()),
                component_count=raw_component_count,
            )
        ]

    # Fallback for missing/offscreen palm anchors: preserve the frame and use
    # connected components only when exactly two substantial pieces exist.
    substantial = [item for item in components if item[1] >= min_component_area]
    substantial.sort(key=lambda item: item[1], reverse=True)
    if len(substantial) == 2:
        boxes = [
            square_crop_from_mask(component, scale=single_hand_scale)
            for _, _, component, _ in substantial
        ]
        if not _box_intersects(boxes[0], boxes[1]):
            return [
                AdaptiveCropInstance(
                    box=box,
                    mode="split_far_hand",
                    scale=float(single_hand_scale),
                    component_seeds_xy=(_component_seed(component),),
                    handedness="unknown",
                    component_area=area,
                    component_count=raw_component_count,
                )
                for box, (_, area, component, _) in zip(boxes, substantial)
            ]
    matching_hands = _ring_matches_component(
        binary, wrist_ring_points, ring_match_distance_px
    )
    joint = len(matching_hands) == 2
    scale = close_joint_scale if joint else single_hand_scale
    return [
        AdaptiveCropInstance(
            box=square_crop_from_mask(binary, scale=scale),
            mode=("joint_fragmented_hands" if joint else "single_fragmented"),
            scale=float(scale),
            component_seeds_xy=None,
            handedness=("both" if joint else "unknown"),
            component_area=int(binary.sum()),
            component_count=raw_component_count,
        )
    ]
