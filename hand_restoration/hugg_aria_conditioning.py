from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CropBox:
    """Exclusive XYXY crop box in one image's pixel coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass(frozen=True)
class FaithfulCondition:
    raw_overlay_rgb: np.ndarray
    overlay_rgb: np.ndarray
    condition_rgb: np.ndarray
    mano_mask: np.ndarray
    visible_hand_mask: np.ndarray
    sam_excluded_mano_mask: np.ndarray
    dilated_hand_mask: np.ndarray
    wrist_mask: np.ndarray
    edit_mask: np.ndarray
    wrist_polygons: tuple[np.ndarray, ...]


def _fit_interval(center: float, length: int, limit: int) -> tuple[int, int]:
    length = min(max(1, int(length)), int(limit))
    start = int(round(center - length / 2.0))
    start = min(max(0, start), limit - length)
    return start, start + length


def square_crop_from_mask(mask: np.ndarray, scale: float = 1.2) -> CropBox:
    """Return one in-bounds square containing the union of all mask components."""

    if mask.ndim != 2:
        raise ValueError("mask must be HW")
    if scale < 1.0:
        raise ValueError("crop scale must be at least 1.0")
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise ValueError("cannot crop an empty MANO alpha mask")
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    side = int(np.ceil(max(width, height) * float(scale)))
    side = min(side, mask.shape[0], mask.shape[1])
    center_x = (float(xs.min()) + float(xs.max()) + 1.0) / 2.0
    center_y = (float(ys.min()) + float(ys.max()) + 1.0) / 2.0
    x0, x1 = _fit_interval(center_x, side, mask.shape[1])
    y0, y1 = _fit_interval(center_y, side, mask.shape[0])
    return CropBox(x0, y0, x1, y1)


def map_crop_box(
    box: CropBox,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> CropBox:
    """Map a crop between aligned images that differ only in resolution."""

    source_h, source_w = source_shape
    target_h, target_w = target_shape
    scale_x = target_w / source_w
    scale_y = target_h / source_h
    if box.width == box.height and abs(scale_x - scale_y) < 1e-9:
        side = int(round(box.width * scale_x))
        center_x = (box.x0 + box.x1) * scale_x / 2.0
        center_y = (box.y0 + box.y1) * scale_y / 2.0
        x0, x1 = _fit_interval(center_x, side, target_w)
        y0, y1 = _fit_interval(center_y, side, target_h)
        return CropBox(x0, y0, x1, y1)
    x0 = int(round(box.x0 * target_w / source_w))
    x1 = int(round(box.x1 * target_w / source_w))
    y0 = int(round(box.y0 * target_h / source_h))
    y1 = int(round(box.y1 * target_h / source_h))
    x0 = min(max(0, x0), target_w - 1)
    y0 = min(max(0, y0), target_h - 1)
    x1 = min(max(x0 + 1, x1), target_w)
    y1 = min(max(y0 + 1, y1), target_h)
    return CropBox(x0, y0, x1, y1)


def crop_resize(
    image: np.ndarray,
    box: CropBox,
    output_size: int,
    interpolation: int,
) -> np.ndarray:
    crop = image[box.y0 : box.y1, box.x0 : box.x1]
    if crop.size == 0:
        raise ValueError(f"empty crop: {box}")
    return cv2.resize(crop, (output_size, output_size), interpolation=interpolation)


def dilate_binary_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if radius_px <= 0:
        return binary.astype(bool)
    diameter = int(radius_px) * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (diameter, diameter)
    )
    return cv2.dilate(binary, kernel).astype(bool)


def _hand_components(mask: np.ndarray, max_hands: int = 2) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    components = []
    ranked = sorted(
        range(1, count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    for label in ranked[:max_hands]:
        if int(stats[label, cv2.CC_STAT_AREA]) >= 16:
            components.append(labels == label)
    return components


def _endpoint_thickness(
    projection: np.ndarray,
    distance_values: np.ndarray,
    low: float,
    high: float,
) -> float:
    selected = (projection >= low) & (projection <= high)
    if not np.any(selected):
        return 0.0
    return float(np.percentile(distance_values[selected], 75))


def _polygon_pixel_coordinates(corners: np.ndarray) -> np.ndarray:
    """Keep polygon geometry in image coordinates; rasterization clips it."""

    return np.rint(corners).astype(np.int32)


def wrist_rectangle_for_component(
    component: np.ndarray,
    length_ratio: float = 0.10,
    width_scale: float = 1.10,
    endpoint_band_ratio: float = 0.18,
    outside_ratio: float = 0.015,
) -> np.ndarray:
    """Estimate a fallback wrist rectangle from a MANO silhouette."""

    ys, xs = np.where(component > 0)
    if xs.size < 16:
        raise ValueError("component is too small for wrist estimation")
    points = np.stack((xs, ys), axis=1).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    transverse = np.asarray((-axis[1], axis[0]), dtype=np.float64)

    longitudinal = (points - center) @ axis
    low, high = float(longitudinal.min()), float(longitudinal.max())
    extent = max(high - low, 1.0)
    distance = cv2.distanceTransform(
        component.astype(np.uint8), cv2.DIST_L2, 5
    )[ys, xs]
    band = max(extent * float(endpoint_band_ratio), 1.0)
    low_thickness = _endpoint_thickness(
        longitudinal, distance, low, low + band
    )
    high_thickness = _endpoint_thickness(
        longitudinal, distance, high - band, high
    )
    if high_thickness > low_thickness:
        axis = -axis
        transverse = -transverse
        longitudinal = (points - center) @ axis
        low, high = float(longitudinal.min()), float(longitudinal.max())
        extent = max(high - low, 1.0)

    wrist_band = longitudinal <= low + max(band, extent * length_ratio)
    cross = (points - center) @ transverse
    local_cross = cross[wrist_band]
    if local_cross.size < 4:
        local_cross = cross
    cross_low, cross_high = np.percentile(local_cross, (2, 98))
    cross_center = (float(cross_low) + float(cross_high)) / 2.0
    half_width = max(
        2.0,
        (float(cross_high) - float(cross_low)) * float(width_scale) / 2.0,
    )
    along_start = low - extent * float(outside_ratio)
    along_end = low + extent * float(length_ratio)
    corners_local = (
        (along_start, cross_center - half_width),
        (along_end, cross_center - half_width),
        (along_end, cross_center + half_width),
        (along_start, cross_center + half_width),
    )
    corners = np.asarray(
        [
            center + along * axis + cross_value * transverse
            for along, cross_value in corners_local
        ],
        dtype=np.float64,
    )
    return _polygon_pixel_coordinates(corners)


def wrist_sleeve_from_ring(
    points: np.ndarray,
    palm_point: np.ndarray,
    orientation_mode: str = "ring_min_area",
    transverse_scale: float = 1.30,
    forearm_length_ratio: float = 0.60,
    hand_overlap_ratio: float = 0.25,
) -> np.ndarray:
    """Build a sleeve cuff anchored at a projected MANO wrist ring.

    ``ring_min_area`` preserves the ring minimum-area frame. ``palm_axis``
    makes the sleeve sides exactly parallel/perpendicular to the projected
    ring-center-to-palm axis while retaining a tight ring-covering extent.
    """

    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    palm_point = np.asarray(palm_point, dtype=np.float64)
    if points.shape[0] < 4:
        raise ValueError("at least four finite wrist-ring points are required")
    if not np.isfinite(palm_point).all():
        raise ValueError("a finite palm anchor is required")
    if orientation_mode not in {"ring_min_area", "palm_axis"}:
        raise ValueError("unknown wrist sleeve orientation mode")
    if transverse_scale < 1.0:
        raise ValueError("wrist sleeve transverse scale must be at least 1.0")
    if forearm_length_ratio <= 0.0 or hand_overlap_ratio < 0.0:
        raise ValueError("wrist sleeve length ratios must be nonnegative")

    minimum_rectangle = cv2.minAreaRect(points)
    center = np.asarray(minimum_rectangle[0], dtype=np.float64)
    palm_direction = palm_point - center
    palm_distance = float(np.linalg.norm(palm_direction))
    if palm_distance < 1e-3:
        raise ValueError("palm anchor coincides with wrist ring")

    if orientation_mode == "ring_min_area":
        minimum_box = cv2.boxPoints(minimum_rectangle).astype(np.float64)
        edge_01 = minimum_box[1] - minimum_box[0]
        edge_12 = minimum_box[2] - minimum_box[1]
        length_01 = float(np.linalg.norm(edge_01))
        length_12 = float(np.linalg.norm(edge_12))
        if min(length_01, length_12) < 1e-3:
            raise ValueError("projected wrist ring is degenerate")
        if length_01 >= length_12:
            transverse = edge_01 / length_01
            ring_width, ring_thickness = length_01, length_12
        else:
            transverse = edge_12 / length_12
            ring_width, ring_thickness = length_12, length_01
        toward_palm = np.asarray((-transverse[1], transverse[0]))
        if float(np.dot(toward_palm, palm_direction)) < 0.0:
            toward_palm = -toward_palm
        cross_center = 0.0
        hand_cover = ring_thickness / 2.0
        forearm_cover = ring_thickness / 2.0
    else:
        toward_palm = palm_direction / palm_distance
        transverse = np.asarray((-toward_palm[1], toward_palm[0]))
        relative = points.astype(np.float64) - center
        cross = relative @ transverse
        along = relative @ toward_palm
        cross_low, cross_high = float(cross.min()), float(cross.max())
        ring_width = cross_high - cross_low
        if ring_width < 1e-3:
            raise ValueError("projected wrist ring is degenerate")
        cross_center = (cross_low + cross_high) / 2.0
        hand_cover = max(float(along.max()), 0.0)
        forearm_cover = max(float(-along.min()), 0.0)

    half_width = ring_width * float(transverse_scale) / 2.0
    hand_extent = max(
        hand_cover, ring_width * float(hand_overlap_ratio)
    )
    forearm_extent = max(
        forearm_cover, ring_width * float(forearm_length_ratio)
    )
    sleeve_axis_center = center + transverse * cross_center
    forearm_edge = sleeve_axis_center - toward_palm * forearm_extent
    hand_edge = sleeve_axis_center + toward_palm * hand_extent
    corners = np.asarray(
        (
            forearm_edge - transverse * half_width,
            hand_edge - transverse * half_width,
            hand_edge + transverse * half_width,
            forearm_edge + transverse * half_width,
        )
    )
    return _polygon_pixel_coordinates(corners)


def wrist_mask_from_ring_points(
    ring_points: np.ndarray,
    palm_points: np.ndarray,
    image_shape: tuple[int, int],
    mano_mask: np.ndarray | None = None,
    orientation_mode: str = "ring_min_area",
    transverse_scale: float = 1.30,
    forearm_length_ratio: float = 0.60,
    hand_overlap_ratio: float = 0.25,
    raster_match_distance_px: float = 6.0,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Rasterize clipped sleeve cuffs for all visible MANO wrist rings."""

    ring_points = np.asarray(ring_points)
    palm_points = np.asarray(palm_points)
    if ring_points.shape[:2] != (2, 16) or ring_points.shape[2] != 2:
        raise ValueError("wrist ring points must have shape [2,16,2]")
    if palm_points.shape != (2, 2):
        raise ValueError("palm points must have shape [2,2]")
    result = np.zeros(image_shape, dtype=np.uint8)
    polygons = []
    distance_to_mano = None
    if mano_mask is not None:
        if mano_mask.shape != image_shape:
            raise ValueError("MANO raster shape does not match wrist image")
        distance_to_mano = cv2.distanceTransform(
            (~(mano_mask > 0)).astype(np.uint8), cv2.DIST_L2, 5
        )
    for points, palm_point in zip(ring_points, palm_points):
        finite = np.isfinite(points).all(axis=1)
        if int(finite.sum()) < 4 or not np.isfinite(palm_point).all():
            continue
        finite_points = points[finite]
        if distance_to_mano is not None:
            clipped_x = np.clip(
                np.rint(finite_points[:, 0]), 0, image_shape[1] - 1
            ).astype(np.int32)
            clipped_y = np.clip(
                np.rint(finite_points[:, 1]), 0, image_shape[0] - 1
            ).astype(np.int32)
            clipped_points = np.stack((clipped_x, clipped_y), axis=1)
            outside_distance = np.linalg.norm(
                finite_points - clipped_points, axis=1
            )
            ring_distance = (
                distance_to_mano[clipped_y, clipped_x] + outside_distance
            )
            if float(ring_distance.min()) > float(raster_match_distance_px):
                continue
        polygon = wrist_sleeve_from_ring(
            finite_points,
            palm_point,
            orientation_mode=orientation_mode,
            transverse_scale=transverse_scale,
            forearm_length_ratio=forearm_length_ratio,
            hand_overlap_ratio=hand_overlap_ratio,
        )
        clipped_mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillConvexPoly(clipped_mask, polygon, 1)
        if not np.any(clipped_mask):
            continue
        result |= clipped_mask
        polygons.append(polygon)
    return result.astype(bool), tuple(polygons)


def wrist_mask_from_mano(
    mano_mask: np.ndarray,
    length_ratio: float = 0.10,
    width_scale: float = 1.10,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    result = np.zeros(mano_mask.shape, dtype=np.uint8)
    polygons = []
    for component in _hand_components(mano_mask, max_hands=2):
        polygon = wrist_rectangle_for_component(
            component,
            length_ratio=length_ratio,
            width_scale=width_scale,
        )
        cv2.fillConvexPoly(result, polygon, 1)
        polygons.append(polygon)
    return result.astype(bool), tuple(polygons)


def build_faithful_condition(
    target_rgb: np.ndarray,
    render_rgb: np.ndarray,
    render_alpha: np.ndarray,
    mano_mask: np.ndarray,
    visible_hand_mask: np.ndarray | None = None,
    wrist_ring_points: np.ndarray | None = None,
    wrist_palm_points: np.ndarray | None = None,
    dilation_px: int = 8,
    wrist_enabled: bool = True,
    wrist_length_ratio: float = 0.10,
    wrist_width_scale: float = 1.10,
    wrist_ring_transverse_scale: float = 1.30,
    wrist_sleeve_orientation: str = "ring_min_area",
    wrist_sleeve_forearm_ratio: float = 0.60,
    wrist_sleeve_hand_overlap_ratio: float = 0.25,
    fill_value: float = 0.0,
    render_opacity: float = 1.0,
) -> FaithfulCondition:
    """Build the paper-style corrupted overlay and explicit edit condition."""

    if target_rgb.shape != render_rgb.shape or target_rgb.ndim != 3:
        raise ValueError("target and render must be matching HWC RGB images")
    if render_alpha.shape != target_rgb.shape[:2]:
        raise ValueError("render alpha shape does not match RGB")
    if mano_mask.shape != target_rgb.shape[:2]:
        raise ValueError("MANO mask shape does not match RGB")
    if (
        visible_hand_mask is not None
        and visible_hand_mask.shape != target_rgb.shape[:2]
    ):
        raise ValueError("visible hand mask shape does not match RGB")
    if not 0.0 <= fill_value <= 1.0:
        raise ValueError("fill value must be in [0,1]")

    mano = mano_mask > 0
    visible = (
        mano
        if visible_hand_mask is None
        else mano & (visible_hand_mask > 0)
    )
    # SAM describes the visible hand. MANO pixels absent from SAM are treated
    # as object-occluded and protected from blanking, overlay, and paste-back.
    sam_excluded_mano = mano & ~visible
    dilated = dilate_binary_mask(visible, dilation_px)
    if wrist_enabled:
        if wrist_ring_points is not None:
            if wrist_palm_points is None:
                raise ValueError("palm anchors are required for wrist rings")
            wrist, polygons = wrist_mask_from_ring_points(
                wrist_ring_points,
                wrist_palm_points,
                mano.shape,
                mano_mask=mano,
                orientation_mode=wrist_sleeve_orientation,
                transverse_scale=wrist_ring_transverse_scale,
                forearm_length_ratio=wrist_sleeve_forearm_ratio,
                hand_overlap_ratio=wrist_sleeve_hand_overlap_ratio,
            )
        else:
            wrist, polygons = wrist_mask_from_mano(
                mano,
                length_ratio=wrist_length_ratio,
                width_scale=wrist_width_scale,
            )
    else:
        wrist = np.zeros_like(mano, dtype=bool)
        polygons = ()
    edit = (dilated & ~sam_excluded_mano) | wrist
    raw_alpha = np.clip(
        render_alpha.astype(np.float32) * float(render_opacity), 0.0, 1.0
    )[..., None]
    alpha = raw_alpha * visible.astype(np.float32)[..., None]
    raw_overlay = target_rgb * (1.0 - raw_alpha) + render_rgb * raw_alpha
    overlay = target_rgb * (1.0 - alpha) + render_rgb * alpha

    condition = target_rgb.copy()
    condition[edit] = float(fill_value)
    condition = condition * (1.0 - alpha) + render_rgb * alpha
    # The wrist is masked after pasting so the renderer's terminal wrist edge is
    # never leaked into the condition.
    condition[wrist] = float(fill_value)
    return FaithfulCondition(
        raw_overlay_rgb=np.clip(raw_overlay, 0.0, 1.0).astype(np.float32),
        overlay_rgb=np.clip(overlay, 0.0, 1.0).astype(np.float32),
        condition_rgb=np.clip(condition, 0.0, 1.0).astype(np.float32),
        mano_mask=mano,
        visible_hand_mask=visible,
        sam_excluded_mano_mask=sam_excluded_mano,
        dilated_hand_mask=dilated,
        wrist_mask=wrist,
        edit_mask=edit,
        wrist_polygons=polygons,
    )


def paste_crop_with_mask(
    base_rgb: np.ndarray,
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    box: CropBox,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a generated crop back and hard-composite it inside its edit mask."""

    resized_rgb = cv2.resize(
        crop_rgb, (box.width, box.height), interpolation=cv2.INTER_CUBIC
    )
    resized_mask = cv2.resize(
        crop_mask.astype(np.uint8),
        (box.width, box.height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    result = base_rgb.copy()
    region = result[box.y0 : box.y1, box.x0 : box.x1]
    region[resized_mask] = resized_rgb[resized_mask]
    full_mask = np.zeros(base_rgb.shape[:2], dtype=bool)
    full_mask[box.y0 : box.y1, box.x0 : box.x1] = resized_mask
    return result, full_mask
