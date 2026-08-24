#!/usr/bin/env python3
"""Build auditable per-frame prompt and post-inference filtering plans for SAM2."""
from __future__ import annotations

import bisect
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from projectaria_tools.core.calibration import CameraCalibration, FISHEYE624
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from data_loaders.loader_hand_poses import Handedness


LEFT = 0
RIGHT = 1
LEFT_BIT = 1
RIGHT_BIT = 2
RGB_STREAM_ID = "214-1"
POSE_TOLERANCE_NS = 10_000_000


@dataclass(frozen=True)
class PromptBox:
    hand_id: int
    box: np.ndarray
    source: str


@dataclass(frozen=True)
class FramePlan:
    frame_index: int
    timestamp_ns: int
    active_hands: int
    left_prompt: Optional[PromptBox]
    right_prompt: Optional[PromptBox]
    qa_pass: bool
    mano_pose_qa_available: bool
    hand_visible: bool
    good_exposure: bool
    gaussian_valid: bool
    gaussian_left_valid: bool
    gaussian_right_valid: bool
    training_candidate: bool
    training_filter_reason: str

    @property
    def prompts(self) -> dict[int, PromptBox]:
        return {
            item.hand_id: item
            for item in (self.left_prompt, self.right_prompt)
            if item is not None
        }


@dataclass(frozen=True)
class Episode:
    episode_index: int
    start_frame: int
    stop_frame_exclusive: int
    active_hands: int
    prompts: dict[int, PromptBox]


def load_bool_mask(path: Path, stream_id: str = RGB_STREAM_ID) -> dict[int, bool]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {
            int(row["timestamp[ns]"]): row["mask"].strip().lower() == "true"
            for row in csv.DictReader(handle)
            if row["stream_id"] == stream_id
        }


def load_raw_boxes(path: Path, stream_id: str = RGB_STREAM_ID) -> dict[int, dict[int, np.ndarray]]:
    boxes: dict[int, dict[int, np.ndarray]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["stream_id"] != stream_id or not row["x_min[pixel]"].strip():
                continue
            timestamp = int(row["timestamp[ns]"])
            hand = int(row["hand_index"])
            boxes.setdefault(timestamp, {})[hand] = np.asarray(
                [
                    float(row["x_min[pixel]"]), float(row["y_min[pixel]"]),
                    float(row["x_max[pixel]"]), float(row["y_max[pixel]"]),
                ],
                np.float64,
            )
    return boxes


def load_timecode_to_device(path: Path) -> tuple[list[int], list[int]]:
    pairs = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            pairs.append((int(row["timecode_ns"]), int(row["devicetime_ns"])))
    pairs.sort()
    return [item[0] for item in pairs], [item[1] for item in pairs]


def closest_device_time(timestamp: int, timecodes: list[int], device_times: list[int]) -> int:
    position = bisect.bisect_left(timecodes, timestamp)
    choices = [index for index in (position - 1, position) if 0 <= index < len(timecodes)]
    best = min(choices, key=lambda index: abs(timecodes[index] - timestamp))
    return device_times[best]


def load_gaussian_mapping(path: Path) -> dict[int, tuple[bool, bool]]:
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            result[int(row["source_frame_index"])] = (
                row["left_valid"].strip().lower() == "true",
                row["right_valid"].strip().lower() == "true",
            )
    return result


def pose_collection(hand_provider, timestamp: int):
    result = hand_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
        acceptable_time_delta=POSE_TOLERANCE_NS,
    )
    return None if result is None else result.pose3d_collection


def world_to_camera(calibration: dict, headset_provider, timestamp: int):
    result = headset_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
        acceptable_time_delta=POSE_TOLERANCE_NS,
    )
    if result is None:
        return None
    t_world_device = np.asarray(result.pose3d.T_world_device.to_matrix(), np.float64)
    t_device_camera = np.asarray(calibration["T_device_camera"], np.float64)
    return np.linalg.inv(t_world_device @ t_device_camera)


def project_points(vertices: np.ndarray, t_camera_world: np.ndarray,
                   calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    camera = (t_camera_world[:3, :3] @ vertices.T + t_camera_world[:3, 3:4]).T
    fx, fy = (float(value) for value in calibration["focal_lengths"])
    cx, cy = (float(value) for value in calibration["principal_point"])
    z = camera[:, 2]
    uv = np.empty((len(camera), 2), np.float64)
    uv[:, 0] = fx * camera[:, 0] / np.maximum(z, 1e-8) + cx
    uv[:, 1] = fy * camera[:, 1] / np.maximum(z, 1e-8) + cy
    return uv, z


def finish_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int,
               padding: float, min_side: float, min_area: float) -> Optional[np.ndarray]:
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(width - 1.0, x2), min(height - 1.0, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    if x2 - x1 < min_side or y2 - y1 < min_side or (x2 - x1) * (y2 - y1) < min_area:
        return None
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return np.asarray(
        [max(0.0, x1 - dx), max(0.0, y1 - dy),
         min(width - 1.0, x2 + dx), min(height - 1.0, y2 + dy)],
        np.float32,
    )


def mano_box(vertices: np.ndarray, t_camera_world: np.ndarray, calibration: dict,
             padding: float, min_side: float, min_area: float) -> Optional[np.ndarray]:
    uv, z = project_points(vertices, t_camera_world, calibration)
    usable = (z > 0.01) & np.isfinite(uv).all(axis=1)
    if int(usable.sum()) < 20:
        return None
    selected = uv[usable]
    return finish_box(
        float(selected[:, 0].min()), float(selected[:, 1].min()),
        float(selected[:, 0].max()), float(selected[:, 1].max()),
        int(calibration["image_width"]), int(calibration["image_height"]),
        padding, min_side, min_area,
    )


class RawBoxProjector:
    """Reproject annotated FISHEYE624 boxes into each derived LINEAR frame."""

    def __init__(self, support_dir: Path, timecode_mapping: Path, image_width: int,
                 image_height: int) -> None:
        paths = MpsDataPathsProvider(str(support_dir)).get_data_paths()
        self.provider = MpsDataProvider(paths)
        self.timecodes, self.device_times = load_timecode_to_device(timecode_mapping)
        self.width = image_width
        self.height = image_height

    def native_calibration(self, timestamp: int) -> CameraCalibration:
        device_time = closest_device_time(timestamp, self.timecodes, self.device_times)
        record = self.provider.get_online_calibration(
            device_timestamp_ns=device_time,
            time_query_options=TimeQueryOptions.CLOSEST,
        )
        matches = [item for item in record.camera_calibs if item.get_label() == "camera-rgb"]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one camera-rgb calibration, got {len(matches)}")
        source = matches[0]
        return CameraCalibration(
            source.get_label(), FISHEYE624, source.projection_params(),
            source.get_transform_device_camera(), self.width, self.height,
            source.get_valid_radius(), source.get_max_solid_angle(),
            source.get_serial_number(),
        )

    def project(self, raw_box: np.ndarray, timestamp: int, output_calibration: dict,
                padding: float, min_side: float, min_area: float) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = (float(value) for value in raw_box)
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(self.width - 1.0, x2), min(self.height - 1.0, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        sample = np.linspace(0.0, 1.0, 33)
        pixels = []
        for value in sample:
            pixels.extend(((x1 + value * (x2 - x1), y1),
                           (x1 + value * (x2 - x1), y2),
                           (x1, y1 + value * (y2 - y1)),
                           (x2, y1 + value * (y2 - y1))))

        native = self.native_calibration(timestamp)
        r_device_native = np.asarray(
            native.get_transform_device_camera().to_matrix(), np.float64
        )[:3, :3]
        r_device_output = np.asarray(output_calibration["T_device_camera"], np.float64)[:3, :3]
        fx, fy = (float(value) for value in output_calibration["focal_lengths"])
        cx, cy = (float(value) for value in output_calibration["principal_point"])
        projected = []
        for pixel in pixels:
            ray_native = np.asarray(native.unproject_no_checks(np.asarray(pixel, np.float64)))
            ray_output = r_device_output.T @ (r_device_native @ ray_native)
            if ray_output[2] <= 1e-6 or not np.isfinite(ray_output).all():
                continue
            projected.append((fx * ray_output[0] / ray_output[2] + cx,
                              fy * ray_output[1] / ray_output[2] + cy))
        if not projected:
            return None
        uv = np.asarray(projected)
        return finish_box(
            float(uv[:, 0].min()), float(uv[:, 1].min()),
            float(uv[:, 0].max()), float(uv[:, 1].max()),
            int(output_calibration["image_width"]),
            int(output_calibration["image_height"]),
            padding, min_side, min_area,
        )


def build_frame_plan(timestamps: list[int], calibrations: list[dict], masks_dir: Path,
                     raw_boxes_path: Path, support_mps_dir: Path,
                     timecode_mapping_path: Path, gaussian_mapping_path: Path,
                     hand_provider, headset_provider, padding: float,
                     min_side: float, min_area: float) -> list[FramePlan]:
    qa = load_bool_mask(masks_dir / "mask_qa_pass.csv")
    mano_pose_qa = load_bool_mask(masks_dir / "mask_hand_pose_available.csv")
    hand_visible = load_bool_mask(masks_dir / "mask_hand_visible.csv")
    good_exposure = load_bool_mask(masks_dir / "mask_good_exposure.csv")
    raw_boxes = load_raw_boxes(raw_boxes_path)
    gaussian = load_gaussian_mapping(gaussian_mapping_path)
    projector = RawBoxProjector(
        support_mps_dir, timecode_mapping_path,
        int(calibrations[0]["image_width"]), int(calibrations[0]["image_height"]),
    )
    plans = []
    for frame_index, (timestamp, calibration) in enumerate(zip(timestamps, calibrations)):
        qa_pass = qa.get(timestamp, True)
        pose_qa = mano_pose_qa.get(timestamp, True)
        visible = hand_visible.get(timestamp, True)
        exposure = good_exposure.get(timestamp, True)
        gaussian_left, gaussian_right = gaussian.get(frame_index, (False, False))
        collection = pose_collection(hand_provider, timestamp)
        poses = {} if collection is None else collection.poses
        t_camera_world = world_to_camera(calibration, headset_provider, timestamp)
        prompt_items: dict[int, PromptBox] = {}
        for hand_id, handedness in ((LEFT, Handedness.Left), (RIGHT, Handedness.Right)):
            box = None
            source = ""
            pose = poses.get(handedness)
            if pose_qa and pose is not None and t_camera_world is not None:
                vertices = hand_provider.get_hand_mesh_vertices(pose)
                if vertices is not None:
                    box = mano_box(
                        vertices.detach().cpu().numpy(), t_camera_world, calibration,
                        padding, min_side, min_area,
                    )
                    source = "mano"
            if box is None:
                raw = raw_boxes.get(timestamp, {}).get(hand_id)
                if raw is not None:
                    box = projector.project(
                        raw, timestamp, calibration, padding, min_side, min_area
                    )
                    source = "original_bbox"
            if box is not None:
                prompt_items[hand_id] = PromptBox(hand_id, box, source)

        active = 0
        if visible:
            active = LEFT_BIT * int(LEFT in prompt_items) | RIGHT_BIT * int(RIGHT in prompt_items)
        gaussian_valid = gaussian_left or gaussian_right
        reasons = []
        if not gaussian_valid:
            reasons.append("gaussian_invalid")
        if not qa_pass:
            reasons.append("qa_fail")
        if not pose_qa:
            reasons.append("mano_pose_unavailable")
        if not visible:
            reasons.append("hand_not_visible")
        if not exposure:
            reasons.append("bad_exposure")
        plans.append(FramePlan(
            frame_index, timestamp, active,
            prompt_items.get(LEFT), prompt_items.get(RIGHT),
            qa_pass, pose_qa, visible, exposure, gaussian_valid,
            gaussian_left, gaussian_right, not reasons,
            "ok" if not reasons else "+".join(reasons),
        ))
    return plans


def build_episodes(plans: list[FramePlan]) -> list[Episode]:
    episodes = []
    index = 0
    while index < len(plans):
        state = plans[index].active_hands
        if state == 0:
            index += 1
            continue
        stop = index + 1
        while stop < len(plans) and plans[stop].active_hands == state:
            stop += 1
        episodes.append(Episode(
            len(episodes), index, stop, state, plans[index].prompts
        ))
        index = stop
    return episodes


def prompt_json(prompts: dict[int, PromptBox]) -> str:
    return json.dumps(
        {str(hand): {"source": value.source, "box": value.box.tolist()}
         for hand, value in prompts.items()},
        separators=(",", ":"),
    )
