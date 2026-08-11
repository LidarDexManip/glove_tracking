---
pretty_name: HUGG ARIA Pinhole
license: other
task_categories:
- video-classification
- object-detection
tags:
- project-aria
- hot3d
- egocentric
- hand-tracking
---

# HUGG ARIA Pinhole

This is a derived, frame-aligned pinhole version of the public
[`LIDAR-GT/HUGG_ARIA`](https://huggingface.co/datasets/LIDAR-GT/HUGG_ARIA)
dataset.

Each RGB frame is generated with the official HOT3D/Project Aria calibration
path:

1. Read stream `214-1` from `recording.vrs` at its exact `TIME_CODE`
   timestamp.
2. Query the per-frame online `FISHEYE624` calibration.
3. Query the per-frame online `LINEAR` calibration.
4. Warp the source image with `projectaria_tools.core.calibration.distort_by_calibration`.

This is pixel-equivalent to `AriaDataProvider.get_undistorted_image()` at the
same timestamp. Every sequence is validated on sampled frames, and the encoded
video is decoded again to verify its frame count.

## Sequence layout

```text
<SEQUENCE>/
  rgb_214_1_pinhole.mp4
  frame_timestamps_214_1.csv
  frame_pinhole_calibration_214_1.jsonl
  headset_trajectory.csv
  mano_hand_pose_trajectory.jsonl
  masks/*.csv
  source_metadata.json
  metadata.json
  _SUCCESS.json
```

`frame_pinhole_calibration_214_1.jsonl` stores the exact target intrinsics and
`T_device_camera` used for every encoded frame. Gaussian/MANO rendering should
use the calibration record with the same `frame_index` and `timestamp_ns`.

No learned hand detector boxes are generated. A hand box can be obtained from
the extrema of valid pixels in a Gaussian/MANO alpha render, which keeps the
box synchronized with the rendered hand prior.

## Video encoding

Videos are H.264, 1408x1408, 30 FPS, `yuv420p`, NVENC CQ 20, with a one-second
GOP. Encoding is lossy; timestamps and camera calibration are lossless text
sidecars.

## License

This derived dataset remains subject to the licenses and terms distributed
with the source HOT3D/HUGG_ARIA data. Per-sequence license files are retained
when available.
