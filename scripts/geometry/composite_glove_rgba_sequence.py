import argparse
import json
import tarfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from hand_tracking_toolkit import camera as camera_models
from hand_tracking_toolkit.dataset import warp_image
from hot3d.hot3d.clips import clip_util

from scripts.data.export_hot3d_clip_undistorted import build_canonical_camera as build_upright_c1_camera


CANONICAL_CAMERA = {
    "name": "canonical_rgb_c1",
    "width": 1280,
    "height": 720,
    "fx": 563.5018310546875,
    "fy": 563.4933471679688,
    "cx": 642.6881713867188,
    "cy": 355.161865234375,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Composite a Blender RGBA glove image sequence onto a C1 HOT3D video, "
            "while removing pixels covered by object modal masks."
        )
    )
    parser.add_argument(
        "--clip-id",
        default="clip-000000",
        help="Clip id under data/train_quest3 and data/train_quest3_processed.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data") / "train_quest3_processed",
        help="Root directory containing processed per-clip folders.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data") / "train_quest3",
        help="Root directory containing raw clip tar files.",
    )
    parser.add_argument(
        "--clip-tar",
        type=Path,
        default=None,
        help="Optional explicit clip tar path. Defaults to <raw-root>/<clip-id>.tar.",
    )
    parser.add_argument(
        "--base-video",
        type=Path,
        default=None,
        help=(
            "Optional explicit base video path. Defaults to "
            "<processed-root>/<clip-id>/undistorted/<stream-id>_undistorted.mp4."
        ),
    )
    parser.add_argument(
        "--rgba-dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit RGBA image directory. Defaults to "
            "<processed-root>/<clip-id>/blender_renders/<render-subdir>."
        ),
    )
    parser.add_argument(
        "--render-subdir",
        default="left_glove",
        help="Subdirectory under blender_renders used when --rgba-dir is omitted.",
    )
    parser.add_argument(
        "--stream-id",
        choices=("1201-1", "1201-2"),
        default="1201-2",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help=(
            "Optional explicit output video path. Defaults to "
            "<processed-root>/<clip-id>/composite/"
            "<clip-id>_<render-subdir>_composited_<stream-id>.mp4."
        ),
    )
    parser.add_argument(
        "--debug-mask-video",
        type=Path,
        default=None,
        help="Optional debug video path visualizing glove/object/final masks.",
    )
    parser.add_argument(
        "--alpha-scale",
        type=float,
        default=1.0,
        help="Additional multiplier applied to the RGBA alpha channel.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If > 0, composite only the first N RGBA frames for quick testing.",
    )
    args = parser.parse_args()

    clip_dir = args.processed_root / args.clip_id
    if args.clip_tar is None:
        args.clip_tar = args.raw_root / f"{args.clip_id}.tar"
    if args.base_video is None:
        args.base_video = (
            clip_dir / "undistorted" / f"{args.stream_id}_undistorted.mp4"
        )
    if args.rgba_dir is None:
        args.rgba_dir = clip_dir / "blender_renders" / args.render_subdir
    if args.output_video is None:
        args.output_video = (
            clip_dir
            / "composite"
            / f"{args.clip_id}_{args.render_subdir}_composited_{args.stream_id}.mp4"
        )
    return args


def make_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


def build_canonical_camera(src_cam) -> camera_models.PinholePlaneCameraModel:
    return build_upright_c1_camera(src_cam, -90.0)


def combined_modal_mask(
    tar: tarfile.TarFile,
    frame_key: str,
    stream_id: str,
    src_cam,
    height: int,
    width: int,
) -> np.ndarray:
    objects = clip_util.load_object_annotations(tar, frame_key)
    combined = np.zeros((height, width), dtype=np.uint8)
    if objects is None:
        return combined

    dst_cam = build_canonical_camera(src_cam)
    for instance_list in objects.values():
        for instance in instance_list:
            if stream_id not in instance.get("masks_modal", {}):
                continue
            rle = instance["masks_modal"][stream_id]
            if not rle.get("rle"):
                continue
            mask = clip_util.decode_binary_mask_rle(rle).astype(np.uint8) * 255
            warped = warp_image(
                src_camera=src_cam,
                dst_camera=dst_cam,
                src_image=mask,
            )
            if warped.ndim == 3:
                warped = warped[..., 0]
            combined = np.maximum(combined, (warped > 127).astype(np.uint8))
    return combined


def build_debug_frame(
    glove_alpha: np.ndarray, object_mask: np.ndarray, final_mask: np.ndarray
) -> np.ndarray:
    vis = np.zeros((glove_alpha.shape[0], glove_alpha.shape[1], 3), dtype=np.uint8)
    vis[..., 2] = (glove_alpha > 0).astype(np.uint8) * 255
    vis[..., 1] = object_mask.astype(np.uint8) * 255
    vis[..., 0] = final_mask.astype(np.uint8) * 255
    return vis


def main() -> None:
    args = parse_args()

    valid_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"}
    rgba_files = sorted(
        [
            p
            for p in args.rgba_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_suffixes
        ]
    )
    if not rgba_files:
        raise FileNotFoundError(f"No RGBA frames found in {args.rgba_dir}")
    if args.max_frames > 0:
        rgba_files = rgba_files[: args.max_frames]
    print(f"Found {len(rgba_files)} image frames in {args.rgba_dir}")

    base_cap = cv2.VideoCapture(str(args.base_video))
    if not base_cap.isOpened():
        raise RuntimeError(f"Failed to open base video: {args.base_video}")

    fps = base_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = make_writer(args.output_video, width, height, fps)
    debug_video_path = args.debug_mask_video
    debug_writer = None
    if debug_video_path is not None:
        debug_writer = make_writer(debug_video_path, width, height, fps)

    summary = []
    frame_idx = 0
    try:
        with tarfile.open(args.clip_tar, "r") as tar:
            for rgba_path in rgba_files:
                if frame_idx % 10 == 0:
                    print(f"Compositing frame {frame_idx:04d}: {rgba_path.name}")
                ok, base_frame = base_cap.read()
                if not ok:
                    break

                rgba = imageio.imread(rgba_path)
                if rgba.shape[0] != height or rgba.shape[1] != width:
                    raise ValueError(
                        f"RGBA frame {rgba_path.name} has shape {rgba.shape[:2]}, expected {(height, width)}"
                    )
                if rgba.shape[-1] != 4:
                    raise ValueError(f"RGBA frame {rgba_path.name} does not have 4 channels")
                frame_key = f"{frame_idx:06d}"
                cameras, _ = clip_util.load_cameras(tar, frame_key)
                src_cam = cameras[args.stream_id]
                object_mask = combined_modal_mask(
                    tar, frame_key, args.stream_id, src_cam, height, width
                )

                glove_rgb = rgba[..., :3].astype(np.float32)
                glove_alpha = rgba[..., 3].astype(np.float32) / 255.0
                glove_alpha *= float(np.clip(args.alpha_scale, 0.0, 10.0))
                glove_alpha = np.clip(glove_alpha, 0.0, 1.0)

                final_alpha = glove_alpha * (1.0 - object_mask.astype(np.float32))
                final_mask = final_alpha > 0.0

                output = base_frame.astype(np.float32)
                alpha_expanded = final_alpha[..., None]
                output = output * (1.0 - alpha_expanded) + glove_rgb * alpha_expanded
                output = output.astype(np.uint8)

                writer.write(output)
                if debug_writer is not None:
                    debug_writer.write(
                        build_debug_frame(
                            glove_alpha=glove_alpha,
                            object_mask=object_mask,
                            final_mask=final_mask,
                        )
                    )

                summary.append(
                    {
                        "frame_key": frame_key,
                        "rgba_frame": rgba_path.name,
                        "glove_alpha_pixels": int(np.count_nonzero(glove_alpha > 0)),
                        "object_mask_pixels": int(object_mask.sum()),
                        "final_alpha_pixels": int(np.count_nonzero(final_mask)),
                    }
                )
                frame_idx += 1
    finally:
        base_cap.release()
        writer.release()
        if debug_writer is not None:
            debug_writer.release()

    summary_path = args.output_video.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved composited video: {args.output_video.resolve()}")
    if debug_video_path is not None:
        print(f"Saved debug mask video: {debug_video_path.resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
