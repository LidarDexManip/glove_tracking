import argparse
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np

from hand_tracking_toolkit import camera as camera_models
from hand_tracking_toolkit.dataset import warp_image
from hot3d.hot3d.clips import clip_util

# Example:
# C:\Users\szeng87\miniforge3\envs\glove-hot3d\python.exe -m scripts.data.export_hot3d_clip_undistorted --stream-id both


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
        description="Convert one HOT3D clip tar into undistorted videos in a canonical pinhole camera."
    )
    parser.add_argument(
        "--clip-tar",
        type=Path,
        default=Path("data") / "train_quest3" / "clip-000000.tar",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data")
        / "train_quest3_processed"
        / "clip-000000"
        / "undistorted",
    )
    parser.add_argument(
        "--stream-id",
        choices=("1201-1", "1201-2", "both"),
        default="both",
        help="Which camera stream to undistort.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--portrait",
        action="store_true",
        help="Rotate the final output video to portrait orientation after warping into C1.",
    )
    parser.add_argument(
        "--upright-roll-deg",
        type=float,
        default=-90.0,
        help=(
            "Rotate the target pinhole camera around its forward axis to make "
            "the content upright while keeping a landscape output canvas."
        ),
    )
    return parser.parse_args()


def stream_list(stream_id: str) -> list[str]:
    if stream_id == "both":
        return ["1201-1", "1201-2"]
    return [stream_id]


def make_writer(path: Path, width: int, height: int, fps: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


def to_three_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    return image


def rotation_z_homogeneous(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    return np.array(
        (
            (c, -s, 0.0, 0.0),
            (s, c, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def build_canonical_camera(
    src_cam, upright_roll_deg: float
) -> camera_models.PinholePlaneCameraModel:
    # Rotate the target camera around its own forward axis so the resulting
    # pinhole image is upright while preserving a landscape output size.
    T_world_from_eye = np.asarray(src_cam.T_world_from_eye, dtype=np.float64)
    T_world_from_eye = T_world_from_eye @ rotation_z_homogeneous(upright_roll_deg)
    return camera_models.PinholePlaneCameraModel(
        width=CANONICAL_CAMERA["width"],
        height=CANONICAL_CAMERA["height"],
        f=(CANONICAL_CAMERA["fx"], CANONICAL_CAMERA["fy"]),
        c=(CANONICAL_CAMERA["cx"], CANONICAL_CAMERA["cy"]),
        distort_coeffs=[],
        T_world_from_eye=T_world_from_eye,
        serial=src_cam.serial,
        label=f"{src_cam.label}_canonical_c1",
    )


def write_metadata(output_dir: Path, portrait: bool, upright_roll_deg: float) -> None:
    metadata = {
        "camera_name": CANONICAL_CAMERA["name"],
        "resolution": [CANONICAL_CAMERA["width"], CANONICAL_CAMERA["height"]],
        "intrinsics": [
            [CANONICAL_CAMERA["fx"], 0.0, CANONICAL_CAMERA["cx"]],
            [0.0, CANONICAL_CAMERA["fy"], CANONICAL_CAMERA["cy"]],
            [0.0, 0.0, 1.0],
        ],
        "distortion_model": "pinhole",
        "distortion_coeffs": [],
        "portrait_output": portrait,
        "upright_roll_deg": upright_roll_deg,
    }
    metadata_path = output_dir / "canonical_camera_c1.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.clip_tar, mode="r") as tar:
        num_frames = clip_util.get_number_of_frames(tar)
        first_frame_key = "000000"
        cameras, _ = clip_util.load_cameras(tar, first_frame_key)

        writers: dict[str, cv2.VideoWriter] = {}
        for sid in stream_list(args.stream_id):
            src_cam = cameras[sid]
            dst_cam = build_canonical_camera(src_cam, args.upright_roll_deg)

            width, height = dst_cam.width, dst_cam.height
            if args.portrait:
                width, height = height, width
            out_path = args.output_dir / f"{sid}_undistorted.mp4"
            writers[sid] = make_writer(out_path, width, height, args.fps)

        try:
            for frame_id in range(num_frames):
                frame_key = f"{frame_id:06d}"
                cameras, _ = clip_util.load_cameras(tar, frame_key)
                for sid in stream_list(args.stream_id):
                    src_image = to_three_channel(
                        clip_util.load_image(tar, frame_key, sid)
                    )
                    dst_cam = build_canonical_camera(
                        cameras[sid], args.upright_roll_deg
                    )
                    undistorted = warp_image(
                        src_camera=cameras[sid],
                        dst_camera=dst_cam,
                        src_image=src_image,
                    ).astype(np.uint8)
                    if args.portrait:
                        undistorted = np.rot90(undistorted, k=3)
                    writers[sid].write(cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR))
        finally:
            for writer in writers.values():
                writer.release()

    write_metadata(args.output_dir, args.portrait, args.upright_roll_deg)
    for sid in stream_list(args.stream_id):
        print(f"Saved undistorted video: {(args.output_dir / f'{sid}_undistorted.mp4').resolve()}")
    print(f"Saved canonical camera metadata: {(args.output_dir / 'canonical_camera_c1.json').resolve()}")


if __name__ == "__main__":
    main()
