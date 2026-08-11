import argparse
import tarfile
from pathlib import Path

import cv2
import numpy as np
import trimesh

from hand_tracking_toolkit import camera as camera_models, rasterizer
from hand_tracking_toolkit.dataset import warp_image
from hot3d.hot3d.clips import clip_util


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
        description="Render glove shell as a white mesh overlay on HOT3D frames warped into canonical C1."
    )
    parser.add_argument(
        "--clip-tar",
        type=Path,
        default=Path("hot3d_data") / "train_quest3" / "clip-000000.tar",
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path("hot3d_glove_sequence_torch"),
    )
    parser.add_argument(
        "--stream-id",
        choices=("1201-1", "1201-2"),
        default="1201-2",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("hot3d_data") / "train_quest3" / "clip-000000_c1_glove_render_1201-2.mp4",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument(
        "--use-rasterizer-rgb",
        action="store_true",
        help="Use the rasterizer's shaded RGB output instead of the default solid white fill.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If > 0, render only the first N exported glove frames.",
    )
    parser.add_argument(
        "--debug-mask-color",
        action="store_true",
        help="Render the glove region with a strong red fill and green contour for debugging.",
    )
    return parser.parse_args()


def read_obj_mesh(path: Path) -> trimesh.Trimesh:
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = [int(part.split("/")[0]) - 1 for part in parts]
                if len(face) == 3:
                    faces.append(face)
                elif len(face) == 4:
                    faces.append([face[0], face[1], face[2]])
                    faces.append([face[0], face[2], face[3]])
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.vertex_normals
    return mesh


def make_canonical_camera(src_cam) -> camera_models.PinholePlaneCameraModel:
    return camera_models.PinholePlaneCameraModel(
        width=CANONICAL_CAMERA["width"],
        height=CANONICAL_CAMERA["height"],
        f=(CANONICAL_CAMERA["fx"], CANONICAL_CAMERA["fy"]),
        c=(CANONICAL_CAMERA["cx"], CANONICAL_CAMERA["cy"]),
        distort_coeffs=[],
        T_world_from_eye=src_cam.T_world_from_eye,
        serial=src_cam.serial,
        label=f"{src_cam.label}_canonical_c1",
    )


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


def main() -> None:
    args = parse_args()
    obj_paths = sorted(args.sequence_dir.glob("glove_right_frame*_sample*.obj"))
    if not obj_paths:
        raise FileNotFoundError(f"No glove OBJ files found in {args.sequence_dir}")

    mesh_by_frame = {}
    for path in obj_paths:
        frame_key = path.stem.split("_frame")[1].split("_sample")[0]
        mesh_by_frame[frame_key] = read_obj_mesh(path)

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(
        args.output_video,
        CANONICAL_CAMERA["width"],
        CANONICAL_CAMERA["height"],
        args.fps,
    )

    try:
        with tarfile.open(args.clip_tar, "r") as tar:
            frame_keys = sorted(mesh_by_frame.keys())
            if args.max_frames > 0:
                frame_keys = frame_keys[: args.max_frames]
            for frame_key in frame_keys:
                cameras, _ = clip_util.load_cameras(tar, frame_key)
                src_cam = cameras[args.stream_id]
                dst_cam = make_canonical_camera(src_cam)
                src_image = clip_util.load_image(tar, frame_key, args.stream_id)
                if src_image.ndim == 2:
                    src_image = np.stack([src_image, src_image, src_image], axis=-1)

                undistorted = warp_image(
                    src_camera=src_cam,
                    dst_camera=dst_cam,
                    src_image=src_image,
                ).astype(np.uint8)

                mesh = mesh_by_frame[frame_key]
                rgb, mask, _ = rasterizer.rasterize_mesh(
                    verts=mesh.vertices,
                    faces=mesh.faces,
                    vert_normals=mesh.vertex_normals,
                    camera=dst_cam,
                )

                if (not args.use_rasterizer_rgb) or rgb.mean() < 5.0:
                    rgb = np.full_like(undistorted, 255, dtype=np.uint8)

                rgb = rgb.astype(np.float32)
                image = undistorted.astype(np.float32)
                overlay = image.copy()
                alpha = float(np.clip(args.alpha, 0.0, 1.0))
                mask_bool = mask.astype(bool)
                if args.debug_mask_color:
                    debug_rgb = image.copy()
                    debug_rgb[..., 0][mask_bool] = 255.0
                    debug_rgb[..., 1][mask_bool] *= 0.2
                    debug_rgb[..., 2][mask_bool] *= 0.2
                    overlay = debug_rgb
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                    )
                    overlay_u8 = overlay.astype(np.uint8)
                    cv2.drawContours(overlay_u8, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
                    overlay = overlay_u8.astype(np.float32)
                else:
                    overlay[mask_bool] = (1.0 - alpha) * image[mask_bool] + alpha * rgb[mask_bool]
                overlay = overlay.astype(np.uint8)

                cv2.putText(
                    overlay,
                    f"{args.stream_id} frame {frame_key}",
                    (24, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    print(f"Saved glove render overlay video: {args.output_video.resolve()}")


if __name__ == "__main__":
    main()
