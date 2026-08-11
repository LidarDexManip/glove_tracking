import argparse
import json
import tarfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from hot3d.hot3d.clips import clip_util
from hand_tracking_toolkit.dataset import warp_image

from scripts.data.export_hot3d_clip_undistorted import CANONICAL_CAMERA, build_canonical_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode HOT3D clip object masks and export them as PNGs and a preview video."
    )
    parser.add_argument(
        "--clip-tar",
        type=Path,
        default=Path("data") / "train_quest3" / "clip-000000.tar",
    )
    parser.add_argument(
        "--stream-id",
        choices=("1201-1", "1201-2"),
        default="1201-2",
    )
    parser.add_argument(
        "--mask-type",
        choices=("modal", "amodal"),
        default="modal",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data")
        / "train_quest3_processed"
        / "clip-000000"
        / "masks",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--canonical-c1",
        action="store_true",
        help="Warp masks into the same upright canonical C1 camera used by undistorted videos.",
    )
    return parser.parse_args()


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
    mask_key = "masks_modal" if args.mask_type == "modal" else "masks_amodal"
    suffix = f"{args.mask_type}_{args.stream_id}"
    if args.canonical_c1:
        suffix += "_c1"
    png_dir = args.output_dir / suffix
    png_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / f"{suffix}.mp4"

    with tarfile.open(args.clip_tar, "r") as tar:
        frame_keys = sorted(
            name.split(".info.json")[0]
            for name in tar.getnames()
            if name.endswith(".info.json")
        )
        first_cameras, _ = clip_util.load_cameras(tar, frame_keys[0])
        if args.canonical_c1:
            width = int(CANONICAL_CAMERA["width"])
            height = int(CANONICAL_CAMERA["height"])
        else:
            width = int(first_cameras[args.stream_id].width)
            height = int(first_cameras[args.stream_id].height)
        writer = make_writer(video_path, width, height, args.fps)

        summary = []
        try:
            for frame_key in frame_keys:
                cameras, _ = clip_util.load_cameras(tar, frame_key)
                objects = clip_util.load_object_annotations(tar, frame_key)
                combined = np.zeros((height, width), dtype=np.uint8)
                object_count = 0
                if objects is not None:
                    for instance_list in objects.values():
                        for instance in instance_list:
                            if args.stream_id not in instance.get(mask_key, {}):
                                continue
                            rle = instance[mask_key][args.stream_id]
                            if not rle.get("rle"):
                                continue
                            mask = clip_util.decode_binary_mask_rle(rle).astype(np.uint8)
                            if args.canonical_c1:
                                src_cam = cameras[args.stream_id]
                                dst_cam = build_canonical_camera(src_cam, -90.0)
                                mask_rgb = np.stack([mask * 255] * 3, axis=-1)
                                warped = warp_image(
                                    src_camera=src_cam,
                                    dst_camera=dst_cam,
                                    src_image=mask_rgb,
                                ).astype(np.uint8)
                                mask = (warped[..., 0] >= 127).astype(np.uint8)
                            combined = np.maximum(combined, mask * 255)
                            object_count += 1

                png_path = png_dir / f"{frame_key}.png"
                imageio.imwrite(png_path, combined)
                writer.write(cv2.cvtColor(np.stack([combined] * 3, axis=-1), cv2.COLOR_RGB2BGR))
                summary.append(
                    {
                        "frame_key": frame_key,
                        "object_masks_used": object_count,
                        "nonzero_pixels": int(np.count_nonzero(combined)),
                    }
                )
        finally:
            writer.release()

    summary_path = args.output_dir / f"{suffix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved mask PNGs to: {png_dir.resolve()}")
    print(f"Saved mask video to: {video_path.resolve()}")
    print(f"Saved mask summary to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
