import argparse
import tarfile
from pathlib import Path

import cv2
import numpy as np

from hot3d.hot3d.clips import clip_util


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a simple preview video from one HOT3D clip tar."
    )
    parser.add_argument(
        "--clip-tar",
        type=Path,
        default=Path("hot3d_data") / "train_quest3" / "clip-000000.tar",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hot3d_data") / "train_quest3" / "clip-000000_preview.mp4",
    )
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.clip_tar, mode="r") as tar:
        num_frames = clip_util.get_number_of_frames(tar)
        first_images = []
        for stream_key in ("1201-1", "1201-2"):
            image = clip_util.load_image(tar, "000000", stream_key)
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            image = np.rot90(image, k=3)
            first_images.append(image)
        first_frame = clip_util.stack_images(first_images).astype(np.uint8)

        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {args.output}")

        try:
            writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
            for frame_id in range(1, num_frames):
                frame_key = f"{frame_id:06d}"
                images = []
                for stream_key in ("1201-1", "1201-2"):
                    image = clip_util.load_image(tar, frame_key, stream_key)
                    if image.ndim == 2:
                        image = np.stack([image, image, image], axis=-1)
                    image = np.rot90(image, k=3)
                    images.append(image)
                frame = clip_util.stack_images(images).astype(np.uint8)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    print(f"Saved preview video to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
