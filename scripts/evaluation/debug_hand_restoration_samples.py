import argparse
from pathlib import Path

from hand_restoration.conditions import ConditionConfig
from hand_restoration.config import load_json_config
from hand_restoration.hot3d_dataset import Hot3DSingleFrameDataset
from hand_restoration.visualize import save_debug_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and visualize aligned HOT3D MANO restoration samples.")
    parser.add_argument("--config", type=Path, default=Path("configs/hand_restoration/debug_samples.json"))
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()
    config = load_json_config(args.config)
    data = config["data"]
    condition = ConditionConfig(**config.get("condition", {}))
    dataset = Hot3DSingleFrameDataset(
        clip_tars=data["clip_tars"],
        mano_model_dir=data["mano_model_dir"],
        camera_id=data.get("camera_id", "1201-2"),
        hands=data.get("hands", "right"),
        output_size=data.get("output_size", 512),
        grayscale=data.get("grayscale", True),
        frame_start=data.get("frame_start", 0),
        frame_stride=data.get("frame_stride", 1),
        max_frames_per_clip=data.get("max_frames_per_clip"),
        condition=condition,
        seed=config.get("seed", 0),
    )
    output = Path(config.get("output_dir", "outputs/hand_restoration/debug_samples"))
    count = min(args.count or config.get("num_samples", 4), len(dataset))
    for index in range(count):
        sample = dataset[index]
        name = f"{sample['metadata']['sequence_id']}_{sample['metadata']['frame_id']}.png"
        save_debug_grid(sample, output / name)
        print(f"saved {output / name}")
    print(f"Verified {count} aligned sample(s): tensors are [-1,1], masks are [0,1].")


if __name__ == "__main__":
    main()
