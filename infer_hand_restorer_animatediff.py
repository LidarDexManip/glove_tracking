"""Compare raw ControlNet outputs with and without AnimateDiff on a full clip."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from hand_restoration.conditions import ConditionConfig
from hand_restoration.config import load_json_config
from hand_restoration.derived_dataset import DerivedHandRestorationDataset
from hand_restoration.diffusion import ControlNetHandRestorer, DiffusionConfig


def to_pil(tensor: torch.Tensor) -> Image.Image:
    image = ((tensor.detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5)
    return Image.fromarray(image.clip(0, 255).astype(np.uint8))


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = Image.fromarray(image)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, 220, 30), fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255))
    return np.asarray(panel)


def frame_to_u8(frame: Image.Image) -> np.ndarray:
    return np.asarray(frame.convert("RGB"))


def write_video(
    path: Path,
    frames: list[np.ndarray],
    fps: float,
    upscale: int = 1,
    h264_crf: int | None = None,
) -> None:
    height, width = frames[0].shape[:2]
    output_size = (width * upscale, height * upscale)
    if h264_crf is not None:
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(
            str(path),
            output_size,
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            output_params=[
                "-crf", str(h264_crf),
                "-preset", "slow",
                "-movflags", "+faststart",
            ],
        )
        writer.send(None)
        try:
            for frame in frames:
                if upscale != 1:
                    frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LANCZOS4)
                writer.send(np.ascontiguousarray(frame))
        finally:
            writer.close()
        return

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    try:
        for frame in frames:
            if upscale != 1:
                frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LANCZOS4)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def window_starts(num_frames: int, window_size: int, stride: int) -> list[int]:
    if num_frames <= window_size:
        return [0]
    starts = list(range(0, num_frames - window_size + 1, stride))
    final_start = num_frames - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def blend_weights(window_size: int) -> np.ndarray:
    # Nonzero triangular weights let overlapping windows cross-fade without
    # dropping the first/last frame of the full sequence.
    center = (window_size - 1) / 2.0
    weights = 1.0 - np.abs(np.arange(window_size) - center) / (center + 1.0)
    return weights.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-id", default="clip-000024")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=150)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--baseline-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--controlnet-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--generated-only",
        action="store_true",
        help="Only export AnimateDiff and no-AnimateDiff raw output videos.",
    )
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument(
        "--h264-crf",
        type=int,
        default=None,
        help="Encode with libx264 at this CRF; requires imageio-ffmpeg.",
    )
    parser.add_argument(
        "--motion-adapter-id",
        default="guoyww/animatediff-motion-adapter-v1-5-3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hand_restoration/animatediff_holdout"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this video inference.")

    root = Path(__file__).resolve().parent
    config = load_json_config(args.config.resolve())
    data = config["data"]
    if data.get("format") != "derived_webdataset":
        raise ValueError("This script currently requires data.format=derived_webdataset.")
    manifest = root / data["val_manifest"]
    dataset = DerivedHandRestorationDataset(
        manifest=manifest,
        output_size=data.get("output_size", 512),
        grayscale=data.get("grayscale", True),
        condition=ConditionConfig(**config.get("condition", {})),
        seed=config.get("seed", 0),
        include_numpy=True,
    )
    wanted = set(range(args.start_frame, args.start_frame + args.num_frames))
    indices = [
        i
        for i, record in enumerate(dataset.samples)
        if record["clip_id"] == args.clip_id and int(record["frame_id"]) in wanted
    ]
    indices.sort(key=lambda i: int(dataset.samples[i]["frame_id"]))
    actual = [int(dataset.samples[i]["frame_id"]) for i in indices]
    expected = list(range(args.start_frame, args.start_frame + args.num_frames))
    if actual != expected:
        raise RuntimeError(f"Requested contiguous frames {expected}, found {actual}.")
    samples = [dataset[i] for i in indices]

    from diffusers import AnimateDiffControlNetPipeline, DDIMScheduler, MotionAdapter

    device = torch.device("cuda")
    dtype = torch.bfloat16
    restorer = ControlNetHandRestorer(
        DiffusionConfig(**config.get("model", {})), device=device
    )
    restorer.load_controlnet(args.checkpoint.resolve())
    motion_adapter = MotionAdapter.from_pretrained(
        args.motion_adapter_id, torch_dtype=dtype
    )
    pipe = AnimateDiffControlNetPipeline(
        vae=restorer.vae,
        text_encoder=restorer.text_encoder,
        tokenizer=restorer.tokenizer,
        unet=restorer.unet,
        motion_adapter=motion_adapter,
        controlnet=restorer.controlnet,
        scheduler=restorer.noise_scheduler,
        feature_extractor=None,
        image_encoder=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        clip_sample=False,
        timestep_spacing="linspace",
    )
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(desc="AnimateDiff")

    condition_frames = [to_pil(sample["condition_rgb"]) for sample in samples]
    size = int(data.get("output_size", 512))
    starts = window_starts(args.num_frames, args.window_size, args.window_stride)
    weights = blend_weights(args.window_size)
    animated_sum = np.zeros((args.num_frames, size, size, 3), dtype=np.float32)
    animated_weight = np.zeros(args.num_frames, dtype=np.float32)
    global_generator = torch.Generator(device=device).manual_seed(args.seed)
    global_latents = torch.randn(
        (1, 4, args.num_frames, size // pipe.vae_scale_factor, size // pipe.vae_scale_factor),
        generator=global_generator,
        device=device,
        dtype=dtype,
    )
    with torch.inference_mode():
        for window_index, start in enumerate(starts):
            end = start + args.window_size
            print(f"AnimateDiff window {window_index + 1}/{len(starts)}: frames {start}-{end - 1}")
            window_frames = pipe(
                prompt=restorer.config.prompt,
                negative_prompt=restorer.config.negative_prompt,
                conditioning_frames=condition_frames[start:end],
                num_frames=args.window_size,
                height=size,
                width=size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_scale,
                latents=global_latents[:, :, start:end].clone(),
                decode_chunk_size=4,
            ).frames[0]
            for local_index, frame in enumerate(window_frames):
                frame_index = start + local_index
                animated_sum[frame_index] += frame_to_u8(frame).astype(np.float32) * weights[local_index]
                animated_weight[frame_index] += weights[local_index]
    animated_frames = [
        (animated_sum[i] / animated_weight[i]).clip(0, 255).astype(np.uint8)
        for i in range(args.num_frames)
    ]

    # Build the image ControlNet baseline once and process frames in batches.
    # Every frame gets identical seeded initial noise, which avoids giving the
    # baseline an unnecessary disadvantage from unrelated random seeds.
    from diffusers import StableDiffusionControlNetPipeline

    baseline_pipe = StableDiffusionControlNetPipeline(
        vae=restorer.vae,
        text_encoder=restorer.text_encoder,
        tokenizer=restorer.tokenizer,
        unet=restorer.unet,
        controlnet=restorer.controlnet,
        scheduler=restorer.noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker=False,
    )
    baseline_pipe.scheduler = DDIMScheduler.from_config(
        baseline_pipe.scheduler.config,
        beta_schedule="linear",
        clip_sample=False,
        timestep_spacing="linspace",
    )
    baseline_pipe.to(device=device, dtype=dtype)
    baseline_frames: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, args.num_frames, args.baseline_batch_size):
            batch = condition_frames[start : start + args.baseline_batch_size]
            print(f"No-AnimateDiff batch: frames {start}-{start + len(batch) - 1}")
            generators = [
                torch.Generator(device=device).manual_seed(args.seed)
                for _ in batch
            ]
            output = baseline_pipe(
                prompt=[restorer.config.prompt] * len(batch),
                negative_prompt=[restorer.config.negative_prompt] * len(batch),
                image=batch,
                height=size,
                width=size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_scale,
                generator=generators,
            ).images
            baseline_frames.extend(frame_to_u8(frame) for frame in output)

    output = args.output_dir / (
        f"{args.clip_id}_frames{args.start_frame:06d}-"
        f"{args.start_frame + args.num_frames - 1:06d}_seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_video(
        output / "animatediff_raw.mp4", animated_frames, args.fps,
        upscale=args.upscale, h264_crf=args.h264_crf,
    )
    write_video(
        output / "no_animatediff_raw.mp4", baseline_frames, args.fps,
        upscale=args.upscale, h264_crf=args.h264_crf,
    )
    if not args.generated_only:
        ground_truth_frames: list[np.ndarray] = []
        comparisons: list[np.ndarray] = []
        for sample, animated_np, baseline_np in zip(samples, animated_frames, baseline_frames):
            target_np = (sample["target_rgb_np"] * 255).clip(0, 255).astype(np.uint8)
            ground_truth_frames.append(target_np)
            comparisons.append(
                np.concatenate(
                    [
                        label_panel(target_np, "Ground truth"),
                        label_panel(animated_np, "AnimateDiff"),
                        label_panel(baseline_np, "No AnimateDiff"),
                    ],
                    axis=1,
                )
            )
        write_video(
            output / "ground_truth.mp4", ground_truth_frames, args.fps,
            upscale=args.upscale, h264_crf=args.h264_crf,
        )
        write_video(
            output / "comparison.mp4", comparisons, args.fps,
            upscale=args.upscale, h264_crf=args.h264_crf,
        )
    metadata = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "clip_id": args.clip_id,
        "frames": actual,
        "motion_adapter_id": args.motion_adapter_id,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "window_starts": starts,
        "raw_model_output": True,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_scale": args.controlnet_scale,
        "seed": args.seed,
        "fps": args.fps,
        "generated_only": args.generated_only,
        "upscale": args.upscale,
        "h264_crf": args.h264_crf,
        "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(0),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
