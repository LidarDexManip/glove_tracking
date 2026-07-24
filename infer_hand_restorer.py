import argparse
from pathlib import Path

from hand_restoration.config import load_json_config
from hand_restoration.inference import (
    build_restorer,
    load_sample,
    run_inference,
    save_experiment_inputs,
    save_rgb,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained ControlNet restorer on one HOT3D sample.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Trained ControlNet checkpoint. Omit only with --pretrained.")
    parser.add_argument("--pretrained", action="store_true", help="Preview the untrained ControlNet initialized from public SD 1.5 weights.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split", choices=("train", "validation"), default="train", help="Select the clip-disjoint dataset side when using a split manifest.")
    parser.add_argument("--clip", default=None, help="Select one clip, e.g. clip-000017 or 000017; it must belong to --split.")
    parser.add_argument("--frame", type=int, default=None, help="Load this exact frame ID instead of using --sample-index.")
    parser.add_argument("--steps", type=int, default=None, help="Override configured diffusion steps; useful for fast baseline previews.")
    args = parser.parse_args()
    if (args.checkpoint is None) == (not args.pretrained):
        parser.error("Provide --checkpoint for a trained model, or use --pretrained (but not both).")
    config = load_json_config(args.config)
    sample = load_sample(config, frame_id=args.frame, sample_index=args.sample_index, split=args.split, clip_name=args.clip)
    restorer = build_restorer(config)
    if args.checkpoint is not None:
        restorer.load_controlnet(args.checkpoint)
    steps = args.steps or config.get("inference", {}).get("steps", 30)
    print(f"Generating on {restorer.device_name} with {steps} diffusion steps.")
    result = run_inference(restorer, sample, config, steps=steps)
    output = Path(config.get("output_dir", "outputs/hand_restoration/inference"))
    output.mkdir(parents=True, exist_ok=True)
    save_experiment_inputs(sample, output)
    save_rgb(output / "generated.png", result.generated)
    save_rgb(output / "restored.png", result.restored)
    model_label = "public SD 1.5 / untrained ControlNet" if args.pretrained else str(args.checkpoint)
    print(f"Saved restoration to {output / 'restored.png'} using {model_label}")
    print(f"PSNR full={result.psnr_full:.3f} dB, masked={result.psnr_masked:.3f} dB")


if __name__ == "__main__":
    main()
