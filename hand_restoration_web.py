#!/usr/bin/env python3
"""Password-protected Gradio UI for HOT3D ControlNet inference."""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from hand_restoration.config import load_json_config
from hand_restoration.derived_dataset import DerivedHandRestorationDataset
from hand_restoration.inference import (
    build_restorer,
    checkpoint_label,
    discover_checkpoints,
    run_inference,
    save_checkpoint_result,
    save_experiment_inputs,
)
from hand_restoration.conditions import ConditionConfig
from hand_restoration.visualize import rgb_float_to_u8


ROOT = Path(__file__).resolve().parent
WEB_CSS = """
.hero {padding: 8px 0 4px}
.hero h1 {font-size: 2rem; margin-bottom: .2rem}
.hero p {color: #94a3b8; max-width: 900px}
.metric-panel {min-height: 180px}
"""


def _absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    image = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=2)


class InferenceApp:
    """Own datasets and one lazily loaded diffusion model."""

    def __init__(self, config_path: Path, checkpoint_dir: Path, output_dir: Path, device: str) -> None:
        self.config_path = config_path
        self.config = load_json_config(config_path)
        data = self.config["data"]
        if data.get("format") != "derived_webdataset":
            raise ValueError("The web UI requires data.format=derived_webdataset.")
        condition = ConditionConfig(**self.config.get("condition", {}))
        common = {
            "output_size": data.get("output_size", 512),
            "grayscale": data.get("grayscale", True),
            "condition": condition,
            "seed": self.config.get("seed", 0),
            "include_numpy": True,
        }
        self.datasets = {
            "train": DerivedHandRestorationDataset(_absolute(data["train_manifest"]), **common),
            "holdout": DerivedHandRestorationDataset(_absolute(data["val_manifest"]), **common),
        }
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.device = device
        self.restorer = None
        self.loaded_checkpoint: Path | None = None
        self.lock = threading.Lock()

    def checkpoints(self) -> list[Path]:
        return discover_checkpoints(self.checkpoint_dir)

    def checkpoint_choices(self) -> list[tuple[str, str]]:
        return [
            (f"{checkpoint_label(path)} · {path.name}", str(path))
            for path in reversed(self.checkpoints())
        ]

    def sample_count(self, split: str) -> int:
        return len(self.datasets[split])

    def sample(self, split: str, index: int) -> dict:
        dataset = self.datasets[split]
        if not 0 <= index < len(dataset):
            raise IndexError(f"Sample index {index} is outside {split} size {len(dataset)}.")
        return dataset[index]

    def preview(self, split: str, index: int):
        sample = self.sample(split, int(index))
        metadata = sample["metadata"]
        details = {
            "split": split,
            "sample_index": int(index),
            "clip_id": metadata["sequence_id"],
            "source_sequence_id": metadata.get("source_sequence_id", ""),
            "frame_id": metadata["frame_id"],
            "camera_id": metadata["camera_id"],
            "dataset_samples": len(self.datasets[split]),
        }
        return (
            rgb_float_to_u8(sample["target_rgb_np"]),
            rgb_float_to_u8(sample["condition_rgb_np"]),
            rgb_float_to_u8(sample["mano_rgb_np"]),
            _mask_rgb(sample["mano_mask_np"]),
            _mask_rgb(sample["edit_mask_np"]),
            details,
        )

    def _load_checkpoint(self, checkpoint: Path) -> None:
        if self.restorer is None:
            self.restorer = build_restorer(self.config, device=self.device)
        if self.loaded_checkpoint != checkpoint:
            self.restorer.load_controlnet(checkpoint)
            self.loaded_checkpoint = checkpoint

    def infer(
        self,
        split: str,
        index: int,
        checkpoint_value: str,
        steps: int,
        seed: int,
        guidance_scale: float,
        controlnet_scale: float,
    ):
        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_file() or checkpoint.parent.resolve() != self.checkpoint_dir.resolve():
            raise ValueError("Select a checkpoint from the configured checkpoint directory.")
        sample = self.sample(split, int(index))
        run_config = json.loads(json.dumps(self.config))
        run_config["inference"] = {
            "steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "controlnet_scale": float(controlnet_scale),
        }
        with self.lock:
            self._load_checkpoint(checkpoint)
            result = run_inference(
                self.restorer,
                sample,
                run_config,
                steps=int(steps),
                seed=int(seed),
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        experiment = self.output_dir / f"{split}_{int(index):06d}_{checkpoint.stem}_{timestamp}"
        experiment.mkdir(parents=True, exist_ok=False)
        save_experiment_inputs(sample, experiment)
        save_checkpoint_result(experiment, checkpoint, result)
        metadata = sample["metadata"]
        metrics = {
            "checkpoint": checkpoint.name,
            "split": split,
            "sample_index": int(index),
            "clip_id": metadata["sequence_id"],
            "source_sequence_id": metadata.get("source_sequence_id", ""),
            "frame_id": metadata["frame_id"],
            "steps": int(steps),
            "seed": int(seed),
            "guidance_scale": float(guidance_scale),
            "controlnet_scale": float(controlnet_scale),
            "generated_psnr_full_db": round(result.generated_psnr_full, 4),
            "generated_psnr_masked_db": round(result.generated_psnr_masked, 4),
            "restored_psnr_full_db": round(result.psnr_full, 4),
            "restored_psnr_masked_db": round(result.psnr_masked, 4),
            "saved_to": str(experiment),
        }
        return (
            rgb_float_to_u8(result.generated),
            rgb_float_to_u8(result.restored),
            metrics,
            f"Completed `{checkpoint.name}` on {split} sample {int(index)}.",
        )


def build_ui(app: InferenceApp):
    import gradio as gr

    choices = app.checkpoint_choices()
    if not choices:
        raise FileNotFoundError(f"No checkpoints found under {app.checkpoint_dir}")

    with gr.Blocks(title="HOT3D Hand Restoration") as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>HOT3D Hand Restoration Lab</h1>
              <p>Inspect derived holdout samples and run deterministic ControlNet inference
              against server-side checkpoints. The model and data never leave the GPU server.</p>
            </div>
            """
        )
        with gr.Row():
            split = gr.Radio(["holdout", "train"], value="holdout", label="Dataset split")
            sample_index = gr.Number(value=0, precision=0, minimum=0, label="Sample index")
            random_button = gr.Button("Random sample")
            preview_button = gr.Button("Preview input", variant="secondary")
        with gr.Row():
            checkpoint = gr.Dropdown(
                choices=choices,
                value=choices[0][1],
                label="Checkpoint",
                allow_custom_value=False,
            )
            refresh_button = gr.Button("Refresh checkpoints")
        with gr.Row():
            steps = gr.Slider(1, 100, value=30, step=1, label="Diffusion steps")
            seed = gr.Number(value=7, precision=0, label="Seed")
            guidance = gr.Slider(1.0, 12.0, value=5.0, step=0.1, label="Guidance scale")
            control = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="ControlNet scale")
        run_button = gr.Button("Run inference", variant="primary", size="lg")
        status = gr.Markdown("Ready. Previewing does not load the diffusion model.")

        with gr.Tabs():
            with gr.Tab("Inputs"):
                with gr.Row():
                    target = gr.Image(label="Target", type="numpy")
                    condition = gr.Image(label="Condition", type="numpy")
                    mano = gr.Image(label="Shaded MANO", type="numpy")
                with gr.Row():
                    mano_mask = gr.Image(label="MANO mask", type="numpy")
                    edit_mask = gr.Image(label="Edit mask", type="numpy")
                    sample_metadata = gr.JSON(label="Sample metadata")
            with gr.Tab("Result"):
                with gr.Row():
                    generated = gr.Image(label="Generated · raw diffusion output", type="numpy")
                    restored = gr.Image(label="Restored · hard-mask composite", type="numpy")
                metrics = gr.JSON(label="Metrics and saved experiment", elem_classes=["metric-panel"])

        def preview_callback(split_value, index_value):
            return app.preview(split_value, int(index_value))

        def random_callback(split_value):
            index = random.randrange(app.sample_count(split_value))
            return index, *app.preview(split_value, index)

        def refresh_callback():
            refreshed = app.checkpoint_choices()
            return gr.Dropdown(
                choices=refreshed,
                value=refreshed[0][1],
                label="Checkpoint",
                allow_custom_value=False,
            )

        preview_outputs = [target, condition, mano, mano_mask, edit_mask, sample_metadata]
        preview_button.click(preview_callback, [split, sample_index], preview_outputs)
        random_button.click(
            random_callback,
            [split],
            [sample_index, *preview_outputs],
        )
        refresh_button.click(refresh_callback, outputs=[checkpoint])
        run_button.click(
            app.infer,
            [split, sample_index, checkpoint, steps, seed, guidance, control],
            [generated, restored, metrics, status],
        )
        demo.load(preview_callback, [split, sample_index], preview_outputs)
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hand_restoration/train_quest3_sequence_derived_512_batch32.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/hand_restoration/train_quest3_sequence_512_batch32_epoch20"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hand_restoration/web_experiments"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true", help="Create a temporary Gradio public URL.")
    parser.add_argument("--username", default=os.environ.get("HAND_UI_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("HAND_UI_PASSWORD"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _absolute(args.config)
    checkpoint_dir = _absolute(args.checkpoint_dir)
    output_dir = _absolute(args.output_dir)
    if (args.share or args.host not in {"127.0.0.1", "localhost"}) and not (
        args.username and args.password
    ):
        raise SystemExit(
            "Refusing remote exposure without authentication. Set HAND_UI_USERNAME "
            "and HAND_UI_PASSWORD or pass --username and --password."
        )
    app = InferenceApp(config_path, checkpoint_dir, output_dir, args.device)
    demo = build_ui(app)
    auth = (args.username, args.password) if args.username and args.password else None
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        show_error=True,
        inbrowser=False,
        blocked_paths=[str(ROOT)],
        css=WEB_CSS,
    )


if __name__ == "__main__":
    main()
