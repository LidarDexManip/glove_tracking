#!/usr/bin/env python3
"""Minimal Gradio UI for filtered in-domain HUGG Aria restoration."""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path

from hand_restoration.config import load_json_config
from hand_restoration.hugg_aria_dataset import HuggAriaOverlayDataset
from hand_restoration.inference import (
    build_restorer,
    checkpoint_label,
    discover_checkpoints,
    run_inference,
    save_rgb,
)
from hand_restoration.visualize import rgb_float_to_u8


ROOT = Path(__file__).resolve().parent
WEB_CSS = """
.hero {padding: 8px 0 4px}
.hero h1 {font-size: 2rem; margin-bottom: .2rem}
.hero p {color: #94a3b8; max-width: 900px}
"""


def absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def manifest_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record_key(record: dict) -> tuple[str, int]:
    return str(record["sequence_id"]), int(record["frame_index"])



class HuggAriaInferenceApp:
    """Serve only records frozen into the filtered seen-eval manifest."""

    def __init__(
        self,
        config_path: Path,
        checkpoint_dirs: Path | list[Path],
        output_dir: Path,
        device: str,
    ) -> None:
        self.config_path = config_path
        self.config = load_json_config(config_path)
        data = self.config["data"]
        if data.get("format") != "hugg_aria_video":
            raise ValueError("The HUGG Aria web UI requires data.format=hugg_aria_video.")

        train_manifest = absolute(data["train_manifest"])
        seen_manifest = absolute(data["val_manifest"])
        train_keys = {record_key(record) for record in manifest_records(train_manifest)}
        seen_records = manifest_records(seen_manifest)
        missing = [record_key(record) for record in seen_records if record_key(record) not in train_keys]
        if missing:
            raise RuntimeError(
                f"Refusing to start: {len(missing)} seen records are not in the filtered train manifest."
            )
        if not seen_records:
            raise RuntimeError("The filtered seen-eval manifest is empty.")

        self.dataset = HuggAriaOverlayDataset(
            manifest=seen_manifest,
            pinhole_root=absolute(data["pinhole_root"]),
            render_root=absolute(data["render_root"]),
            mask_root=absolute(data["mask_root"]),
            output_size=data.get("output_size", 512),
            condition_variant=data.get("condition_variant", "sam_mask"),
            render_opacity=data.get("render_opacity", 1.0),
            render_kind=data.get("render_kind", "gaussian"),
            render_alpha_filename=data.get("render_alpha_filename", "alpha.mkv"),
            alpha_threshold=data.get("alpha_threshold", 0.0),
            loss_mask_source=data.get("loss_mask_source", "overlay"),
            loss_mask_root=(
                absolute(data["loss_mask_root"])
                if data.get("loss_mask_root")
                else None
            ),
            loss_mask_alpha_filename=data.get(
                "loss_mask_alpha_filename", "alpha.mkv"
            ),
            include_numpy=True,
            max_open_sequences=data.get("max_open_sequences", 2),
        )
        self.sample_labels = [
            f"{index:03d} | {record['sequence_id']} | frame {int(record['frame_index'])}"
            for index, record in enumerate(seen_records)
        ]
        self.sample_indices = {label: index for index, label in enumerate(self.sample_labels)}
        if isinstance(checkpoint_dirs, Path):
            checkpoint_dirs = [checkpoint_dirs]
        self.checkpoint_dirs = tuple(path.resolve() for path in checkpoint_dirs)
        self.output_dir = output_dir
        self.device = device
        self.restorer = None
        self.loaded_checkpoint: Path | None = None
        self.dataset_lock = threading.Lock()
        self.lock = threading.Lock()

    def checkpoints(self) -> list[Path]:
        now = time.time()
        checkpoints = []
        for directory in self.checkpoint_dirs:
            try:
                discovered = discover_checkpoints(directory)
            except (FileNotFoundError, NotADirectoryError):
                continue
            checkpoints.extend(
                path for path in discovered
                if path.stat().st_size > 1_000_000_000
                and now - path.stat().st_mtime >= 5
            )
        return checkpoints

    def checkpoint_choices(self) -> list[tuple[str, str]]:
        return [
            (f"{checkpoint_label(path)} · {path.name}", str(path.resolve()))
            for path in reversed(self.checkpoints())
        ]

    def sample(self, label: str) -> dict:
        if label not in self.sample_indices:
            raise ValueError("Select a sample from the filtered seen-eval list.")
        # Dataset performs a second guard against training_eligible=false in SQLite.
        # Gradio may use a different worker thread for consecutive callbacks, while
        # the dataset intentionally caches SQLite connections and video decoders.
        with self.dataset_lock:
            return self.dataset[self.sample_indices[label]]

    def preview(self, label: str):
        sample = self.sample(label)
        metadata = sample["metadata"]
        status = (
            f"Eligible sample: {metadata['sequence_id']} "
            f"frame {metadata['frame_id']}."
        )
        return (
            rgb_float_to_u8(sample["condition_rgb_np"]),
            None,
            rgb_float_to_u8(sample["target_rgb_np"]),
            status,
        )

    def _load_checkpoint(self, checkpoint: Path) -> None:
        if self.restorer is None:
            self.restorer = build_restorer(self.config, device=self.device)
        if self.loaded_checkpoint != checkpoint:
            self.restorer.load_controlnet(checkpoint)
            self.loaded_checkpoint = checkpoint

    def infer(self, label: str, checkpoint_value: str, steps: int, seed: int):
        checkpoint = Path(checkpoint_value).resolve()
        valid = {path.resolve() for path in self.checkpoints()}
        if checkpoint not in valid or checkpoint.parent not in self.checkpoint_dirs:
            raise ValueError("Select a complete checkpoint from the configured training output.")
        sample = self.sample(label)
        with self.lock:
            self._load_checkpoint(checkpoint)
            result = run_inference(
                self.restorer,
                sample,
                self.config,
                steps=int(steps),
                seed=int(seed),
            )

        index = self.sample_indices[label]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        experiment = self.output_dir / f"sample_{index:03d}_{checkpoint.stem}_{timestamp}"
        experiment.mkdir(parents=True, exist_ok=False)
        save_rgb(experiment / "condition.png", sample["condition_rgb_np"])
        save_rgb(experiment / "raw_model_output.png", result.generated)
        save_rgb(experiment / "gt.png", sample["target_rgb_np"])
        metadata = sample["metadata"]
        status = (
            f"Done: `{checkpoint.name}` · `{metadata['sequence_id']}` frame "
            f"{metadata['frame_id']} · saved to `{experiment}`."
        )
        return rgb_float_to_u8(result.generated), status


def build_ui(app: HuggAriaInferenceApp):
    import gradio as gr

    choices = app.checkpoint_choices()
    if not choices:
        raise FileNotFoundError(
            f"No complete checkpoints found under {app.checkpoint_dirs}"
        )

    with gr.Blocks(title="HUGG Aria Hand Restoration") as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>HUGG Aria Hand Restoration</h1>
              <p>Condition input, raw diffusion output, and ground truth. Every selectable frame is from
              the frozen, training-eligible in-domain manifest.</p>
            </div>
            """
        )
        with gr.Row():
            sample = gr.Dropdown(
                choices=app.sample_labels,
                value=app.sample_labels[0],
                label=f"Eligible seen sample ({len(app.sample_labels)} total)",
                allow_custom_value=False,
            )
            random_button = gr.Button("Random eligible sample")
        with gr.Row():
            checkpoint = gr.Dropdown(
                choices=choices,
                value=choices[0][1],
                label="Checkpoint",
                allow_custom_value=False,
            )
            refresh_button = gr.Button("Refresh checkpoints")
            steps = gr.Slider(1, 60, value=20, step=1, label="Diffusion steps")
            seed = gr.Number(value=7, precision=0, label="Seed")
        run_button = gr.Button("Run inference", variant="primary", size="lg")
        status = gr.Markdown("Ready. Model loading happens on the first inference.")
        with gr.Row():
            condition = gr.Image(label="Condition input", type="numpy")
            output = gr.Image(label="Raw model output", type="numpy")
            target = gr.Image(label="GT", type="numpy")

        def random_sample():
            label = random.choice(app.sample_labels)
            condition_value, output_value, target_value, status_value = app.preview(label)
            return label, condition_value, output_value, target_value, status_value

        def refresh_checkpoints():
            refreshed = app.checkpoint_choices()
            if not refreshed:
                raise gr.Error("No complete checkpoints are available.")
            return gr.Dropdown(
                choices=refreshed,
                value=refreshed[0][1],
                label="Checkpoint",
                allow_custom_value=False,
            )

        sample.change(app.preview, [sample], [condition, output, target, status])
        random_button.click(
            random_sample, outputs=[sample, condition, output, target, status]
        )
        refresh_button.click(refresh_checkpoints, outputs=[checkpoint])
        run_button.click(
            app.infer,
            [sample, checkpoint, steps, seed],
            [output, status],
        )
        demo.load(app.preview, [sample], [condition, output, target, status])

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/hand_restoration/"
            "hugg_aria_aligned_sam_weight10_chunk5s_512_8gpu_batch256_resume10epochs.json"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hand_restoration/hugg_aria_web_experiments"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--username", default=os.environ.get("HAND_UI_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("HAND_UI_PASSWORD"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.share or args.host not in {"127.0.0.1", "localhost"}) and not (
        args.username and args.password
    ):
        raise SystemExit("Refusing remote exposure without username and password.")
    checkpoint_dirs = args.checkpoint_dir or [
        Path(
            "outputs/hand_restoration/"
            "hugg_aria_aligned_sam_weight10_chunk5s_512_8gpu_batch256_resume10epochs"
        ),
    ]
    app = HuggAriaInferenceApp(
        absolute(args.config),
        [absolute(path) for path in checkpoint_dirs],
        absolute(args.output_dir),
        args.device,
    )
    demo = build_ui(app)
    auth = (args.username, args.password) if args.username and args.password else None
    demo.queue(default_concurrency_limit=1, max_size=4).launch(
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
