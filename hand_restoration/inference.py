from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from .conditions import ConditionConfig
from .data_config import resolve_clip_splits
from .diffusion import ControlNetHandRestorer, DiffusionConfig
from .hot3d_dataset import Hot3DSingleFrameDataset
from .visualize import rgb_float_to_u8, save_debug_grid


_STEP_CHECKPOINT = re.compile(r"^controlnet_step(\d+)\.pt$")


@dataclass(frozen=True)
class InferenceResult:
    generated: np.ndarray
    restored: np.ndarray
    generated_psnr_full: float
    generated_psnr_masked: float
    psnr_full: float
    psnr_masked: float


def psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = pred - target
    values = diff[mask > 0] if mask is not None else diff.reshape(-1)
    mse = float(np.mean(values * values))
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def load_sample(
    config: dict,
    frame_id: int | None = None,
    sample_index: int = 0,
    split: str = "train",
) -> dict:
    data = config["data"]
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    train_clips, val_clips, _ = resolve_clip_splits(config, Path(__file__).resolve().parents[1])
    clip_tars = train_clips if split == "train" else val_clips
    if not clip_tars:
        raise ValueError(f"No {split} clips configured.")
    exact_frame = frame_id is not None
    frame_start = int(frame_id) if exact_frame else data.get("frame_start", 0)
    dataset = Hot3DSingleFrameDataset(
        clip_tars=clip_tars,
        mano_model_dir=data["mano_model_dir"],
        camera_id=data.get("camera_id", "1201-2"),
        hands=data.get("hands", "right"),
        output_size=data.get("output_size", 512),
        grayscale=data.get("grayscale", True),
        frame_start=frame_start,
        frame_stride=1 if exact_frame else data.get("frame_stride", 1),
        max_frames_per_clip=1 if exact_frame else data.get("max_frames_per_clip"),
        condition=ConditionConfig(**config.get("condition", {})),
        seed=config.get("seed", 0),
        require_mano_in_frame=data.get("require_mano_in_frame", False),
        min_visible_mano_vertices=data.get("min_visible_mano_vertices", 1),
    )
    if exact_frame:
        sample_index = 0
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample-index {sample_index} is outside dataset size {len(dataset)}")
    sample = dataset[sample_index]
    if exact_frame and int(sample["metadata"]["frame_id"]) != int(frame_id):
        actual = sample["metadata"]["frame_id"]
        raise ValueError(
            f"Frame {frame_id:06d} has no requested MANO annotation; "
            f"the next available frame is {actual}."
        )
    return sample


def build_restorer(config: dict, device: str | torch.device | None = None) -> ControlNetHandRestorer:
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return ControlNetHandRestorer(DiffusionConfig(**config.get("model", {})), device=selected_device)


def run_inference(
    restorer: ControlNetHandRestorer,
    sample: dict,
    config: dict,
    *,
    steps: int | None = None,
    seed: int | None = None,
) -> InferenceResult:
    inference_config = config.get("inference", {})
    generated = np.asarray(
        restorer.generate(
            sample["condition_rgb"].unsqueeze(0),
            steps=steps or inference_config.get("steps", 30),
            guidance_scale=inference_config.get("guidance_scale", 5.0),
            controlnet_scale=inference_config.get("controlnet_scale", 1.0),
            seed=config.get("seed", 0) if seed is None else seed,
        )
    ).astype(np.float32) / 255.0
    edit_mask = (sample["edit_mask_np"] > 0)[..., None]
    restored = np.where(edit_mask, generated, sample["condition_rgb_np"])
    target = sample["target_rgb_np"]
    return InferenceResult(
        generated=generated,
        restored=restored,
        generated_psnr_full=psnr(generated, target),
        generated_psnr_masked=psnr(generated, target, sample["edit_mask_np"]),
        psnr_full=psnr(restored, target),
        psnr_masked=psnr(restored, target, sample["edit_mask_np"]),
    )


def discover_checkpoints(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Checkpoint directory does not exist: {directory}")

    step_checkpoints: list[tuple[int, Path]] = []
    final_checkpoints: list[Path] = []
    for path in directory.glob("*.pt"):
        match = _STEP_CHECKPOINT.match(path.name)
        if match:
            step_checkpoints.append((int(match.group(1)), path))
        elif path.name == "controlnet_final.pt":
            final_checkpoints.append(path)
    checkpoints = [path for _, path in sorted(step_checkpoints)] + sorted(final_checkpoints)
    if not checkpoints:
        raise FileNotFoundError(
            f"No controlnet_step*.pt or controlnet_final.pt checkpoints found in {directory}"
        )
    return checkpoints


def create_experiment_directory(root: str | Path, frame_id: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(root) / f"frame_{frame_id:06d}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), cv2.cvtColor(rgb_float_to_u8(image), cv2.COLOR_RGB2BGR))
    if not success:
        raise OSError(f"Failed to save image: {path}")


def save_experiment_inputs(sample: dict, output: str | Path) -> None:
    output = Path(output)
    save_rgb(output / "target.png", sample["target_rgb_np"])
    save_rgb(output / "condition.png", sample["condition_rgb_np"])
    save_debug_grid(sample, output / "input_debug_grid.png")


def save_checkpoint_result(
    output: str | Path,
    checkpoint: str | Path,
    result: InferenceResult,
) -> Path:
    checkpoint = Path(checkpoint)
    result_dir = Path(output) / checkpoint.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(result_dir / "generated.png", result.generated)
    save_rgb(result_dir / "restored.png", result.restored)
    metrics = {
        "checkpoint": str(checkpoint.resolve()),
        "generated_psnr_full_db": result.generated_psnr_full,
        "generated_psnr_masked_db": result.generated_psnr_masked,
        "restored_psnr_full_db": result.psnr_full,
        "restored_psnr_masked_db": result.psnr_masked,
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return result_dir


def checkpoint_label(path: str | Path) -> str:
    path = Path(path)
    match = _STEP_CHECKPOINT.match(path.name)
    return f"step {int(match.group(1))}" if match else "final"
