"""Train ControlNet with clip-disjoint, deterministic HOT3D validation."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import random
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from hand_restoration.conditions import ConditionConfig
from hand_restoration.config import load_json_config
from hand_restoration.data_config import resolve_clip_splits
from hand_restoration.diffusion import ControlNetHandRestorer, DiffusionConfig
from hand_restoration.hot3d_dataset import Hot3DSingleFrameDataset
from hand_restoration.derived_dataset import DerivedHandRestorationDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_batch_metadata(batch: dict, key: str) -> str:
    value = batch.get("metadata", {}).get(key, "")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    return str(value)


def make_dataset(config: dict, clip_tars: list[str], include_numpy: bool = False) -> Hot3DSingleFrameDataset:
    data = config["data"]
    return Hot3DSingleFrameDataset(
        clip_tars=clip_tars,
        mano_model_dir=data["mano_model_dir"],
        camera_id=data.get("camera_id", "1201-2"),
        hands=data.get("hands", "right"),
        output_size=data.get("output_size", 512),
        grayscale=data.get("grayscale", True),
        frame_start=data.get("frame_start", 0),
        frame_stride=data.get("frame_stride", 1),
        max_frames_per_clip=data.get("max_frames_per_clip"),
        condition=ConditionConfig(**config.get("condition", {})),
        seed=config.get("seed", 0),
        include_numpy=include_numpy,
        require_mano_in_frame=data.get("require_mano_in_frame", False),
        min_visible_mano_vertices=data.get("min_visible_mano_vertices", 1),
    )


def make_derived_dataset(config: dict, manifest: Path, include_numpy: bool = False) -> DerivedHandRestorationDataset:
    data = config["data"]
    return DerivedHandRestorationDataset(
        manifest=manifest,
        output_size=data.get("output_size", 512),
        grayscale=data.get("grayscale", True),
        condition=ConditionConfig(**config.get("condition", {})),
        seed=config.get("seed", 0),
        include_numpy=include_numpy,
        mask_member=data.get("mask_member", "mask.png"),
    )


def fixed_subset(dataset, limit: int | None, seed: int):
    if not limit or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].sort().values.tolist()
    return Subset(dataset, indices)


def loader_options(train_cfg: dict, workers: int) -> dict:
    options = {"num_workers": workers, "pin_memory": bool(train_cfg.get("pin_memory", False))}
    if workers > 0:
        options["persistent_workers"] = bool(train_cfg.get("persistent_workers", False))
        options["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    return options


def resolve_period(value, steps_per_epoch: int) -> int:
    if value == "epoch":
        return steps_per_epoch
    match = re.fullmatch(r"(\d+)epochs?", str(value))
    if match:
        return int(match.group(1)) * steps_per_epoch
    return max(1, int(value))


def save_run_metadata(output: Path, config_path: Path, split_path: Path | None, config: dict, train_size: int, val_size: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.json")
    if split_path is not None:
        shutil.copy2(split_path, output / "split.json")
    packages = {}
    for name in ("torch", "torchvision", "accelerate", "diffusers", "transformers", "huggingface-hub", "numpy", "opencv-python", "smplx", "trimesh"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "packages": packages,
        "train_samples": train_size,
        "validation_samples": val_size,
        "seed": config.get("seed", 0),
    }
    (output / "run_metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def validation_loss(restorer, loader, accelerator, seed: int, max_batches: int | None) -> float:
    restorer.controlnet.eval()
    total = torch.zeros(2, device=accelerator.device, dtype=torch.float64)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        generator = torch.Generator(device=accelerator.device).manual_seed(seed + batch_index)
        loss = restorer.training_loss(batch["target_rgb"], batch["condition_rgb"], generator=generator)
        batch_size = batch["target_rgb"].shape[0]
        total += torch.tensor([float(loss) * batch_size, batch_size], device=accelerator.device, dtype=torch.float64)
    total = accelerator.reduce(total, reduction="sum")
    restorer.controlnet.train()
    if total[1].item() == 0:
        raise RuntimeError("Validation loader produced no samples.")
    return float((total[0] / total[1]).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None, help="ControlNet .pt checkpoint from this script.")
    args = parser.parse_args()
    from accelerate import Accelerator
    from diffusers.optimization import get_scheduler

    root = Path(__file__).resolve().parent
    config_path = args.config.resolve()
    config = load_json_config(config_path)
    seed = int(config.get("seed", 0))
    set_seed(seed)
    train_cfg = config["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        mixed_precision=train_cfg.get("mixed_precision", "no"),
        log_with="wandb" if train_cfg.get("use_wandb", False) else None,
    )
    data_format = config["data"].get("format", "raw_hot3d")
    if data_format == "derived_webdataset":
        data = config["data"]
        train_manifest = root / data["train_manifest"]
        val_manifest = root / data["val_manifest"] if data.get("val_manifest") else None
        train_dataset = make_derived_dataset(config, train_manifest)
        val_dataset = make_derived_dataset(config, val_manifest) if val_manifest else None
        split_path = root / data["source_split_json"] if data.get("source_split_json") else None
    elif data_format == "raw_hot3d":
        train_clips, val_clips, split_path = resolve_clip_splits(config, root)
        train_dataset = make_dataset(config, train_clips)
        val_dataset = make_dataset(config, val_clips) if val_clips else None
    else:
        raise ValueError(f"Unsupported data.format: {data_format}")
    train_dataset = fixed_subset(train_dataset, train_cfg.get("max_train_samples"), seed)
    if val_dataset is not None:
        val_dataset = fixed_subset(val_dataset, train_cfg.get("max_validation_samples"), int(train_cfg.get("validation_seed", seed)))
    batch_size = int(train_cfg.get("batch_size", 1))
    workers = int(train_cfg.get("num_workers", 0))
    train_generator = torch.Generator().manual_seed(seed)
    common_loader_options = loader_options(train_cfg, workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=train_generator, **common_loader_options)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=int(train_cfg.get("validation_batch_size", batch_size)), shuffle=False, **common_loader_options)

    micro_batches = math.ceil(len(train_dataset) / (batch_size * accelerator.num_processes))
    optimizer_steps_per_epoch = math.ceil(micro_batches / accelerator.gradient_accumulation_steps)
    configured_steps = train_cfg.get("max_train_steps")
    if configured_steps is None:
        steps = optimizer_steps_per_epoch * int(train_cfg.get("num_train_epochs", 1))
    else:
        steps = int(configured_steps)
    output = root / train_cfg.get("output_dir", "outputs/hand_restoration/train")
    if accelerator.is_main_process:
        save_run_metadata(output, config_path, split_path, config, len(train_dataset), len(val_dataset) if val_dataset else 0)
        print(f"train_samples={len(train_dataset)} validation_samples={len(val_dataset) if val_dataset else 0} optimizer_steps_per_epoch={optimizer_steps_per_epoch} max_train_steps={steps}")

    device = accelerator.device
    restorer = ControlNetHandRestorer(DiffusionConfig(**config.get("model", {})), device=device)
    restorer.vae.to(device)
    restorer.text_encoder.to(device)
    restorer.unet.to(device)
    if args.resume:
        restorer.load_controlnet(args.resume)
    resume_step = 0
    if args.resume:
        match = re.search(r"(?:controlnet_step|global_step)(\d+)$", args.resume.stem)
        if match:
            resume_step = int(match.group(1))
    optimizer = torch.optim.AdamW(restorer.trainable_parameters, lr=train_cfg.get("learning_rate", 1e-5), betas=(0.9, 0.999), weight_decay=train_cfg.get("weight_decay", 1e-2))
    scheduler = get_scheduler(train_cfg.get("lr_scheduler", "constant"), optimizer=optimizer, num_training_steps=steps * accelerator.num_processes, num_warmup_steps=train_cfg.get("warmup_steps", 0))
    if val_loader is None:
        restorer.controlnet, optimizer, train_loader, scheduler = accelerator.prepare(restorer.controlnet, optimizer, train_loader, scheduler)
    else:
        restorer.controlnet, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(restorer.controlnet, optimizer, train_loader, val_loader, scheduler)
    restorer.controlnet.train()

    log_file = None
    log_writer = None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fields = ["run_id", "resume_from", "run_step", "total_step", "epoch", "batch_in_epoch", "run_samples_seen", "sequence_id", "frame_id", "loss", "loss_ema", "validation_loss", "learning_rate", "grad_norm", "step_seconds", "elapsed_seconds", "cuda_allocated_mb", "cuda_reserved_mb"]
    if accelerator.is_main_process:
        log_path = output / "training_log.csv"
        write_header = not log_path.exists() or log_path.stat().st_size == 0
        log_file = log_path.open("a", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(log_file, fieldnames=fields)
        if write_header:
            log_writer.writeheader()

    global_step = 0
    samples_seen = 0
    epoch = 0
    loss_ema = None
    ema_decay = float(train_cfg.get("loss_ema_decay", 0.98))
    csv_every = max(1, int(train_cfg.get("csv_log_every", 1)))
    checkpoint_setting = train_cfg.get("checkpoint_every", 250)
    checkpoint_every = resolve_period(checkpoint_setting, optimizer_steps_per_epoch)
    validation_setting = train_cfg.get("validation_every", checkpoint_setting)
    validation_every = resolve_period(validation_setting, optimizer_steps_per_epoch)
    validation_seed = int(train_cfg.get("validation_seed", seed))
    max_validation_batches = train_cfg.get("max_validation_batches")
    run_start = previous_step_end = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    try:
        while global_step < steps:
            for batch_in_epoch, batch in enumerate(train_loader):
                samples_seen += int(batch["target_rgb"].shape[0]) * accelerator.num_processes
                grad_norm_value = float("nan")
                with accelerator.accumulate(restorer.controlnet):
                    loss = restorer.training_loss(batch["target_rgb"], batch["condition_rgb"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite diffusion loss at optimizer step {global_step}: {loss.item()}")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(restorer.controlnet.parameters(), train_cfg.get("max_grad_norm", 1.0))
                        grad_norm_value = float(grad_norm.detach().float().item())
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if not accelerator.sync_gradients:
                    continue

                global_step += 1
                total_step = resume_step + global_step
                loss_value = float(loss.detach().float().item())
                loss_ema = loss_value if loss_ema is None else ema_decay * loss_ema + (1.0 - ema_decay) * loss_value
                validation_value = None
                validation_due = val_loader is not None and (total_step % validation_every == 0 or global_step == steps)
                if validation_due:
                    validation_value = validation_loss(restorer, val_loader, accelerator, validation_seed, max_validation_batches)
                now = time.perf_counter()
                step_seconds = now - previous_step_end
                previous_step_end = now
                if accelerator.is_main_process and global_step % csv_every == 0:
                    allocated = torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
                    reserved = torch.cuda.memory_reserved(device) / 1024**2 if torch.cuda.is_available() else 0
                    log_writer.writerow({"run_id": run_id, "resume_from": str(args.resume or ""), "run_step": global_step, "total_step": total_step, "epoch": epoch, "batch_in_epoch": batch_in_epoch, "run_samples_seen": samples_seen, "sequence_id": format_batch_metadata(batch, "sequence_id"), "frame_id": format_batch_metadata(batch, "frame_id"), "loss": f"{loss_value:.9g}", "loss_ema": f"{loss_ema:.9g}", "validation_loss": "" if validation_value is None else f"{validation_value:.9g}", "learning_rate": f"{optimizer.param_groups[0]['lr']:.9g}", "grad_norm": f"{grad_norm_value:.9g}", "step_seconds": f"{step_seconds:.6f}", "elapsed_seconds": f"{now-run_start:.6f}", "cuda_allocated_mb": f"{allocated:.3f}", "cuda_reserved_mb": f"{reserved:.3f}"})
                    log_file.flush()
                if accelerator.is_main_process and global_step % train_cfg.get("log_every", 10) == 0:
                    suffix = "" if validation_value is None else f" validation_loss={validation_value:.6f}"
                    print(f"step={total_step:05d} epoch={epoch} samples={samples_seen} loss={loss_value:.6f} loss_ema={loss_ema:.6f}{suffix}")
                checkpoint_due = total_step % checkpoint_every == 0
                if checkpoint_due:
                    accelerator.wait_for_everyone()
                if accelerator.is_main_process and checkpoint_due:
                    torch.save({"config": restorer.config.__dict__, "global_step": total_step, "state_dict": accelerator.unwrap_model(restorer.controlnet).state_dict()}, output / f"controlnet_step{total_step:06d}.pt")
                if global_step >= steps:
                    break
            epoch += 1
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.save({"config": restorer.config.__dict__, "global_step": resume_step + global_step, "state_dict": accelerator.unwrap_model(restorer.controlnet).state_dict()}, output / "controlnet_final.pt")
            print(f"Saved {output / 'controlnet_final.pt'}")
            print(f"Saved training log to {output / 'training_log.csv'}")
    finally:
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
