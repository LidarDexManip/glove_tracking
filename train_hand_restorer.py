"""Train ControlNet with clip-disjoint, deterministic HOT3D validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from hand_restoration.diffusion import (
    LOSS_STAT_NAMES,
    ControlNetHandRestorer,
    DiffusionConfig,
)
from hand_restoration.hot3d_dataset import Hot3DSingleFrameDataset
from hand_restoration.derived_dataset import DerivedHandRestorationDataset
from hand_restoration.hugg_aria_dataset import HuggAriaGaussianDataset
from hand_restoration.samplers import TemporalChunkShuffleSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_fingerprint(dataset) -> str:
    digest = hashlib.sha256(type(dataset).__name__.encode("utf-8"))
    if isinstance(dataset, Subset):
        digest.update(dataset_fingerprint(dataset.dataset).encode("ascii"))
        digest.update(stable_hash(list(dataset.indices)).encode("ascii"))
    elif hasattr(dataset, "samples"):
        for record in dataset.samples:
            digest.update(stable_hash(record).encode("ascii"))
    else:
        digest.update(str(len(dataset)).encode("ascii"))
    return digest.hexdigest()


def next_training_position(
    epoch: int, batch_in_epoch: int, batches_per_epoch: int
) -> tuple[int, int]:
    next_batch = int(batch_in_epoch) + 1
    if next_batch >= int(batches_per_epoch):
        return int(epoch) + 1, 0
    return int(epoch), next_batch


class TrainingProgress:
    """Loop state stored alongside Accelerate model/optimizer/RNG state."""

    format_version = 1

    def __init__(
        self,
        *,
        config_hash: str,
        dataset_hash: str,
        target_steps: int,
        optimizer_steps_per_epoch: int,
        batches_per_epoch: int,
        num_processes: int,
        total_step_offset: int,
    ) -> None:
        self.config_hash = config_hash
        self.dataset_hash = dataset_hash
        self.target_steps = int(target_steps)
        self.optimizer_steps_per_epoch = int(optimizer_steps_per_epoch)
        self.batches_per_epoch = int(batches_per_epoch)
        self.num_processes = int(num_processes)
        self.total_step_offset = int(total_step_offset)
        self.global_step = 0
        self.epoch = 0
        self.next_batch_in_epoch = 0
        self.samples_seen = 0
        self.loss_ema: float | None = None
        self.train_generator_state = torch.Generator().get_state()

    def state_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "target_steps": self.target_steps,
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "batches_per_epoch": self.batches_per_epoch,
            "num_processes": self.num_processes,
            "total_step_offset": self.total_step_offset,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "next_batch_in_epoch": self.next_batch_in_epoch,
            "samples_seen": self.samples_seen,
            "loss_ema": self.loss_ema,
            "train_generator_state": self.train_generator_state,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("format_version", 0)) != self.format_version:
            raise RuntimeError(
                f"Unsupported training progress format: "
                f"{state.get('format_version')}"
            )
        for key in (
            "config_hash", "dataset_hash", "target_steps",
            "optimizer_steps_per_epoch", "batches_per_epoch", "num_processes",
            "total_step_offset", "global_step", "epoch",
            "next_batch_in_epoch", "samples_seen", "loss_ema",
            "train_generator_state",
        ):
            setattr(self, key, state[key])

    def validate(
        self,
        *,
        config_hash: str,
        dataset_hash: str,
        target_steps: int,
        optimizer_steps_per_epoch: int,
        batches_per_epoch: int,
        num_processes: int,
    ) -> None:
        expected = {
            "config_hash": config_hash,
            "dataset_hash": dataset_hash,
            "target_steps": int(target_steps),
            "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
            "batches_per_epoch": int(batches_per_epoch),
            "num_processes": int(num_processes),
        }
        mismatches = {
            key: (getattr(self, key), value)
            for key, value in expected.items()
            if getattr(self, key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Exact resume requires the original config, dataset, batch "
                f"geometry, and world size; mismatches={mismatches}"
            )
        if not 0 <= int(self.global_step) <= int(self.target_steps):
            raise RuntimeError(
                f"Invalid saved global_step={self.global_step} for "
                f"target_steps={self.target_steps}"
            )
        if not 0 <= int(self.next_batch_in_epoch) < int(self.batches_per_epoch):
            raise RuntimeError(
                f"Invalid next_batch_in_epoch={self.next_batch_in_epoch}"
            )


_TRAINER_STATE_RE = re.compile(r"^trainer_state_step(\d+)$")


def save_exact_training_state(
    accelerator,
    output: Path,
    progress: TrainingProgress,
    total_step: int,
    keep_last: int,
) -> Path:
    """Atomically save Accelerate state and retain the newest checkpoints."""

    if keep_last <= 0:
        raise ValueError("full_state_keep_last must be positive")
    name = f"trainer_state_step{int(total_step):06d}"
    final = output / name
    partial = output / f".{name}.incomplete"
    if accelerator.is_main_process:
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(partial))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shutil.rmtree(final, ignore_errors=True)
        partial.replace(final)
        states = []
        for path in output.iterdir():
            match = _TRAINER_STATE_RE.fullmatch(path.name)
            if match and path.is_dir():
                states.append((int(match.group(1)), path))
        for _, stale in sorted(states)[:-keep_last]:
            shutil.rmtree(stale)
        print(f"Saved exact trainer state to {final}")
    accelerator.wait_for_everyone()
    return final


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
    )


def make_hugg_aria_dataset(config: dict, manifest: Path, include_numpy: bool = False) -> HuggAriaGaussianDataset:
    data = config["data"]
    root = Path(__file__).resolve().parent
    return HuggAriaGaussianDataset(
        manifest=manifest,
        pinhole_root=root / data["pinhole_root"],
        gaussian_root=root / data["gaussian_root"],
        mask_root=root / data["mask_root"],
        output_size=data.get("output_size", 512),
        condition_variant=data.get("condition_variant", "sam_mask"),
        gaussian_opacity=data.get("gaussian_opacity", 1.0),
        gaussian_threshold=data.get("gaussian_threshold", 16),
        render_kind=data.get("render_kind", "gaussian"),
        loss_mask_source=data.get("loss_mask_source", "overlay"),
        loss_mask_root=(
            root / data["loss_mask_root"]
            if data.get("loss_mask_root")
            else None
        ),
        loss_mask_threshold=data.get("loss_mask_threshold"),
        include_numpy=include_numpy,
        max_open_sequences=data.get("max_open_sequences", 2),
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


def summarize_loss_statistics(statistics: torch.Tensor) -> dict[str, float]:
    values = dict(zip(LOSS_STAT_NAMES, statistics.detach().double().cpu().tolist()))

    def mean(sum_key: str, count_key: str) -> float:
        count = values[count_key]
        return float("nan") if count <= 0 else values[sum_key] / count

    total_count = values["unweighted_count"]
    return {
        "weighted_loss": mean("weighted_error_sum", "weighted_count"),
        "unweighted_loss": mean("unweighted_error_sum", "unweighted_count"),
        "hand_region_loss": mean("hand_error_sum", "hand_count"),
        "background_region_loss": mean(
            "background_error_sum", "background_count"
        ),
        "hand_latent_fraction": (
            float("nan") if total_count <= 0 else values["hand_count"] / total_count
        ),
    }


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
def validation_loss(
    restorer, loader, accelerator, seed: int, max_batches: int | None,
    hand_loss_weight: float,
) -> dict[str, float]:
    restorer.controlnet.eval()
    total = torch.zeros(
        len(LOSS_STAT_NAMES), device=accelerator.device, dtype=torch.float64
    )
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        generator = torch.Generator(device=accelerator.device).manual_seed(seed + batch_index)
        _, statistics = restorer.training_loss(
            batch["target_rgb"], batch["condition_rgb"],
            edit_mask=batch.get("edit_mask"),
            hand_loss_weight=hand_loss_weight,
            generator=generator,
            return_statistics=True,
        )
        total += statistics.double()
    total = accelerator.reduce(total, reduction="sum")
    restorer.controlnet.train()
    if total[3].item() == 0:
        raise RuntimeError("Validation loader produced no samples.")
    return summarize_loss_statistics(total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume", type=Path, default=None,
        help=(
            "ControlNet .pt for a weights-only warm start, or a "
            "trainer_state_stepNNNNNN directory for exact continuation."
        ),
    )
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
    elif data_format == "hugg_aria_video":
        data = config["data"]
        train_manifest = root / data["train_manifest"]
        val_manifest = root / data["val_manifest"] if data.get("val_manifest") else None
        train_dataset = make_hugg_aria_dataset(config, train_manifest)
        val_dataset = make_hugg_aria_dataset(config, val_manifest) if val_manifest else None
        split_path = root / data["manifest_summary"] if data.get("manifest_summary") else None
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
    chunk_frames = train_cfg.get("chunk_shuffle_frames")
    train_sampler = None
    if chunk_frames is not None:
        if isinstance(train_dataset, Subset):
            raise ValueError("chunk_shuffle_frames cannot be combined with max_train_samples")
        if not hasattr(train_dataset, "samples"):
            raise ValueError("chunk_shuffle_frames requires a manifest-backed dataset")
        train_sampler = TemporalChunkShuffleSampler(
            train_dataset.samples, int(chunk_frames), seed=seed
        )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=bool(train_cfg.get("shuffle", True)) if train_sampler is None else False,
        sampler=train_sampler, generator=train_generator,
        **common_loader_options,
    )
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
    exact_resume = bool(args.resume and args.resume.is_dir())
    if args.resume and not args.resume.exists():
        raise FileNotFoundError(args.resume)
    if args.resume and not exact_resume:
        restorer.load_controlnet(args.resume)
    resume_step = 0
    if args.resume and not exact_resume:
        match = re.search(
            r"(?:controlnet_step|global_step)(\d+)$", args.resume.stem
        )
        if match:
            resume_step = int(match.group(1))
        else:
            checkpoint_metadata = torch.load(
                args.resume, map_location="cpu", weights_only=True
            )
            resume_step = int(checkpoint_metadata.get("global_step", 0))
    optimizer = torch.optim.AdamW(restorer.trainable_parameters, lr=train_cfg.get("learning_rate", 1e-5), betas=(0.9, 0.999), weight_decay=train_cfg.get("weight_decay", 1e-2))
    # AcceleratedScheduler advances once per process at every optimizer step;
    # scale both scheduler lengths so config warmup_steps remains expressed in
    # optimizer steps and does not silently shrink when world size increases.
    scheduler = get_scheduler(
        train_cfg.get("lr_scheduler", "constant"), optimizer=optimizer,
        num_training_steps=steps * accelerator.num_processes,
        num_warmup_steps=int(train_cfg.get("warmup_steps", 0))
        * accelerator.num_processes,
    )
    if val_loader is None:
        restorer.controlnet, optimizer, train_loader, scheduler = accelerator.prepare(restorer.controlnet, optimizer, train_loader, scheduler)
    else:
        restorer.controlnet, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(restorer.controlnet, optimizer, train_loader, val_loader, scheduler)

    expected_config_hash = stable_hash(config)
    expected_dataset_hash = dataset_fingerprint(train_dataset)
    batches_per_epoch = len(train_loader)
    progress = TrainingProgress(
        config_hash=expected_config_hash,
        dataset_hash=expected_dataset_hash,
        target_steps=steps,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        batches_per_epoch=batches_per_epoch,
        num_processes=accelerator.num_processes,
        total_step_offset=resume_step,
    )
    accelerator.register_for_checkpointing(progress)
    if exact_resume:
        accelerator.load_state(str(args.resume.resolve()))
        progress.validate(
            config_hash=expected_config_hash,
            dataset_hash=expected_dataset_hash,
            target_steps=steps,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            batches_per_epoch=batches_per_epoch,
            num_processes=accelerator.num_processes,
        )
        train_generator.set_state(progress.train_generator_state)
        resume_step = int(progress.total_step_offset)
        if accelerator.is_main_process:
            print(
                f"Exact resume from {args.resume}: "
                f"run_step={progress.global_step}/{steps} "
                f"total_step={resume_step + progress.global_step} "
                f"epoch={progress.epoch} "
                f"next_batch={progress.next_batch_in_epoch}"
            )
    restorer.controlnet.train()

    log_file = None
    log_writer = None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fields = [
        "run_id", "resume_from", "run_step", "total_step", "epoch",
        "batch_in_epoch", "run_samples_seen", "sequence_id", "frame_id",
        "loss", "loss_ema", "unweighted_loss", "hand_region_loss",
        "background_region_loss", "hand_latent_fraction", "validation_loss",
        "validation_unweighted_loss", "validation_hand_region_loss",
        "validation_background_region_loss", "validation_hand_latent_fraction",
        "learning_rate", "grad_norm", "step_seconds", "elapsed_seconds",
        "cuda_allocated_mb", "cuda_reserved_mb",
    ]
    if accelerator.is_main_process:
        log_path = output / "training_log.csv"
        write_header = not log_path.exists() or log_path.stat().st_size == 0
        log_file = log_path.open("a", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(log_file, fieldnames=fields)
        if write_header:
            log_writer.writeheader()
    hand_loss_weight = float(train_cfg.get("hand_loss_weight", 1.0))

    global_step = int(progress.global_step) if exact_resume else 0
    samples_seen = int(progress.samples_seen) if exact_resume else 0
    epoch = int(progress.epoch) if exact_resume else 0
    resume_batch_in_epoch = (
        int(progress.next_batch_in_epoch) if exact_resume else 0
    )
    loss_ema = progress.loss_ema if exact_resume else None
    ema_decay = float(train_cfg.get("loss_ema_decay", 0.98))
    csv_every = max(1, int(train_cfg.get("csv_log_every", 1)))
    checkpoint_setting = train_cfg.get("checkpoint_every", 250)
    checkpoint_every = resolve_period(checkpoint_setting, optimizer_steps_per_epoch)
    save_full_state = bool(train_cfg.get("save_full_state", True))
    full_state_keep_last = int(train_cfg.get("full_state_keep_last", 2))
    if full_state_keep_last <= 0:
        raise ValueError("training.full_state_keep_last must be positive")
    validation_setting = train_cfg.get("validation_every", checkpoint_setting)
    validation_every = resolve_period(validation_setting, optimizer_steps_per_epoch)
    validation_seed = int(train_cfg.get("validation_seed", seed))
    max_validation_batches = train_cfg.get("max_validation_batches")
    run_start = previous_step_end = time.perf_counter()
    pending_statistics = torch.zeros(
        len(LOSS_STAT_NAMES), device=device, dtype=torch.float64
    )
    optimizer.zero_grad(set_to_none=True)
    try:
        while global_step < steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            skipped_batches = resume_batch_in_epoch
            for batch_in_epoch, batch in enumerate(train_loader):
                if batch_in_epoch < skipped_batches:
                    continue
                samples_seen += int(batch["target_rgb"].shape[0]) * accelerator.num_processes
                grad_norm_value = float("nan")
                with accelerator.accumulate(restorer.controlnet):
                    loss, batch_statistics = restorer.training_loss(
                        batch["target_rgb"], batch["condition_rgb"],
                        edit_mask=batch.get("edit_mask"),
                        hand_loss_weight=hand_loss_weight,
                        return_statistics=True,
                    )
                    pending_statistics += batch_statistics.double()
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
                reduced_statistics = accelerator.reduce(
                    pending_statistics, reduction="sum"
                )
                loss_metrics = summarize_loss_statistics(reduced_statistics)
                pending_statistics.zero_()
                loss_value = loss_metrics["weighted_loss"]
                loss_ema = loss_value if loss_ema is None else ema_decay * loss_ema + (1.0 - ema_decay) * loss_value
                validation_metrics = None
                validation_due = val_loader is not None and (total_step % validation_every == 0 or global_step == steps)
                if validation_due:
                    validation_metrics = validation_loss(
                        restorer, val_loader, accelerator, validation_seed,
                        max_validation_batches, hand_loss_weight,
                    )
                now = time.perf_counter()
                step_seconds = now - previous_step_end
                previous_step_end = now
                if accelerator.is_main_process and global_step % csv_every == 0:
                    allocated = torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
                    reserved = torch.cuda.memory_reserved(device) / 1024**2 if torch.cuda.is_available() else 0
                    log_writer.writerow({
                        "run_id": run_id, "resume_from": str(args.resume or ""),
                        "run_step": global_step, "total_step": total_step,
                        "epoch": epoch, "batch_in_epoch": batch_in_epoch,
                        "run_samples_seen": samples_seen,
                        "sequence_id": format_batch_metadata(batch, "sequence_id"),
                        "frame_id": format_batch_metadata(batch, "frame_id"),
                        "loss": f"{loss_value:.9g}", "loss_ema": f"{loss_ema:.9g}",
                        "unweighted_loss": f"{loss_metrics['unweighted_loss']:.9g}",
                        "hand_region_loss": f"{loss_metrics['hand_region_loss']:.9g}",
                        "background_region_loss": f"{loss_metrics['background_region_loss']:.9g}",
                        "hand_latent_fraction": f"{loss_metrics['hand_latent_fraction']:.9g}",
                        "validation_loss": "" if validation_metrics is None else f"{validation_metrics['weighted_loss']:.9g}",
                        "validation_unweighted_loss": "" if validation_metrics is None else f"{validation_metrics['unweighted_loss']:.9g}",
                        "validation_hand_region_loss": "" if validation_metrics is None else f"{validation_metrics['hand_region_loss']:.9g}",
                        "validation_background_region_loss": "" if validation_metrics is None else f"{validation_metrics['background_region_loss']:.9g}",
                        "validation_hand_latent_fraction": "" if validation_metrics is None else f"{validation_metrics['hand_latent_fraction']:.9g}",
                        "learning_rate": f"{optimizer.param_groups[0]['lr']:.9g}",
                        "grad_norm": f"{grad_norm_value:.9g}",
                        "step_seconds": f"{step_seconds:.6f}",
                        "elapsed_seconds": f"{now-run_start:.6f}",
                        "cuda_allocated_mb": f"{allocated:.3f}",
                        "cuda_reserved_mb": f"{reserved:.3f}",
                    })
                    log_file.flush()
                if accelerator.is_main_process and global_step % train_cfg.get("log_every", 10) == 0:
                    suffix = "" if validation_metrics is None else f" validation_loss={validation_metrics['weighted_loss']:.6f}"
                    print(
                        f"step={total_step:05d} epoch={epoch} samples={samples_seen} "
                        f"loss={loss_value:.6f} loss_ema={loss_ema:.6f} "
                        f"hand_loss={loss_metrics['hand_region_loss']:.6f} "
                        f"background_loss={loss_metrics['background_region_loss']:.6f} "
                        f"hand_fraction={loss_metrics['hand_latent_fraction']:.4f}{suffix}"
                    )
                checkpoint_due = total_step % checkpoint_every == 0
                if checkpoint_due:
                    accelerator.wait_for_everyone()
                if accelerator.is_main_process and checkpoint_due:
                    torch.save({"config": restorer.config.__dict__, "global_step": total_step, "state_dict": accelerator.unwrap_model(restorer.controlnet).state_dict()}, output / f"controlnet_step{total_step:06d}.pt")
                if checkpoint_due and save_full_state:
                    next_epoch, next_batch = next_training_position(
                        epoch, batch_in_epoch, batches_per_epoch
                    )
                    progress.total_step_offset = resume_step
                    progress.global_step = global_step
                    progress.epoch = next_epoch
                    progress.next_batch_in_epoch = next_batch
                    progress.samples_seen = samples_seen
                    progress.loss_ema = loss_ema
                    progress.train_generator_state = train_generator.get_state()
                    save_exact_training_state(
                        accelerator, output, progress, total_step,
                        full_state_keep_last,
                    )
                if global_step >= steps:
                    break
            epoch += 1
            resume_batch_in_epoch = 0
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
