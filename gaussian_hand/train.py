#!/usr/bin/env python3
"""Train a subject-bound Gaussian Hand, optionally with synchronous multi-GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist

from gaussian_hand.data.dataset import GaussianHandDataset
from gaussian_hand.losses import reconstruction_loss
from gaussian_hand.model import GaussianHand
from gaussian_hand.model.pose_refinement import PoseRefinement
from gaussian_hand.render import render_gaussian_hand

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--adaptive-every", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--allow-world-size-mismatch", action="store_true")
    return parser.parse_args()


def distributed_setup(device_arg: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=device)
        return world_size, dist.get_rank(), device
    return 1, 0, torch.device(device_arg)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sync_gradients(parameters, world_size: int):
    if world_size == 1:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def mean_metrics(parts: dict, loss: torch.Tensor, device, world_size: int):
    keys = ("loss", "l1", "dssim", "alpha", "distortion")
    values = torch.stack((loss.detach(), *(parts[k].detach() for k in keys[1:])))
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values.div_(world_size)
    return {key: float(value) for key, value in zip(keys, values)}


def save_checkpoint(path, step, model, pose_refinement, optimizer, config,
                    manifest, world_size):
    records = [json.loads(line) for line in manifest.open() if line.strip()]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    mano_path = resolve(config["mano_model"])
    payload = {
        "format_version": 2,
        "step": step,
        "global_frame_updates": step * world_size,
        "distributed_world_size": world_size,
        "model_id": config["model_id"],
        "binding": {"dataset": "HOT3D", "subject_id": "P0001",
                    "handedness": "right"},
        "source_sequences": sorted({item["sequence"] for item in records}),
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "code_commit": commit,
        "mano_model": str(mano_path),
        "mano_sha256": hashlib.sha256(mano_path.read_bytes()).hexdigest(),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model": model.state_dict(),
        "pose_refinement": pose_refinement.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, path)


@torch.no_grad()
def evaluate(model, dataset, device, resolution, loss_config, max_frames=32):
    model.eval()
    indices = np.linspace(
        0, len(dataset) - 1, min(max_frames, len(dataset))).astype(int)
    losses, mses = [], []
    for index in indices:
        sample = dataset[int(index)]
        x = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in sample.items()}
        render = render_gaussian_hand(
            model, x["mano_pca"], x["wrist_q_wxyz"], x["wrist_t_world"],
            x["world_to_camera"], x["intrinsics"], resolution, resolution)
        loss, _ = reconstruction_loss(
            render, x["image"], x["mask"], **loss_config)
        target = x["image"] * x["mask"]
        losses.append(float(loss))
        mses.append(float((render["rgb"] - target).square().mean()))
    model.train()
    mse = max(float(np.mean(mses)), 1e-12)
    return {"frames": len(indices), "reconstruction_loss": float(np.mean(losses)),
            "masked_psnr": float(-10 * np.log10(mse))}


@torch.no_grad()
def save_visual_gallery(path, model, dataset, device, resolution, count=12):
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset))).astype(int)
    samples = []
    for index in indices:
        sample = dataset[int(index)]
        x = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in sample.items()}
        render = render_gaussian_hand(
            model, x["mano_pca"], x["wrist_q_wxyz"], x["wrist_t_world"],
            x["world_to_camera"], x["intrinsics"], resolution, resolution)
        target = (x["image"].cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
        prediction = (render["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(
            np.uint8)[..., ::-1]
        alpha = (render["alpha"][..., 0].clamp(0, 1).cpu().numpy() * 255).astype(
            np.uint8)
        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        panels = []
        for label, panel in (("target", target), ("render", prediction),
                             ("alpha", alpha)):
            panel = cv2.resize(panel, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.putText(panel, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (0, 255, 0), 1, cv2.LINE_AA)
            panels.append(panel)
        row = np.concatenate(panels, axis=1)
        cv2.putText(row, f"{sample['sequence']}:{sample['frame_index']}",
                    (6, 250), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        samples.append(row)
    columns = 4
    blank = np.zeros_like(samples[0])
    while len(samples) % columns:
        samples.append(blank)
    rows = [np.concatenate(samples[i:i + columns], axis=1)
            for i in range(0, len(samples), columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.concatenate(rows, axis=0),
                       [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Failed to write {path}")
    model.train()


def main():
    args = arguments()
    world_size, rank, device = distributed_setup(args.device)
    is_main = rank == 0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = json.loads(args.config.read_text())
    expected_world_size = int(config.get("distributed_world_size", 1))
    if world_size != expected_world_size and not args.allow_world_size_mismatch:
        raise ValueError(
            f"Config requires world_size={expected_world_size}, got {world_size}")
    steps = args.steps or int(config["steps"])
    resolution = args.resolution or int(config["resolution"])
    dataset_root = resolve(config["dataset_root"])
    dataset = GaussianHandDataset(
        dataset_root, config.get("training_split", "train"), resolution)
    device_betas = torch.tensor(
        json.loads((dataset_root / "frames.jsonl").open().readline())["mano_betas"],
        dtype=torch.float32)
    model = GaussianHand(
        str(resolve(config["mano_model"])), device_betas,
        int(config["gaussians_per_face"]),
        float(config["normal_offset_max_m"])).to(device)
    pose_refinement = PoseRefinement(len(dataset)).to(device)
    resume = None
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("model_id") != config["model_id"]:
            raise ValueError("Resume checkpoint model_id mismatch")
        current_manifest_hash = hashlib.sha256(
            (dataset_root / "frames.jsonl").read_bytes()).hexdigest()
        if resume.get("manifest_sha256") != current_manifest_hash:
            raise ValueError("Resume dataset manifest/order mismatch")
        if resume["config"].get("training_split", "train") != config.get("training_split", "train"):
            raise ValueError("Resume training split mismatch")
        if int(resume.get("distributed_world_size", world_size)) != world_size:
            raise ValueError("Resume checkpoint world-size mismatch")
        model.load_resizable_state_dict(resume["model"])
        pose_refinement.load_state_dict(resume["pose_refinement"])

    lr = config["learning_rates"]
    optimizer = torch.optim.Adam([
        {"params": [model.barycentric_logits, model.log_scales, model.angles,
                    model.normal_offset_logits], "lr": lr["surface"]},
        {"params": [model.albedo_logits], "lr": lr["albedo"]},
        {"params": [model.opacity_logits], "lr": lr["opacity"]},
        {"params": [model.mano.canonical_delta], "lr": lr["canonical_mesh"]},
        {"params": model.lighting.parameters(), "lr": lr["lighting"]},
        {"params": pose_refinement.parameters(), "lr": lr["pose_delta"]},
    ], eps=1e-8)
    start_step = 0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"])

    if is_main:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "resolved_config.json").write_text(json.dumps({
            **config, "steps": steps, "resolution": resolution,
            "seed": args.seed, "actual_world_size": world_size,
        }, indent=2) + "\n")
    if world_size > 1:
        dist.barrier()

    checkpoint_every = args.checkpoint_every or int(config["checkpoint_every"])
    adaptive_every = (
        args.adaptive_every if args.adaptive_every is not None
        else int(config.get("adaptive_control", {}).get("every", 0)))
    visual_every = int(config.get("visual_every", 0))
    evaluation_every = int(config.get("evaluation_every", 0))
    gradient_scores = torch.zeros(model.gaussian_count, device=device)
    gradient_samples = 0
    started = time.perf_counter()
    measured_start = None
    measured_steps = 0

    if is_main and start_step == 0:
        save_visual_gallery(
            args.output / "visual" / "step_000000.jpg", model, dataset,
            device, resolution, int(config.get("visual_frames", 12)))
    if world_size > 1:
        dist.barrier()

    for step in range(start_step + 1, steps + 1):
        sample_index = (
            args.seed + step * 1_000_003 + rank * 9_176) % len(dataset)
        sample = dataset[sample_index]
        tensors = {k: (v.to(device) if torch.is_tensor(v) else v)
                   for k, v in sample.items()}
        index = torch.tensor([tensors["dataset_index"]], device=device)
        pca, quaternion, translation = pose_refinement(
            index, tensors["mano_pca"][None], tensors["wrist_q_wxyz"][None],
            tensors["wrist_t_world"][None])
        render = render_gaussian_hand(
            model, pca[0], quaternion[0], translation[0],
            tensors["world_to_camera"], tensors["intrinsics"],
            resolution, resolution)
        loss, parts = reconstruction_loss(
            render, tensors["image"], tensors["mask"], **config["loss"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step < int(config.get("canonical_mesh_start_step", 0)):
            model.mano.canonical_delta.grad = None
        parameters = list(model.parameters()) + list(pose_refinement.parameters())
        sync_gradients(parameters, world_size)
        with torch.no_grad():
            gradient_scores += model.barycentric_logits.grad.norm(dim=-1)
            gradient_scores += .25 * model.log_scales.grad.norm(dim=-1)
            gradient_samples += 1
        torch.nn.utils.clip_grad_norm_(
            parameters, 10.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            model.log_scales.clamp_(-10.0, -3.0)
            model.barycentric_logits.clamp_(-12.0, 12.0)
            model.opacity_logits.clamp_(-12.0, 12.0)
            model.mano.canonical_delta.clamp_(-0.001, 0.001)
            pose_refinement.deltas.weight[:, :15].clamp_(-0.5, 0.5)
            pose_refinement.deltas.weight[:, 15:18].clamp_(-0.2, 0.2)
            pose_refinement.deltas.weight[:, 18:21].clamp_(-0.01, 0.01)

        if step == 6:
            torch.cuda.synchronize(device)
            measured_start = time.perf_counter()
        if step > 6:
            measured_steps += 1
        if adaptive_every and step % adaptive_every == 0:
            control = config.get("adaptive_control", {})
            density = model.adaptive_density_control(
                optimizer, gradient_scores / max(gradient_samples, 1),
                float(control.get("split_fraction", .05)),
                int(control.get("max_gaussians", 20000)),
                float(control.get("opacity_prune_threshold", .02)
                      if step >= int(control.get("prune_start_step", 10000))
                      else 0))
            changed = model.reassign_to_nearest_triangles()
            if is_main:
                print(json.dumps({
                    "step": step, "global_frame_updates": step * world_size,
                    "adaptive_density": density,
                    "adaptive_reassigned": changed}), flush=True)
            gradient_scores = torch.zeros(model.gaussian_count, device=device)
            gradient_samples = 0

        if step % args.log_every == 0 or step == start_step + 1:
            averaged = mean_metrics(parts, loss, device, world_size)
            if is_main:
                elapsed = time.perf_counter() - started
                record = {
                    "step": step,
                    "global_frame_updates": step * world_size,
                    **averaged,
                    "gaussians": model.gaussian_count,
                    "seconds_per_global_step": elapsed / (step - start_step),
                    "cuda_peak_gib_per_rank":
                        torch.cuda.max_memory_allocated(device) / 2**30,
                }
                if measured_start is not None and measured_steps:
                    torch.cuda.synchronize(device)
                    record["warm_seconds_per_global_step"] = (
                        time.perf_counter() - measured_start) / measured_steps
                print(json.dumps(record), flush=True)

        evaluate_now = evaluation_every and step % evaluation_every == 0
        visual_now = visual_every and step % visual_every == 0
        if evaluate_now or visual_now or step == steps:
            if is_main:
                if evaluate_now or step == steps:
                    metrics = evaluate(
                        model, dataset, device, resolution, config["loss"],
                        int(config.get("evaluation_frames", 32)))
                    print(json.dumps(
                        {"step": step, "training_set_evaluation": metrics}),
                        flush=True)
                if visual_now or step == steps:
                    save_visual_gallery(
                        args.output / "visual" / f"step_{step:06d}.jpg",
                        model, dataset, device, resolution,
                        int(config.get("visual_frames", 12)))
            if world_size > 1:
                dist.barrier()

        if is_main and (step % checkpoint_every == 0 or step == steps):
            save_checkpoint(
                args.output / f"step_{step:06d}.pt", step, model,
                pose_refinement, optimizer, config,
                dataset_root / "frames.jsonl", world_size)

    if world_size > 1:
        dist.barrier()
    if is_main:
        print(json.dumps({
            "status": "complete", "steps": steps,
            "global_frame_updates": steps * world_size,
            "elapsed_seconds": time.perf_counter() - started,
            "final_gaussians": model.gaussian_count}), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
