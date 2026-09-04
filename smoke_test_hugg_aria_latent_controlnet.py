#!/usr/bin/env python3
"""Run one real HUGG sample through latent-mask ControlNet forward/backward."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hand_restoration.config import load_json_config
from hand_restoration.diffusion import ControlNetHandRestorer, DiffusionConfig
from train_hand_restorer import make_hugg_aria_dataset


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = Path(__file__).resolve().parent
    config = load_json_config(args.config.resolve())
    manifest = root / config["data"]["train_manifest"]
    dataset = make_hugg_aria_dataset(config, manifest)
    sample = dataset[args.sample_index]
    device = torch.device(args.device)
    restorer = ControlNetHandRestorer(
        DiffusionConfig(**config["model"]), device=device
    )
    for module in (
        restorer.vae,
        restorer.text_encoder,
        restorer.unet,
        restorer.controlnet,
    ):
        module.to(device)

    target = sample["target_rgb"].unsqueeze(0)
    condition = sample["condition_rgb"].unsqueeze(0)
    hand_mask = sample["condition_mask"].unsqueeze(0)
    loss_mask = sample["loss_mask"].unsqueeze(0)
    control = restorer._control_condition(
        condition.to(device), hand_mask.to(device)
    )
    loss = restorer.training_loss(
        target,
        condition,
        condition_mask=hand_mask,
        edit_mask=loss_mask,
        hand_loss_weight=float(config["training"]["hand_loss_weight"]),
    )
    loss.backward()
    final_projection = (
        restorer.controlnet.controlnet_cond_embedding.projection[-1]
    )
    gradient = final_projection.weight.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("latent-mask condition projection has no finite gradient")
    generated_size = None
    if args.generation_steps > 0:
        generated = restorer.generate(
            condition,
            condition_mask=hand_mask,
            steps=args.generation_steps,
            guidance_scale=1.0,
            controlnet_scale=1.0,
            seed=7,
        )
        generated_size = list(generated.size)
    print(
        json.dumps(
            {
                "status": "ok",
                "sample_index": args.sample_index,
                "target_shape": list(target.shape),
                "control_condition_shape": list(control.shape),
                "loss": float(loss.detach().cpu()),
                "condition_projection_grad_norm": float(
                    gradient.float().norm().detach().cpu()
                ),
                "generated_size": generated_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
