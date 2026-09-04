"""Manual two-GPU exact-resume equivalence smoke test.

Run baseline, interrupted save, and resume as separate Accelerate launches, then
compare the emitted snapshots byte-for-byte at the tensor/state level.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train_hand_restorer import TrainingProgress, save_exact_training_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "save", "resume"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    accelerator = Accelerator()
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.SiLU(),
        torch.nn.Linear(8, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=4
    )
    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )
    progress = TrainingProgress(
        config_hash="distributed-smoke-config",
        dataset_hash="distributed-smoke-data",
        target_steps=4,
        optimizer_steps_per_epoch=4,
        batches_per_epoch=4,
        num_processes=accelerator.num_processes,
        total_step_offset=0,
    )
    accelerator.register_for_checkpointing(progress)
    if args.mode == "resume":
        if args.resume is None:
            raise ValueError("--resume is required in resume mode")
        accelerator.load_state(str(args.resume.resolve()))
        progress.validate(
            config_hash="distributed-smoke-config",
            dataset_hash="distributed-smoke-data",
            target_steps=4,
            optimizer_steps_per_epoch=4,
            batches_per_epoch=4,
            num_processes=accelerator.num_processes,
        )

    stop = 2 if args.mode == "save" else 4
    while progress.global_step < stop:
        optimizer.zero_grad()
        inputs = torch.randn(16, 4, device=accelerator.device)
        targets = torch.randn(16, 2, device=accelerator.device)
        loss = (model(inputs) - targets).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        progress.global_step += 1
        progress.epoch = 0
        progress.next_batch_in_epoch = progress.global_step
        progress.samples_seen += 16 * accelerator.num_processes
        progress.loss_ema = float(loss.detach().item())

    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "save":
        progress.train_generator_state = torch.Generator().manual_seed(99).get_state()
        state = save_exact_training_state(
            accelerator, args.output, progress,
            total_step=progress.global_step, keep_last=2,
        )
        if accelerator.is_main_process:
            (args.output / "state_path.json").write_text(
                json.dumps({"state": str(state.resolve())}) + "\n",
                encoding="utf-8",
            )
    else:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.save(
                {
                    "model": {
                        key: value.detach().cpu()
                        for key, value in accelerator.unwrap_model(
                            model
                        ).state_dict().items()
                    },
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "scheduler": copy.deepcopy(scheduler.state_dict()),
                    "progress": progress.state_dict(),
                },
                args.output / f"{args.mode}.pt",
            )


if __name__ == "__main__":
    main()
