from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest
import torch
from accelerate import Accelerator

from hand_restoration.samplers import TemporalChunkShuffleSampler
from train_hand_restorer import (
    TrainingProgress,
    next_training_position,
    resume_compatible_config_hash,
    save_exact_training_state,
    save_run_metadata,
)


def assert_optimizer_state_equal(left: dict, right: dict) -> None:
    assert left["param_groups"] == right["param_groups"]
    assert left["state"].keys() == right["state"].keys()
    for parameter, values in left["state"].items():
        assert values.keys() == right["state"][parameter].keys()
        for name, value in values.items():
            restored = right["state"][parameter][name]
            if torch.is_tensor(value):
                assert torch.equal(value, restored)
            else:
                assert value == restored


def test_accelerate_exact_state_round_trip(tmp_path) -> None:
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5
    )
    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )
    progress = TrainingProgress(
        config_hash="config",
        dataset_hash="dataset",
        target_steps=5,
        optimizer_steps_per_epoch=5,
        batches_per_epoch=5,
        num_processes=accelerator.num_processes,
        total_step_offset=10,
    )
    accelerator.register_for_checkpointing(progress)

    def train_step() -> None:
        optimizer.zero_grad()
        loss = model(torch.randn(4, 3, device=accelerator.device)).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

    train_step()
    progress.global_step = 1
    progress.next_batch_in_epoch = 1
    progress.samples_seen = 4
    progress.loss_ema = 0.25
    state_dir = save_exact_training_state(
        accelerator, tmp_path, progress, total_step=11, keep_last=2
    )
    model_before = {
        key: value.detach().clone()
        for key, value in accelerator.unwrap_model(model).state_dict().items()
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    expected_torch = torch.rand(5, device=accelerator.device)
    expected_python = random.random()
    expected_numpy = np.random.random()

    train_step()
    progress.global_step = 2
    torch.rand(7, device=accelerator.device)
    random.random()
    np.random.random()
    accelerator.load_state(str(state_dir))

    assert all(
        torch.equal(value, accelerator.unwrap_model(model).state_dict()[key])
        for key, value in model_before.items()
    )
    assert_optimizer_state_equal(optimizer_before, optimizer.state_dict())
    assert scheduler.state_dict() == scheduler_before
    assert torch.equal(
        expected_torch, torch.rand(5, device=accelerator.device)
    )
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert (
        progress.global_step,
        progress.next_batch_in_epoch,
        progress.samples_seen,
        progress.loss_ema,
    ) == (1, 1, 4, 0.25)


def test_progress_validation_and_next_position() -> None:
    progress = TrainingProgress(
        config_hash="config",
        dataset_hash="dataset",
        target_steps=10,
        optimizer_steps_per_epoch=5,
        batches_per_epoch=5,
        num_processes=8,
        total_step_offset=100,
    )
    progress.validate(
        config_hash="config",
        dataset_hash="dataset",
        target_steps=10,
        optimizer_steps_per_epoch=5,
        batches_per_epoch=5,
        num_processes=8,
    )
    with pytest.raises(RuntimeError, match="mismatches"):
        progress.validate(
            config_hash="changed",
            dataset_hash="dataset",
            target_steps=10,
            optimizer_steps_per_epoch=5,
            batches_per_epoch=5,
            num_processes=8,
        )
    assert next_training_position(3, 2, 5) == (3, 3)
    assert next_training_position(3, 4, 5) == (4, 0)


def test_progress_allows_explicit_target_extension() -> None:
    progress = TrainingProgress(
        config_hash="config",
        dataset_hash="dataset",
        target_steps=10,
        optimizer_steps_per_epoch=5,
        batches_per_epoch=5,
        num_processes=8,
        total_step_offset=0,
    )
    progress.global_step = 10
    progress.epoch = 2
    progress.validate(
        config_hash="config",
        dataset_hash="dataset",
        target_steps=20,
        optimizer_steps_per_epoch=5,
        batches_per_epoch=5,
        num_processes=8,
        allow_target_extension=True,
    )
    assert progress.target_steps == 20
    assert progress.global_step == 10
    assert progress.epoch == 2

    with pytest.raises(RuntimeError, match="cannot shrink"):
        progress.validate(
            config_hash="config",
            dataset_hash="dataset",
            target_steps=5,
            optimizer_steps_per_epoch=5,
            batches_per_epoch=5,
            num_processes=8,
            allow_target_extension=True,
        )


def test_legacy_progress_migrates_with_verified_config() -> None:
    original = TrainingProgress(
        config_hash="legacy-full-config",
        dataset_hash="dataset",
        target_steps=10,
        optimizer_steps_per_epoch=5,
        batches_per_epoch=5,
        num_processes=8,
        total_step_offset=0,
    )
    legacy_state = original.state_dict()
    legacy_state["format_version"] = 1

    restored = TrainingProgress(
        config_hash="new-compatible-config",
        dataset_hash="dataset",
        target_steps=20,
        optimizer_steps_per_epoch=5,
        batches_per_epoch=5,
        num_processes=8,
        total_step_offset=0,
    )
    restored.load_state_dict(legacy_state)
    restored.validate(
        config_hash="new-compatible-config",
        dataset_hash="dataset",
        target_steps=20,
        optimizer_steps_per_epoch=5,
        batches_per_epoch=5,
        num_processes=8,
        allow_target_extension=True,
        legacy_config_compatible=True,
    )
    assert restored.format_version == 2
    assert restored.loaded_format_version == 2
    assert restored.config_hash == "new-compatible-config"
    assert restored.target_steps == 20


def test_resume_config_hash_ignores_only_training_horizon() -> None:
    base = {
        "data": {"train_manifest": "same"},
        "training": {
            "num_train_epochs": 10,
            "learning_rate": 5e-6,
            "lr_scheduler": "constant_with_warmup",
        },
    }
    extended = copy.deepcopy(base)
    extended["training"]["num_train_epochs"] = 20
    assert (
        resume_compatible_config_hash(base)
        == resume_compatible_config_hash(extended)
    )

    extended["training"]["learning_rate"] = 1e-5
    assert (
        resume_compatible_config_hash(base)
        != resume_compatible_config_hash(extended)
    )


def test_resume_metadata_preserves_initial_config(tmp_path) -> None:
    config_v1 = tmp_path / "config_v1.json"
    config_v2 = tmp_path / "config_v2.json"
    split = tmp_path / "manifest_summary.json"
    config_v1.write_text('{"training":{"num_train_epochs":10}}')
    config_v2.write_text('{"training":{"num_train_epochs":20}}')
    split.write_text('{"train_frames":3}')
    output = tmp_path / "run"

    save_run_metadata(
        output,
        config_v1,
        split,
        {"seed": 7},
        train_size=3,
        val_size=1,
    )
    resume_state = output / "trainer_state_step000003"
    resume_state.mkdir()
    save_run_metadata(
        output,
        config_v2,
        split,
        {"seed": 7},
        train_size=3,
        val_size=1,
        resume_from=resume_state,
    )

    assert (output / "config.json").read_text() == config_v1.read_text()
    resume_configs = list(output.glob("config_resume_*.json"))
    assert len(resume_configs) == 1
    assert resume_configs[0].read_text() == config_v2.read_text()
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["saved_config"] == resume_configs[0].name
    assert metadata["resume_from"] == str(resume_state.resolve())


def test_temporal_sampler_resume_recreates_epoch_suffix() -> None:
    records = [
        {"sequence_id": "sequence-a", "frame_index": frame}
        for frame in range(20)
    ] + [
        {"sequence_id": "sequence-b", "frame_index": frame}
        for frame in range(20)
    ]
    original = TemporalChunkShuffleSampler(records, chunk_frames=5, seed=7)
    original.set_epoch(4)
    full_order = list(original)

    resumed = TemporalChunkShuffleSampler(records, chunk_frames=5, seed=7)
    resumed.set_epoch(4)
    resumed_order = list(resumed)
    assert resumed_order[13:] == full_order[13:]
