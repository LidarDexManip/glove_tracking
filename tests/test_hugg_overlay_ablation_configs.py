from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hand_restoration.hugg_aria_dataset import overlay_and_loss_masks


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/hand_restoration"
GAUSSIAN_CONFIG = (
    CONFIG_DIR
    / "hugg_aria_ablation_gaussian_weight10_lr5e6_constant_5epochs.json"
)
MANO_CONFIG = (
    CONFIG_DIR
    / "hugg_aria_ablation_mano_weight10_lr5e6_constant_5epochs.json"
)


def normalized(config: dict) -> dict:
    result = json.loads(json.dumps(config))
    result["data"].pop("gaussian_root")
    result["data"].pop("render_kind")
    result["training"].pop("output_dir")
    return result


def test_overlay_ablation_configs_only_change_render_source_and_output() -> None:
    gaussian = json.loads(GAUSSIAN_CONFIG.read_text())
    mano = json.loads(MANO_CONFIG.read_text())

    assert normalized(gaussian) == normalized(mano)
    assert gaussian["data"]["render_kind"] == "gaussian"
    assert mano["data"]["render_kind"] == "mano"
    assert gaussian["data"]["gaussian_root"] != mano["data"]["gaussian_root"]
    assert gaussian["training"]["output_dir"] != mano["training"]["output_dir"]
    assert gaussian["data"]["loss_mask_source"] == "sam"
    assert "ablation_common" in gaussian["data"]["train_manifest"]


def test_tuned_ablation_training_geometry() -> None:
    config = json.loads(GAUSSIAN_CONFIG.read_text())
    training = config["training"]

    assert training["batch_size"] == 32
    assert training["gradient_accumulation_steps"] == 1
    assert training["learning_rate"] == 5e-6
    assert training["lr_scheduler"] == "constant_with_warmup"
    assert training["warmup_steps"] == 100
    assert training["hand_loss_weight"] == 10.0
    assert training["chunk_shuffle_frames"] == 150
    assert training["num_train_epochs"] == 5
    assert training["save_full_state"] is True


def test_sam_loss_mask_is_independent_of_render_coverage() -> None:
    sam = np.asarray([[True, True], [False, True]])
    gaussian = np.asarray([[True, False], [False, True]])
    mano = np.asarray([[False, True], [False, True]])

    gaussian_overlay, gaussian_loss = overlay_and_loss_masks(
        gaussian, sam, "sam"
    )
    mano_overlay, mano_loss = overlay_and_loss_masks(mano, sam, "sam")

    assert not np.array_equal(gaussian_overlay, mano_overlay)
    assert np.array_equal(gaussian_loss, sam)
    assert np.array_equal(mano_loss, sam)
    legacy_overlay, legacy_loss = overlay_and_loss_masks(
        gaussian, sam, "overlay"
    )
    assert np.array_equal(legacy_loss, legacy_overlay)

