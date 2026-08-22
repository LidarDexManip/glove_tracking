from __future__ import annotations

import pytest
import torch

from hand_restoration.diffusion import LOSS_STAT_NAMES, regional_weighted_mse
from train_hand_restorer import summarize_loss_statistics


def test_regional_weighted_mse_reports_hand_and_background() -> None:
    squared_error = torch.ones((1, 2, 2, 2))
    squared_error[0, :, 0, 0] = 4.0
    edit_mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

    loss, statistics = regional_weighted_mse(
        squared_error, edit_mask, hand_loss_weight=10.0
    )
    metrics = summarize_loss_statistics(statistics)

    expected = (4.0 * 2 * 10 + 1.0 * 2 * 3) / (2 * 10 + 2 * 3)
    assert len(statistics) == len(LOSS_STAT_NAMES)
    assert loss.item() == pytest.approx(expected)
    assert metrics["unweighted_loss"] == pytest.approx(1.75)
    assert metrics["hand_region_loss"] == pytest.approx(4.0)
    assert metrics["background_region_loss"] == pytest.approx(1.0)
    assert metrics["hand_latent_fraction"] == pytest.approx(0.25)


def test_regional_weighted_mse_without_mask_matches_plain_mse() -> None:
    squared_error = torch.tensor([[[[1.0, 3.0]]]])
    loss, statistics = regional_weighted_mse(
        squared_error, edit_mask=None, hand_loss_weight=10.0
    )
    metrics = summarize_loss_statistics(statistics)

    assert loss.item() == pytest.approx(2.0)
    assert metrics["unweighted_loss"] == pytest.approx(2.0)
    assert metrics["background_region_loss"] == pytest.approx(2.0)
    assert metrics["hand_region_loss"] != metrics["hand_region_loss"]
