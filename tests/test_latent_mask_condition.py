import torch

from hand_restoration.diffusion import LatentMaskConditionEmbedding


def test_latent_mask_projection_shape_and_zero_initialization() -> None:
    module = LatentMaskConditionEmbedding(output_channels=320)
    condition = torch.randn(2, 5, 64, 64)
    output = module(condition)
    assert output.shape == (2, 320, 64, 64)
    assert torch.count_nonzero(output) == 0


def test_latent_mask_projection_rejects_wrong_channel_count() -> None:
    module = LatentMaskConditionEmbedding(output_channels=32)
    try:
        module(torch.randn(1, 4, 8, 8))
    except ValueError as error:
        assert "Bx5xHxW" in str(error)
    else:
        raise AssertionError("four-channel input should be rejected")
