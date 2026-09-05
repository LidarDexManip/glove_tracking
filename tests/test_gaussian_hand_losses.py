import pytest
import torch

from gaussian_hand.losses import ssim


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ssim_intensity_gradient_matches_finite_difference(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # HWC inputs become channels-last strided NCHW after permutation. This
    # exercises the actual image layout that previously gave incorrect CUDA
    # avg_pool2d backward gradients in the installed environment.
    torch.manual_seed(9)
    image = torch.rand(40, 40, 3, device=device) * 0.6 + 0.2
    target = image * 0.75
    scale = torch.tensor(1.0, device=device, requires_grad=True)
    loss = 1 - ssim(image * scale, target)
    loss.backward()
    with torch.no_grad():
        epsilon = 0.005
        plus = 1 - ssim(image * (1 + epsilon), target)
        minus = 1 - ssim(image * (1 - epsilon), target)
        numerical = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(scale.grad, numerical, rtol=0.01, atol=0.002)
