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


def test_sam_forearm_pixels_do_not_change_geometry_or_color_loss():
    from gaussian_hand.losses import reconstruction_loss
    torch.manual_seed(4)
    image = torch.rand(40, 40, 3)
    support = torch.zeros(40, 40, 1)
    support[8:32, 8:32] = 1
    rgb = (torch.rand(40,40,3)*support).requires_grad_()
    render = {"rgb":rgb, "alpha":support*.9, "mesh_mask":support,
              "distortion":torch.zeros_like(support)}
    loss, _ = reconstruction_loss(render,image,support)
    enlarged = torch.ones_like(support)
    changed = image.clone()
    changed[support[...,0] == 0] = 10
    other, _ = reconstruction_loss(render,changed,enlarged)
    torch.testing.assert_close(loss,other)
    loss.backward()
    assert rgb.grad[support[...,0] == 0].abs().max() == 0


def test_mano_pixels_outside_sam_do_not_change_color_loss():
    from gaussian_hand.losses import reconstruction_loss
    torch.manual_seed(5)
    image = torch.rand(40, 40, 3)
    support = torch.zeros(40, 40, 1)
    support[5:35, 5:35] = 1
    sam = torch.zeros_like(support)
    sam[10:30, 10:30] = 1
    rgb = (torch.rand(40, 40, 3) * support).requires_grad_()
    render = {
        "rgb": rgb,
        "alpha": support * .9,
        "mesh_mask": support,
        "distortion": torch.zeros_like(support),
    }
    loss, _ = reconstruction_loss(render, image, sam)
    changed = image.clone()
    ignored = (support[..., 0] > 0) & (sam[..., 0] == 0)
    changed[ignored] = 10
    other, _ = reconstruction_loss(render, changed, sam)
    torch.testing.assert_close(loss, other)
    loss.backward()
    assert rgb.grad[ignored].abs().max() == 0


def test_premultiplied_overlay_preserves_background_outside_support():
    from gaussian_hand.render import composite_over_background
    background = torch.rand(8,8,3)
    alpha = torch.zeros(8,8,1)
    alpha[2:6,2:6] = .5
    rgb = torch.ones_like(background)*.4*alpha
    result = composite_over_background(background,rgb,alpha)
    torch.testing.assert_close(result[0],background[0])
    torch.testing.assert_close(result[3,3],torch.full((3,),.2)+.5*background[3,3])
