import torch

from gaussian_hand.model.surface_gaussians import GaussianHand


def _minimal_gaussian_hand(count=4):
    model = GaussianHand.__new__(GaussianHand)
    torch.nn.Module.__init__(model)
    model.register_buffer("face_ids", torch.arange(count))
    model.barycentric_logits = torch.nn.Parameter(torch.zeros(count, 3))
    model.log_scales = torch.nn.Parameter(torch.zeros(count, 2))
    model.angles = torch.nn.Parameter(torch.zeros(count))
    model.opacity_logits = torch.nn.Parameter(torch.full((count,), 2.0))
    model.albedo_logits = torch.nn.Parameter(torch.zeros(count, 3))
    model.normal_offset_logits = torch.nn.Parameter(torch.zeros(count))
    return model


def test_density_control_respects_numeric_cap():
    model = _minimal_gaussian_hand()
    optimizer = torch.optim.Adam(model.parameters())
    result = model.adaptive_density_control(
        optimizer, torch.arange(4.0), split_fraction=0.5,
        max_gaussians=4, opacity_threshold=0.0)
    assert result["after"] == 4
    assert result["split"] == 0


def test_density_control_accepts_no_cap():
    model = _minimal_gaussian_hand()
    optimizer = torch.optim.Adam(model.parameters())
    result = model.adaptive_density_control(
        optimizer, torch.arange(4.0), split_fraction=0.5,
        max_gaussians=None, opacity_threshold=0.0)
    assert result["after"] == 6
    assert result["split"] == 2
