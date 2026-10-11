"""Image reconstruction losses."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11, reduction=True) -> torch.Tensor:
    x, y = x.permute(2, 0, 1)[None].contiguous(), y.permute(2, 0, 1)[None].contiguous()
    p = window // 2
    mx, my = F.avg_pool2d(x, window, 1, p), F.avg_pool2d(y, window, 1, p)
    vx = F.avg_pool2d(x.square(), window, 1, p) - mx.square()
    vy = F.avg_pool2d(y.square(), window, 1, p) - my.square()
    vxy = F.avg_pool2d(x*y, window, 1, p) - mx*my
    score = ((2*mx*my + .01**2)*(2*vxy + .03**2) /
             ((mx.square()+my.square()+.01**2)*(vx+vy+.03**2)))
    return score.mean() if reduction else score.mean(1)[0]

def reconstruction_loss(render, image, mask, dssim_weight=.2,
                        alpha_weight=.1, distortion_weight=1e-4, supervision=None):
    if supervision is not None and "mesh_mask" not in render:
        target = image * mask
        valid = supervision[..., 0]
        l1 = ((render['rgb']-target).abs()*supervision).sum()/(3*supervision.sum()).clamp_min(1)
        full_window = (F.avg_pool2d(valid[None,None],11,1,5)[0,0]>.999).float()
        score = ssim(render['rgb'],target,reduction=False)
        dssim = ((1-score)*full_window).sum()/full_window.sum().clamp_min(1)
        alpha = ((render['alpha']-mask).abs()*supervision).sum()/supervision.sum().clamp_min(1)
        distortion = (render['distortion'].abs()*supervision).sum()/supervision.sum().clamp_min(1)
        total = (1-dssim_weight)*l1+dssim_weight*dssim+alpha_weight*alpha+distortion_weight*distortion
        return total, {'l1':l1,'dssim':dssim,'alpha':alpha,'distortion':distortion}
    if "mesh_mask" in render:
        support = render["mesh_mask"]
        reference = mask * support
        # Eroded SSIM windows must be wholly inside reliable color reference.
        valid = reference[..., 0]
        l1 = ((render["rgb"] - image).abs() * reference).sum() / (3 * reference.sum()).clamp_min(1)
        full_window = (F.avg_pool2d(valid[None, None], 11, 1, 5)[0, 0] > .999).float()
        score = ssim(render["rgb"], image, reduction=False)
        dssim = ((1-score) * full_window).sum() / full_window.sum().clamp_min(1)
        # Coverage is defined by MANO, never by the larger SAM/forearm region.
        alpha = ((render["alpha"] - support).abs() * support).sum() / support.sum().clamp_min(1)
        distortion = (render["distortion"].abs()*support).sum() / support.sum().clamp_min(1)
        total = (1-dssim_weight)*l1 + dssim_weight*dssim + alpha_weight*alpha + distortion_weight*distortion
        return total, {"l1":l1,"dssim":dssim,"alpha":alpha,"distortion":distortion}
    target, rgb = image*mask, render["rgb"]
    l1 = (rgb-target).abs().mean()
    dssim = 1-ssim(rgb,target)
    alpha = (render["alpha"]-mask).abs().mean()
    distortion = render["distortion"].abs().mean()
    total = ((1-dssim_weight)*l1+dssim_weight*dssim+
             alpha_weight*alpha+distortion_weight*distortion)
    return total, {"l1":l1,"dssim":dssim,"alpha":alpha,"distortion":distortion}
