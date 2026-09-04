"""Image reconstruction losses."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    p = window // 2
    mx, my = F.avg_pool2d(x, window, 1, p), F.avg_pool2d(y, window, 1, p)
    vx = F.avg_pool2d(x.square(), window, 1, p) - mx.square()
    vy = F.avg_pool2d(y.square(), window, 1, p) - my.square()
    vxy = F.avg_pool2d(x*y, window, 1, p) - mx*my
    score = ((2*mx*my + .01**2)*(2*vxy + .03**2) /
             ((mx.square()+my.square()+.01**2)*(vx+vy+.03**2)))
    return score.mean()

def reconstruction_loss(render, image, mask, dssim_weight=.2,
                        alpha_weight=.1, distortion_weight=1e-4):
    target, rgb = image*mask, render["rgb"]
    l1 = (rgb-target).abs().mean()
    dssim = 1-ssim(rgb,target)
    alpha = (render["alpha"]-mask).abs().mean()
    distortion = render["distortion"].abs().mean()
    total = ((1-dssim_weight)*l1+dssim_weight*dssim+
             alpha_weight*alpha+distortion_weight*distortion)
    return total, {"l1":l1,"dssim":dssim,"alpha":alpha,"distortion":distortion}
