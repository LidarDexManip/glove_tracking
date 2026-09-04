"""Learnable per-training-frame MANO pose corrections."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

def axis_angle_to_quaternion(axis_angle):
    angle=torch.linalg.vector_norm(axis_angle,dim=-1,keepdim=True); half=angle*.5
    ratio = torch.sin(half) / angle.clamp_min(1e-8)
    scale=torch.where(angle>1e-8,ratio,.5-angle.square()/48)
    return torch.cat((torch.cos(half),axis_angle*scale),-1)

def quaternion_multiply(a,b):
    aw,ax,ay,az=a.unbind(-1); bw,bx,by,bz=b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,
        aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)

class PoseRefinement(nn.Module):
    def __init__(self, frame_count):
        super().__init__(); self.deltas=nn.Embedding(frame_count,21)
        nn.init.zeros_(self.deltas.weight)
    def forward(self, indices,pca,quaternion,translation):
        d=self.deltas(indices)
        q=F.normalize(quaternion_multiply(axis_angle_to_quaternion(d[...,15:18]),quaternion),dim=-1)
        return pca+d[...,:15],q,translation+d[...,18:21]
