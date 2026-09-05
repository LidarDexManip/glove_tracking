"""Glove2Hand-style 2D Gaussians grounded on a learnable MANO surface."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .mano import LearnableMANO
from .math import matrix_to_quaternion, sh_degree2


def _triangle_frame(triangles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edge0 = triangles[:, 1] - triangles[:, 0]
    edge1 = triangles[:, 2] - triangles[:, 0]
    tangent0 = F.normalize(edge0, dim=-1)
    normal = F.normalize(torch.cross(edge0, edge1, dim=-1), dim=-1)
    tangent1 = F.normalize(torch.cross(normal, tangent0, dim=-1), dim=-1)
    frame = torch.stack((tangent0, tangent1, normal), dim=-1)
    coordinates = torch.stack((
        torch.stack((torch.linalg.vector_norm(edge0, dim=-1),
                     (edge1 * tangent0).sum(-1)), -1),
        torch.stack((torch.zeros_like(edge0[:, 0]),
                     (edge1 * tangent1).sum(-1)), -1),
    ), -2)
    return frame, coordinates


class PoseLighting(nn.Module):
    """Pose-conditioned palm/back degree-2 SH illumination."""

    def __init__(self, pose_dim: int = 18, hidden: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(pose_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * 9 * 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            neutral_dc = 1.0 / 0.28209479177387814
            bias = self.network[-1].bias.view(2, 9, 3)
            bias[:, 0] = neutral_dc

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.network(pose).view(-1, 2, 9, 3)


class GaussianHand(nn.Module):
    """Canonical MANO plus deforming, triangle-anchored 2D Gaussian surfels."""

    def __init__(self, mano_model_path: str, betas: torch.Tensor,
                 gaussians_per_face: int = 4, normal_offset_max: float = 0.003) -> None:
        super().__init__()
        self.mano = LearnableMANO(mano_model_path)
        self.register_buffer("subject_betas", betas.detach().clone().reshape(1, -1))
        face_count = self.mano.faces.shape[0]
        face_ids = torch.arange(face_count).repeat_interleave(gaussians_per_face)
        self.register_buffer("face_ids", face_ids)
        count = len(face_ids)

        patterns = torch.tensor([
            [1., 1., 1.], [3., 1., 1.], [1., 3., 1.], [1., 1., 3.]
        ])
        weights = patterns[torch.arange(count) % len(patterns)]
        self.barycentric_logits = nn.Parameter(weights.log())
        initial_scales = torch.tensor((0.0018, 0.0012)).log()
        self.log_scales = nn.Parameter(initial_scales.expand(count, -1).clone())
        self.angles = nn.Parameter(torch.zeros(count))
        self.opacity_logits = nn.Parameter(torch.full((count,), 2.0))
        self.albedo_logits = nn.Parameter(torch.zeros(count, 3))
        self.normal_offset_logits = nn.Parameter(torch.full((count,), -5.0))
        self.normal_offset_max = float(normal_offset_max)
        self.lighting = PoseLighting()

    def load_resizable_state_dict(self, state_dict: dict) -> None:
        """Load a checkpoint after adaptive control changed Gaussian count."""
        names=("barycentric_logits","log_scales","angles","opacity_logits",
               "albedo_logits","normal_offset_logits")
        for name in names:
            value=state_dict[name].to(self.subject_betas.device)
            if tuple(getattr(self,name).shape) != tuple(value.shape):
                setattr(self,name,nn.Parameter(torch.empty_like(value)))
        self.face_ids=state_dict["face_ids"].to(self.subject_betas.device).clone()
        self.load_state_dict(state_dict)

    @property
    def gaussian_count(self) -> int:
        return int(self.face_ids.numel())

    @staticmethod
    def _point_triangle_distance(points: torch.Tensor,
                                 triangles: torch.Tensor) -> torch.Tensor:
        """Squared distance for B points against F triangles."""
        p = points[:, None, :]
        a, b, c = (triangles[None, :, i, :] for i in range(3))
        ab, ac = b-a, c-a
        normal = F.normalize(torch.cross(ab, ac, dim=-1), dim=-1)
        signed = ((p-a)*normal).sum(-1)
        projected = p-signed[..., None]*normal
        v2 = projected-a
        d00=(ab*ab).sum(-1); d01=(ab*ac).sum(-1); d11=(ac*ac).sum(-1)
        d20=(v2*ab).sum(-1); d21=(v2*ac).sum(-1)
        denominator=(d00*d11-d01.square()).clamp_min(1e-12)
        wb=(d11*d20-d01*d21)/denominator
        wc=(d00*d21-d01*d20)/denominator
        inside=(wb >= 0) & (wc >= 0) & (wb+wc <= 1)
        plane_distance=signed.square()

        def segment_distance(x, y):
            edge=y-x
            t=((p-x)*edge).sum(-1)/(edge.square().sum(-1).clamp_min(1e-12))
            closest=x+t.clamp(0,1)[...,None]*edge
            return (p-closest).square().sum(-1)

        edge_distance=torch.minimum(
            segment_distance(a,b),torch.minimum(segment_distance(b,c),
                                                segment_distance(c,a)))
        return torch.where(inside,plane_distance,edge_distance)

    @torch.no_grad()
    def reassign_to_nearest_triangles(self) -> int:
        """Re-anchor surfels after canonical-mesh updates (paper adaptive control)."""
        canonical = self.mano.canonical(self.subject_betas)[0]
        all_triangles = canonical[self.mano.faces]
        old_triangles = all_triangles[self.face_ids]
        weights = self.barycentric_logits.softmax(-1)
        frame, _ = _triangle_frame(old_triangles)
        points = (old_triangles * weights[..., None]).sum(1)
        points += frame[:, :, 2] * (
            torch.sigmoid(self.normal_offset_logits) * self.normal_offset_max)[:, None]
        nearest=[]
        for begin in range(0,len(points),128):
            distances=self._point_triangle_distance(points[begin:begin+128],all_triangles)
            nearest.append(distances.argmin(-1))
        new_ids=torch.cat(nearest)
        tri=all_triangles[new_ids]
        a,b,c=tri[:,0],tri[:,1],tri[:,2]
        normal=F.normalize(torch.cross(b-a,c-a,dim=-1),dim=-1)
        projected=points-((points-a)*normal).sum(-1,keepdim=True)*normal
        v0,v1,v2=b-a,c-a,projected-a
        d00=(v0*v0).sum(-1);d01=(v0*v1).sum(-1);d11=(v1*v1).sum(-1)
        d20=(v2*v0).sum(-1);d21=(v2*v1).sum(-1)
        denominator=(d00*d11-d01.square()).clamp_min(1e-12)
        wb=(d11*d20-d01*d21)/denominator;wc=(d00*d21-d01*d20)/denominator
        bary=torch.stack((1-wb-wc,wb,wc),-1).clamp_min(1e-4)
        bary=bary/bary.sum(-1,keepdim=True)
        changed=int((new_ids != self.face_ids).sum())
        self.face_ids.copy_(new_ids);self.barycentric_logits.copy_(bary.log())
        return changed

    @torch.no_grad()
    def adaptive_density_control(self, optimizer: torch.optim.Optimizer,
                                 scores: torch.Tensor, split_fraction: float,
                                 max_gaussians: int,
                                 opacity_threshold: float = 0.02) -> dict:
        """Split high-gradient surfels and prune transparent ones."""
        old_count=self.gaussian_count
        if scores.shape != (old_count,):
            raise ValueError(f"score shape {scores.shape} != {(old_count,)}")
        keep=torch.sigmoid(self.opacity_logits) >= opacity_threshold
        if int(keep.sum()) < max(256,old_count//2):
            keep=torch.ones_like(keep)
        capacity=max(0,max_gaussians-int(keep.sum()))
        split_count=min(capacity,max(1,int(old_count*split_fraction)))
        eligible=torch.where(keep)[0]
        if split_count:
            split=eligible[torch.topk(scores[eligible],min(split_count,len(eligible))).indices]
        else:
            split=eligible[:0]
        base=torch.where(keep)[0]
        index_map=torch.cat((base,split))
        old_parameters={}
        names=("barycentric_logits","log_scales","angles","opacity_logits",
               "albedo_logits","normal_offset_logits")
        for name in names:
            old=getattr(self,name);old_parameters[name]=old
            values=old.detach()[index_map].clone()
            if len(split):
                tail=slice(len(base),None)
                if name=="log_scales":
                    values[tail]-=math.log(math.sqrt(2))
                elif name=="barycentric_logits":
                    pattern=torch.tensor((.12,-.06,-.06),device=values.device,
                                         dtype=values.dtype)
                    values[tail]+=pattern.roll(int(old_count%3))
            new=nn.Parameter(values)
            setattr(self,name,new)
            for group in optimizer.param_groups:
                group["params"]=[new if item is old else item for item in group["params"]]
            state=optimizer.state.pop(old,{})
            resized={}
            for key,value in state.items():
                if torch.is_tensor(value) and value.ndim and value.shape[0]==old_count:
                    resized[key]=value[index_map].clone()
                else:
                    resized[key]=value
            optimizer.state[new]=resized
        self.face_ids=self.face_ids[index_map].clone()
        return {"before":old_count,"after":self.gaussian_count,
                "split":len(split),"pruned":old_count-int(keep.sum())}

    def attributes(self, pca: torch.Tensor, wrist_q_wxyz: torch.Tensor,
                   wrist_t_world: torch.Tensor) -> dict[str, torch.Tensor]:
        if pca.ndim == 1:
            pca = pca[None]
            wrist_q_wxyz = wrist_q_wxyz[None]
            wrist_t_world = wrist_t_world[None]
        if len(pca) != 1:
            raise ValueError("Current renderer uses one camera/frame per call")
        betas = self.subject_betas.expand(len(pca), -1)
        canonical = self.mano.canonical(betas)[0]
        deformed = self.mano(pca, wrist_q_wxyz, wrist_t_world, betas)[0]
        faces = self.mano.faces[self.face_ids]
        tri_c, tri_d = canonical[faces], deformed[faces]
        frame_c, basis_c = _triangle_frame(tri_c)
        frame_d, basis_d = _triangle_frame(tri_d)
        eye2 = torch.eye(2, device=basis_c.device, dtype=basis_c.dtype)
        deformation = basis_d @ torch.linalg.inv(basis_c + 1e-6 * eye2)

        weights = self.barycentric_logits.softmax(-1)
        means = (tri_d * weights[..., None]).sum(1)
        normal = frame_d[:, :, 2]
        means = means + normal * (
            torch.sigmoid(self.normal_offset_logits) * self.normal_offset_max)[:, None]

        cosine, sine = self.angles.cos(), self.angles.sin()
        rotation2 = torch.stack((
            torch.stack((cosine, -sine), -1),
            torch.stack((sine, cosine), -1)), -2)
        scale2 = self.log_scales.exp()
        covariance = rotation2 @ torch.diag_embed(scale2.square()) @ rotation2.transpose(-1, -2)
        covariance = deformation @ covariance @ deformation.transpose(-1, -2)
        covariance = .5 * (covariance + covariance.transpose(-1, -2))
        a, b, d = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 1, 1]
        discriminant = torch.sqrt((a - d).square() + 4 * b.square() + 1e-12)
        eigenvalues = torch.stack(((a + d + discriminant) * .5,
                                   (a + d - discriminant) * .5), -1)
        theta = .5 * torch.atan2(2 * b, a - d)
        cos_theta, sin_theta = theta.cos(), theta.sin()
        eigenvectors = torch.stack((
            torch.stack((cos_theta, -sin_theta), -1),
            torch.stack((sin_theta, cos_theta), -1)), -2)
        tangent_axes = frame_d[:, :, :2] @ eigenvectors
        rotations = torch.cat((tangent_axes, normal[..., None]), -1)
        quaternions = matrix_to_quaternion(rotations)
        scales = torch.cat((eigenvalues.clamp_min(1e-12).sqrt(),
                            torch.full((len(means), 1), 1e-5, device=means.device,
                                       dtype=means.dtype)), -1)

        sh = self.lighting(torch.cat((pca, wrist_t_world), -1))[0]
        basis = sh_degree2(normal)
        palm_weight = torch.sigmoid(-20.0 * frame_c[:, 2, 2])
        coefficients = (palm_weight[:, None, None] * sh[0] +
                        (1 - palm_weight[:, None, None]) * sh[1])
        raw_illumination = torch.einsum("nk,nkc->nc", basis, coefficients)
        # A hard non-negative clamp creates a dead SH state: once all channels
        # cross below zero, image gradients can no longer recover lighting.
        illumination = F.softplus(raw_illumination, beta=5.0)
        # Multiplication in odds space is bounded, differentiable at both ends,
        # and is exactly neutral when illumination is one.
        colors = torch.sigmoid(
            self.albedo_logits + torch.log(illumination.clamp_min(1e-6)))
        return {
            "means": means, "quats": quaternions, "scales": scales,
            "opacities": torch.sigmoid(self.opacity_logits),
            "colors": colors, "normals": normal,
        }
