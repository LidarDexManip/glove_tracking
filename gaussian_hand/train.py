#!/usr/bin/env python3
"""Train the subject-bound Glove2Hand-style Gaussian Hand."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from gaussian_hand.data.dataset import GaussianHandDataset
from gaussian_hand.losses import reconstruction_loss
from gaussian_hand.model import GaussianHand
from gaussian_hand.model.pose_refinement import PoseRefinement
from gaussian_hand.render import render_gaussian_hand

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=None)
    parser.add_argument("--resolution",type=int,default=None)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--seed",type=int,default=20260904)
    parser.add_argument("--log-every",type=int,default=10)
    parser.add_argument("--checkpoint-every",type=int,default=None)
    parser.add_argument("--adaptive-every",type=int,default=None)
    parser.add_argument("--resume",type=Path,default=None)
    return parser.parse_args()


def resolve(path: str) -> Path:
    value=Path(path)
    return value if value.is_absolute() else ROOT/value


def save_checkpoint(path,step,model,pose_refinement,optimizer,config,manifest):
    records=[json.loads(line) for line in manifest.open() if line.strip()]
    try:
        commit=subprocess.check_output(
            ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        commit="unknown"
    mano_path=resolve(config["mano_model"])
    payload={
        "format_version":1,"step":step,"model_id":config["model_id"],
        "binding":{"dataset":"HOT3D","subject_id":"P0001","handedness":"right"},
        "source_sequences":sorted({item["sequence"] for item in records}),
        "config":config,
        "config_sha256":hashlib.sha256(
            json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "code_commit":commit,"mano_model":str(mano_path),
        "mano_sha256":hashlib.sha256(mano_path.read_bytes()).hexdigest(),
        "manifest":str(manifest),
        "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model":model.state_dict(),"pose_refinement":pose_refinement.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    torch.save(payload,path)


@torch.no_grad()
def validate(model, dataset, device, resolution, loss_config, max_frames=32):
    model.eval()
    indices=np.linspace(0,len(dataset)-1,min(max_frames,len(dataset))).astype(int)
    losses=[];mses=[]
    for index in indices:
        sample=dataset[int(index)]
        x={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in sample.items()}
        render=render_gaussian_hand(model,x["mano_pca"],x["wrist_q_wxyz"],
            x["wrist_t_world"],x["world_to_camera"],x["intrinsics"],
            resolution,resolution)
        loss,_=reconstruction_loss(render,x["image"],x["mask"],**loss_config)
        target=x["image"]*x["mask"]
        losses.append(float(loss));mses.append(float((render["rgb"]-target).square().mean()))
    model.train()
    mse=max(float(np.mean(mses)),1e-12)
    return {"frames":len(indices),"loss":float(np.mean(losses)),
            "psnr":float(-10*np.log10(mse))}


def main():
    args=arguments()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    config=json.loads(args.config.read_text())
    steps=args.steps or int(config["steps"])
    resolution=args.resolution or int(config["resolution"])
    dataset_root=resolve(config["dataset_root"])
    dataset=GaussianHandDataset(dataset_root,"train",resolution)
    validation_dataset=GaussianHandDataset(dataset_root,"val",resolution)
    device=torch.device(args.device)
    first=dataset[0]
    model=GaussianHand(str(resolve(config["mano_model"])),
        first["mano_pca"].new_tensor(
            json.loads((dataset_root/"frames.jsonl").open().readline())["mano_betas"]),
        int(config["gaussians_per_face"]),float(config["normal_offset_max_m"])).to(device)
    pose_refinement=PoseRefinement(len(dataset)).to(device)
    resume=None
    if args.resume is not None:
        resume=torch.load(args.resume,map_location=device,weights_only=False)
        if resume.get("model_id") != config["model_id"]:
            raise ValueError("Resume checkpoint model_id mismatch")
        model.load_resizable_state_dict(resume["model"])
        pose_refinement.load_state_dict(resume["pose_refinement"])
    lr=config["learning_rates"]
    optimizer=torch.optim.Adam([
        {"params":[model.barycentric_logits,model.log_scales,model.angles,
                   model.normal_offset_logits],"lr":lr["surface"]},
        {"params":[model.albedo_logits],"lr":lr["albedo"]},
        {"params":[model.opacity_logits],"lr":lr["opacity"]},
        {"params":[model.mano.canonical_delta],"lr":lr["canonical_mesh"]},
        {"params":model.lighting.parameters(),"lr":lr["lighting"]},
        {"params":pose_refinement.parameters(),"lr":lr["pose_delta"]},
    ],eps=1e-8)
    start_step=0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        start_step=int(resume["step"])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"resolved_config.json").write_text(json.dumps({
        **config,"steps":steps,"resolution":resolution,"seed":args.seed},indent=2)+"\n")
    checkpoint_every=args.checkpoint_every or int(config["checkpoint_every"])
    started=time.perf_counter(); measured_start=None; measured_steps=0
    adaptive_every=(args.adaptive_every if args.adaptive_every is not None
                    else int(config.get("adaptive_control",{}).get("every",0)))
    gradient_scores=torch.zeros(model.gaussian_count,device=device)
    gradient_samples=0
    for step in range(start_step+1,steps+1):
        sample=dataset[random.randrange(len(dataset))]
        tensors={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in sample.items()}
        index=torch.tensor([tensors["dataset_index"]],device=device)
        pca,q,t=pose_refinement(index,tensors["mano_pca"][None],
                                tensors["wrist_q_wxyz"][None],
                                tensors["wrist_t_world"][None])
        render=render_gaussian_hand(model,pca[0],q[0],t[0],
            tensors["world_to_camera"],tensors["intrinsics"],resolution,resolution)
        loss,parts=reconstruction_loss(render,tensors["image"],tensors["mask"],
                                       **config["loss"])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if step < int(config.get("canonical_mesh_start_step", 0)):
            model.mano.canonical_delta.grad = None
        with torch.no_grad():
            gradient_scores += model.barycentric_logits.grad.norm(dim=-1)
            gradient_scores += .25 * model.log_scales.grad.norm(dim=-1)
            gradient_samples += 1
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(pose_refinement.parameters()),
            10.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            model.log_scales.clamp_(-10.0, -3.0)
            model.barycentric_logits.clamp_(-12.0, 12.0)
            model.opacity_logits.clamp_(-12.0, 12.0)
            model.mano.canonical_delta.clamp_(-0.001, 0.001)
            pose_refinement.deltas.weight[:, :15].clamp_(-0.5, 0.5)
            pose_refinement.deltas.weight[:, 15:18].clamp_(-0.2, 0.2)
            pose_refinement.deltas.weight[:, 18:21].clamp_(-0.01, 0.01)
        if step==6:
            torch.cuda.synchronize(device); measured_start=time.perf_counter()
        if step>6:
            measured_steps+=1
        if adaptive_every and step%adaptive_every==0:
            control=config.get("adaptive_control",{})
            density=model.adaptive_density_control(
                optimizer,gradient_scores/max(gradient_samples,1),
                float(control.get("split_fraction",.05)),
                int(control.get("max_gaussians",20000)),
                float(control.get("opacity_prune_threshold",.02)
                      if step >= int(control.get("prune_start_step",10000)) else 0))
            changed=model.reassign_to_nearest_triangles()
            print(json.dumps({"step":step,"adaptive_density":density,
                              "adaptive_reassigned":changed}),flush=True)
            gradient_scores=torch.zeros(model.gaussian_count,device=device)
            gradient_samples=0
        if step%args.log_every==0 or step==1:
            elapsed=time.perf_counter()-started
            record={"step":step,"loss":float(loss.detach()),
                    **{k:float(v.detach()) for k,v in parts.items()},
                    "seconds_per_step":elapsed/step,
                    "cuda_peak_gib":torch.cuda.max_memory_allocated(device)/2**30}
            if measured_start is not None and measured_steps:
                torch.cuda.synchronize(device)
                record["warm_seconds_per_step"]=(time.perf_counter()-measured_start)/measured_steps
            print(json.dumps(record),flush=True)
        if step%int(config["validate_every"])==0:
            metrics=validate(model,validation_dataset,device,resolution,
                             config["loss"],int(config.get("validation_frames",32)))
            print(json.dumps({"step":step,"validation":metrics}),flush=True)
        if step%checkpoint_every==0 or step==steps:
            save_checkpoint(args.output/f"step_{step:06d}.pt",step,model,
                pose_refinement,optimizer,config,dataset_root/"frames.jsonl")
    print(json.dumps({"status":"complete","steps":steps,
                      "elapsed_seconds":time.perf_counter()-started}),flush=True)


if __name__=="__main__":
    main()
