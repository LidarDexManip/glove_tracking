#!/usr/bin/env python3
"""Render a frame-aligned Gaussian hand video for every available MANO pose."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

for _name,_value in {"bool":bool,"int":int,"float":float,"complex":complex,
                     "object":object,"unicode":str,"str":str}.items():
    if _name not in np.__dict__: setattr(np,_name,_value)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"hot3d/hot3d"))
from data_loaders.HeadsetPose3dProvider import load_headset_pose_provider_from_csv
from projectaria_tools.core.sensor_data import TimeDomain,TimeQueryOptions

from gaussian_hand.data.geometry import crop_intrinsics,project_points,square_crop
from gaussian_hand.model import GaussianHand
from gaussian_hand.render import render_gaussian_hand

TOLERANCE_NS=10_000_000

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--sequence",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pinhole-root",type=Path,default=ROOT/"data/HUGG_ARIA_PINHOLE")
    p.add_argument("--size",type=int,default=512)
    p.add_argument("--fps",type=float,default=30)
    p.add_argument("--max-frames",type=int,default=0)
    p.add_argument("--start-frame",type=int,default=0)
    p.add_argument("--device",default="cuda")
    return p.parse_args()

def world_to_camera(calibration,provider,timestamp):
    result=provider.get_pose_at_timestamp(
        timestamp_ns=timestamp,time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,acceptable_time_delta=TOLERANCE_NS)
    if result is None:return None
    wd=np.asarray(result.pose3d.T_world_device.to_matrix(),np.float64)
    dc=np.asarray(calibration["T_device_camera"],np.float64)
    return np.linalg.inv(wd@dc)

def nearest_right_pose(timestamps,hands,timestamp):
    index=bisect.bisect_left(timestamps,timestamp)
    choices=[i for i in (index-1,index) if 0 <= i < len(timestamps)]
    if not choices:return None
    best=min(choices,key=lambda i:abs(timestamps[i]-timestamp))
    return hands[best] if abs(timestamps[best]-timestamp) <= TOLERANCE_NS else None

def ffmpeg_rgb(path,size,fps):
    return subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo",
        "-pix_fmt","rgb24","-s",f"{size}x{size}","-r",str(fps),"-i","-",
        "-an","-c:v","libx264","-preset","veryfast","-crf","18","-pix_fmt",
        "yuv420p",str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)

def ffmpeg_alpha(path,size,fps):
    return subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo",
        "-pix_fmt","gray","-s",f"{size}x{size}","-r",str(fps),"-i","-",
        "-an","-c:v","ffv1","-level","3",str(path)],
        stdin=subprocess.PIPE,stderr=subprocess.PIPE)

def main():
    args=arguments();args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    expected={"dataset":"HOT3D","subject_id":"P0001","handedness":"right"}
    if checkpoint.get("binding") != expected:
        raise ValueError(f"Checkpoint identity mismatch: {checkpoint.get('binding')}")
    config=checkpoint["config"];state=checkpoint["model"]
    betas=state["subject_betas"][0].to(args.device)
    model=GaussianHand(str(ROOT/config["mano_model"]),betas,
        int(config["gaussians_per_face"]),float(config["normal_offset_max_m"])).to(args.device)
    model.load_resizable_state_dict(state);model.eval()

    source=args.pinhole_root/args.sequence
    with (source/"frame_timestamps_214_1.csv").open(newline="") as f:
        timestamps=[int(row["timestamp_ns"]) for row in csv.DictReader(f)]
    with (source/"frame_pinhole_calibration_214_1.jsonl").open() as f:
        calibrations=[json.loads(line) for line in f if line.strip()]
    pose_items=[]
    with (source/"mano_hand_pose_trajectory.jsonl").open() as f:
        for line in f:
            item=json.loads(line)
            pose_items.append((int(item["timestamp_ns"]),item["hand_poses"].get("1")))
    pose_items.sort(key=lambda item:item[0])
    pose_timestamps=[item[0] for item in pose_items]
    pose_hands=[item[1] for item in pose_items]
    headset=load_headset_pose_provider_from_csv(str(source/"headset_trajectory.csv"))
    stop=(min(len(timestamps),args.start_frame+args.max_frames)
          if args.max_frames else len(timestamps))
    count=stop-args.start_frame
    rgb_encoder=ffmpeg_rgb(args.output/"render.mp4",args.size,args.fps)
    alpha_encoder=ffmpeg_alpha(args.output/"alpha.mkv",args.size,args.fps)
    blank_rgb=np.zeros((args.size,args.size,3),np.uint8)
    blank_alpha=np.zeros((args.size,args.size),np.uint8)
    manifest=[];valid_count=0
    try:
        for frame_index in range(args.start_frame,stop):
            timestamp,calibration=timestamps[frame_index],calibrations[frame_index]
            hand=nearest_right_pose(pose_timestamps,pose_hands,timestamp)
            tcw=world_to_camera(calibration,headset,timestamp)
            rgb,alpha=blank_rgb,blank_alpha;record={"frame_index":frame_index,
                "timestamp_ns":timestamp,"mano_valid":False}
            if hand is not None and tcw is not None:
                with torch.no_grad():
                    pca=torch.tensor(hand["pose"],device=args.device)
                    q=torch.tensor(hand["wrist_xform"]["q_wxyz"],device=args.device)
                    t=torch.tensor(hand["wrist_xform"]["t_xyz"],device=args.device)
                    vertices=model.mano(pca,q,t,betas)[0].cpu().numpy()
                    fx,fy=calibration["focal_lengths"];cx,cy=calibration["principal_point"]
                    k=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float64)
                    uv,z=project_points(vertices,tcw,k)
                    width,height=int(calibration["image_width"]),int(calibration["image_height"])
                    in_image=((z>.01)&(uv[:,0]>=0)&(uv[:,0]<width)&
                              (uv[:,1]>=0)&(uv[:,1]<height))
                    box=square_crop(uv[z>.01],width,height,1.35)
                    if box is not None and in_image.mean() >= .05:
                        ck=crop_intrinsics(k,box,args.size)
                        out=render_gaussian_hand(model,pca,q,t,
                            torch.tensor(tcw,device=args.device,dtype=torch.float32),
                            torch.tensor(ck,device=args.device,dtype=torch.float32),
                            args.size,args.size)
                        rgb=(out["rgb"].clamp(0,1).cpu().numpy()*255).astype(np.uint8)
                        alpha=(out["alpha"][...,0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)
                        record.update({"mano_valid":True,"crop_xyxy":list(box),
                                       "crop_intrinsics":ck.tolist()});valid_count+=1
            rgb_encoder.stdin.write(np.ascontiguousarray(rgb).tobytes())
            alpha_encoder.stdin.write(np.ascontiguousarray(alpha).tobytes())
            manifest.append(record)
    finally:
        for encoder in (rgb_encoder,alpha_encoder):
            encoder.stdin.close()
        errors=[]
        for name,encoder in (("rgb",rgb_encoder),("alpha",alpha_encoder)):
            error=encoder.stderr.read().decode();code=encoder.wait()
            if code:errors.append(f"{name}: {error}")
        if errors:raise RuntimeError("; ".join(errors))
    with (args.output/"frames.jsonl").open("w") as f:
        for item in manifest:f.write(json.dumps(item,separators=(",",":"))+"\n")
    summary={"format_version":1,"model_id":config["model_id"],
        "checkpoint":str(args.checkpoint),"sequence":args.sequence,
        "total_frames":count,"rendered_mano_frames":valid_count,
        "source_start_frame":args.start_frame,
        "frame_alignment":"manifest stores source frame_index; output frame 0 maps to source_start_frame",
        "crop_semantics":"per-frame 1.35x projected learnable MANO square"}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
