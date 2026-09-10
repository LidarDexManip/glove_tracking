"""CPU-only alignment check against the frozen naive MANO raster."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import torch

from gaussian_hand.export_diffusion_cache import CacheRenderer, DEFAULT_MODELS, ROOT, manifest_frames
from gaussian_hand.data.geometry import project_points, rasterize_silhouette
from gaussian_hand.render_scene import scaled_intrinsics
from gaussian_hand.render_sequence import nearest_hand_pose, world_to_camera, load_headset_pose_provider_from_csv


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(1)
    renderer=CacheRenderer(DEFAULT_MODELS,"cpu");rows=[]
    for sequence,frames in sorted(manifest_frames([args.manifest]).items()):
        source=ROOT/"data/HUGG_ARIA_PINHOLE"/sequence
        models=renderer.subject(sequence.split("_")[0])
        with (source/"frame_timestamps_214_1.csv").open() as f:
            timestamps=[int(r["timestamp_ns"]) for r in csv.DictReader(f)]
        calibrations=[json.loads(x) for x in (source/"frame_pinhole_calibration_214_1.jsonl").read_text().splitlines()]
        poses=sorted((json.loads(x) for x in (source/"mano_hand_pose_trajectory.jsonl").read_text().splitlines()),key=lambda x:x["timestamp_ns"])
        headset=load_headset_pose_provider_from_csv(str(source/"headset_trajectory.csv"))
        capture=cv2.VideoCapture(str(ROOT/"data/HUGG_ARIA_MANO_ALIGNED"/sequence/"alpha.mkv"))
        try:
            for frame,ts in sorted(frames.items()):
                assert timestamps[frame]==ts
                hand_poses=nearest_hand_pose([int(x["timestamp_ns"]) for x in poses],[x["hand_poses"] for x in poses],ts)
                cal=calibrations[frame]; camera=world_to_camera(cal,headset,ts)
                if not hand_poses or camera is None:raise RuntimeError(f"Missing GT: {sequence}:{frame}")
                support=np.zeros((512,512),np.uint8);available=[]
                for side,key in (("left","0"),("right","1")):
                    pose=hand_poses.get(key)
                    if pose is None:continue
                    model=models[side];tensor=lambda x:torch.tensor(x,dtype=torch.float32)
                    v=model.mano(tensor(pose["pose"]),tensor(pose["wrist_xform"]["q_wxyz"]),
                                 tensor(pose["wrist_xform"]["t_xyz"]),model.subject_betas[0])[0].numpy()
                    uv,z=project_points(v,camera,scaled_intrinsics(cal))
                    support|=rasterize_silhouette(uv,z,model.mano.faces.numpy(),512,512,1)
                    available.append(side)
                capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,old=capture.read()
                if not ok:raise RuntimeError("Cannot decode baseline alpha")
                old=old[...,0]>0;new=support>0
                iou=float((old&new).sum()/max(1,(old|new).sum()))
                rows.append({"sequence":sequence,"frame_index":frame,"timestamp_ns":ts,
                             "hands":available,"mano_support_iou":iou,
                             "baseline_area":int(old.sum()),"new_area":int(new.sum())})
        finally:capture.release()
        print(json.dumps(rows[-1]),flush=True)
    result={"status":"passed" if min(r["mano_support_iou"] for r in rows)>=.90 else "needs_review",
            "scope":"CPU geometry only; not a Gaussian appearance/rendering validation",
            "unique_frames":len(rows),"min_iou":min(r["mano_support_iou"] for r in rows),
            "mean_iou":float(np.mean([r["mano_support_iou"] for r in rows])),"frames":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="frames"}))
    if result["status"]!="passed":raise SystemExit(1)


if __name__=="__main__":main()
