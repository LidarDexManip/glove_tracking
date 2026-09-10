"""Summarize prepared hands and create diverse MANO/reference review sheets."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def view_direction(record):
    camera = np.linalg.inv(np.asarray(record["world_to_camera"]))[:3,3]
    d = camera - np.asarray(record["wrist_t_world"])
    d /= max(np.linalg.norm(d),1e-8)
    q = np.asarray(record["wrist_q_wxyz"]); q /= np.linalg.norm(q)
    w,x,y,z = q
    rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return rotation.T @ d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args(); cv2.setNumThreads(1)
    inventory=json.loads((args.root/"inventory.json").read_text())
    rows=[]
    for key in inventory["jobs"]:
        directory=args.root/key/"dataset"; manifest=directory/"frames.jsonl"
        if not (directory/"summary.json").exists():
            rows.append({"hand":key,"status":"not_prepared"});continue
        records=[json.loads(x) for x in manifest.read_text().splitlines()]
        counts=Counter(r["sequence"] for r in records)
        views=np.array([view_direction(r) for r in records])
        betas=np.asarray([r["mano_betas"] for r in records])
        variation=float(np.max(np.abs(betas-betas[0])))
        octants=np.sum((views>0)*np.array([1,2,4]),axis=1)
        side_min=min(float((views[:,2]>0).mean()),float((views[:,2]<=0).mean()))
        flags=[]
        if len(records)<500:flags.append("fewer_than_500_frames")
        if len(counts)<3:flags.append("fewer_than_3_sequences")
        if side_min<.05:flags.append("view_Z_hemisphere_imbalance")
        if variation>1e-5:flags.append("inconsistent_subject_betas")
        features=np.concatenate((np.asarray([r["mano_pca"] for r in records]),views*2),axis=1)
        features=(features-features.mean(0))/np.maximum(features.std(0),.1)
        chosen=[]
        for seq in sorted(counts):
            indices=[i for i,r in enumerate(records) if r["sequence"]==seq]
            chosen.append(indices[len(indices)//2])
        chosen=chosen[:24]
        while len(chosen)<min(24,len(records)):
            distances=np.min(((features[:,None]-features[chosen][None])**2).sum(-1),axis=1)
            distances[chosen]=-1
            chosen.append(int(distances.argmax()))
        tiles=[]
        for i in chosen:
            record=records[i]; rgb=cv2.imread(str(directory/record["image"]))
            mesh=cv2.imread(str(directory/record["mesh_mask"]),0)>0
            reference=cv2.imread(str(directory/record["reference_mask"]),0)>0
            tile=(rgb.astype(np.float32)*np.where(mesh[...,None],1.,.25)).astype(np.uint8)
            missed=mesh & ~reference
            tile[missed]=(tile[missed]*.5+np.array([0,0,128])).astype(np.uint8)
            tile=cv2.resize(tile,(192,192))
            cv2.putText(tile,f"{record['sequence'][-8:]}:{record['frame_index']}",(4,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1)
            cv2.putText(tile,f"vis={record['visible_mano_ratio']:.2f} sharp={record['sharpness']:.0f}",(4,185),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,255,0),1)
            tiles.append(tile)
        while len(tiles)%6:tiles.append(np.zeros_like(tiles[0]))
        sheet=np.concatenate([np.concatenate(tiles[i:i+6],axis=1) for i in range(0,len(tiles),6)],axis=0)
        cv2.imwrite(str(directory/"diverse_review.jpg"),sheet)
        row={"hand":key,"status":"prepared_with_warnings" if flags else "prepared",
             "frames":len(records),"sequence_counts":dict(counts),"warnings":flags,
             "view_octants":dict(Counter(map(int,octants))),
             "hand_view_positive_Z_fraction":float((views[:,2]>0).mean()),
             "view_metric_note":"Hand-coordinate view-direction proxy, not measured surface coverage",
             "subject_beta_max_variation":variation,"manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
             "contact_sheet":str(directory/"diverse_review.jpg")}
        rows.append(row)
    report={"hand_count":len(rows),"training_policy":"all accepted frames train",
            "formal_training_started":False,"hands":rows}
    (args.root/"readiness.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps([{k:r[k] for k in ("hand","status","frames","warnings") if k in r} for r in rows],indent=2))

if __name__=="__main__":main()
