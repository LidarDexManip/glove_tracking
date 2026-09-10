"""Prepare all GT Aria subject/hand pairs, CPU only; never starts training."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/gaussian_hand/pretrain_v2")
    parser.add_argument("--subject", action="append")
    args = parser.parse_args()
    definitions = json.loads((ROOT/"data/clip_definitions.json").read_text())
    split = json.loads((ROOT/"data/clip_splits.json").read_text())
    train_sequences = {definitions[str(i)]["sequence_id"] for i in split["train"]["Aria"]}
    subjects = defaultdict(list)
    excluded = []
    for directory in sorted((ROOT/"data/HUGG_ARIA_PINHOLE").glob("P*")):
        metadata = json.loads((directory/"source_metadata.json").read_text())
        if directory.name not in train_sequences or metadata.get("have_hand_object_pose_gt") is not True:
            excluded.append(directory.name)
            continue
        if not (directory/"mano_hand_pose_trajectory.jsonl").stat().st_size:
            raise ValueError(f"Empty GT MANO: {directory}")
        sam = ROOT/"outputs/sam2_hugg_aria_masks_v3_pilot"/directory.name
        if not (sam/"_SUCCESS.json").is_file():
            raise ValueError(f"SAM incomplete: {directory.name}")
        subjects[metadata["participant_id"]].append(directory.name)
    args.output.mkdir(parents=True, exist_ok=True)
    config_dir = ROOT/"gaussian_hand/configs/pretrain_v2"
    config_dir.mkdir(parents=True, exist_ok=True)
    base_train = json.loads((ROOT/"gaussian_hand/configs/train/P0001_right_512.json").read_text())
    jobs = []
    for subject, sequences in sorted(subjects.items()):
        if args.subject and subject not in args.subject: continue
        for hand in ("left","right"):
            key = f"{subject}_{hand}"
            output = args.output/key
            binding = {"dataset":"HOT3D","subject_id":subject,"handedness":hand}
            model_id = f"hot3d-{subject}-{hand}-mano-surface-v2"
            config = {
                "binding":{**binding,"model_id":model_id},
                "pinhole_root":"data/HUGG_ARIA_PINHOLE", "raw_root":"data/raw/hot3d_aria",
                "sequences":sequences,
                "selection_policy":{
                    "min_mano_inside_image":.98,"min_mano_inside_crop":.995,
                    "min_visible_mano_silhouette_ratio":.90,
                    "min_projected_hand_area_fraction":.01,
                    "occlusion_policy":"SAM-visible coverage of projected MANO; no raw object bbox proxy"},
                "crop":{"scale_around_projected_mano":1.35,"output_size":512},
                "split_policy":{"train_sequences":sequences,"validation_sequences":[],"test_sequences":[]},
                "sampling":{"audit_mask_downsample":4,"minimum_frame_gap":1},
                "photometric_filter":{"min_laplacian_variance":8.,"max_clipped_fraction":.05}}
            config_path = config_dir/f"{key}.json"
            config_path.write_text(json.dumps(config,indent=2)+"\n")
            train = json.loads(json.dumps(base_train))
            train.update(binding=binding,model_id=model_id,dataset_root=str(output/"dataset"),
                         mano_model=f"mano_v1_2/models/MANO_{hand.upper()}.pkl",
                         strict_surface=True,normal_offset_max_m=0.,canonical_mesh_start_step=0)
            train["learning_rates"].update(canonical_mesh=0.,pose_delta=0.)
            train["reproduction_assumptions"] = [
                "Fixed GT MANO mesh and pose for source-video overlay; no canonical or pose corrections",
                "Gaussian centers exactly on MANO triangles; scales bounded by local triangle altitude and 4mm",
                "MANO silhouette is rendering support and alpha coverage target; SAM only selects color reference",
                "Photometric loss on visible MANO/SAM intersection, SSIM only on fully supported windows",
                "Existing SH appearance model retained; no manual relighting interface"]
            (config_dir/f"{key}_train.json").write_text(json.dumps(train,indent=2)+"\n")
            jobs.append((key,config_path,output))
    inventory = {"included_subjects":dict(subjects),"excluded_no_gt_or_test":excluded,
                 "jobs":[key for key,_,_ in jobs],"policy":"all accepted frames train; CPU preparation only"}
    (args.output/"inventory.json").write_text(json.dumps(inventory,indent=2)+"\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    def worker(job):
        key,config,output = job
        output.mkdir(parents=True,exist_ok=True)
        started = time.monotonic()
        commands = [
            [sys.executable,"-m","gaussian_hand.data.audit_hot3d","--config",str(config),"--output",str(output/"audit")],
            [sys.executable,"-m","gaussian_hand.data.export_crops","--config",str(config),"--manifest",str(output/"audit/frames.jsonl"),"--output",str(output/"dataset"),"--contact-count","48"]]
        status = "prepared"
        for phase, command in zip(("audit","export"),commands):
            with (output/f"{phase}.log").open("w") as log:
                result = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                status = f"{phase}_failed"
                break
        report = {"hand":key,"status":status,"elapsed_seconds":time.monotonic()-started}
        if status == "prepared":
            summary = json.loads((output/"dataset/summary.json").read_text())
            report.update(frames=summary["frames"],photometric_rejections=summary["photometric_rejections"])
        print(json.dumps(report),flush=True)
        return report
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(worker,jobs))
    (args.output/"preparation_report.json").write_text(json.dumps(results,indent=2)+"\n")
    if any(r["status"]!="prepared" for r in results): raise SystemExit(1)

if __name__ == "__main__": main()
