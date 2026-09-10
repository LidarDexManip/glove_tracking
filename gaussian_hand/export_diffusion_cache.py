"""Render new Gaussian checkpoints for frozen diffusion manifests, never train."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np
import torch

from gaussian_hand.model import GaussianHand
from gaussian_hand.render_scene import render_scene, scaled_intrinsics
from gaussian_hand.render_sequence import (nearest_hand_pose, world_to_camera,
    load_headset_pose_provider_from_csv, ffmpeg_rgb, ffmpeg_alpha)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "outputs/gaussian_hand/train_mano_surface_v2_30k"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_frames(paths):
    result = defaultdict(dict)
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                r = json.loads(line)
                seq, frame, timestamp = r["sequence_id"], int(r["frame_index"]), int(r["timestamp_ns"])
                previous = result[seq].get(frame)
                if previous is not None and previous != timestamp:
                    raise ValueError("Contradictory frame timestamps in manifests")
                result[seq][frame] = timestamp
    return result


def fingerprint():
    files = [Path(__file__), ROOT/"gaussian_hand/render_scene.py",
             ROOT/"gaussian_hand/render_sequence.py", ROOT/"gaussian_hand/model/surface_gaussians.py",
             ROOT/"gaussian_hand/model/mano.py", ROOT/"gaussian_hand/data/geometry.py"]
    return hashlib.sha256("".join(sha256(p) for p in files).encode()).hexdigest()


class CacheRenderer:
    def __init__(self, model_root, device):
        self.root, self.device = Path(model_root), device
        self.models, self.hashes = {}, {}

    def subject(self, subject):
        for side in ("left", "right"):
            key = f"{subject}_{side}"
            if key in self.models:
                continue
            path = self.root/key/"step_030000.pt"
            ck = torch.load(path, map_location=self.device, weights_only=False)
            if ck["binding"] != {"dataset": "HOT3D", "subject_id": subject, "handedness": side}:
                raise ValueError(f"Checkpoint identity mismatch: {path}")
            if not ck["config"].get("strict_surface") or ck["step"] != 30000:
                raise ValueError(f"Unexpected checkpoint: {path}")
            model = GaussianHand.from_config(ck["config"], ck["model"]["subject_betas"][0], ROOT).to(self.device)
            model.load_resizable_state_dict(ck["model"])
            if torch.count_nonzero(model.mano.canonical_delta).item():
                raise ValueError("Nonzero geometry correction")
            model.eval()
            self.models[key], self.hashes[key] = model, sha256(path)
        return {side: self.models[f"{subject}_{side}"] for side in ("left", "right")}

    def sequence(self, sequence, required, output_root, storage):
        source = ROOT/"data/HUGG_ARIA_PINHOLE"/sequence
        metadata = json.loads((source/"source_metadata.json").read_text())
        subject = sequence.split("_")[0]
        if metadata.get("participant_id") != subject or not metadata.get("have_hand_object_pose_gt"):
            raise ValueError(f"Not subject-bound GT data: {sequence}")
        models = self.subject(subject)
        with (source/"frame_timestamps_214_1.csv").open() as f:
            timestamps = [int(r["timestamp_ns"]) for r in csv.DictReader(f)]
        calibrations = [json.loads(x) for x in (source/"frame_pinhole_calibration_214_1.jsonl").read_text().splitlines()]
        if len(timestamps) != len(calibrations):
            raise ValueError(f"Camera/frame sidecar mismatch: {sequence}")
        for frame, ts in required.items():
            if frame < 0 or frame >= len(timestamps) or timestamps[frame] != ts:
                raise ValueError(f"Manifest/source frame alignment mismatch: {sequence}:{frame}")
        pose_items = sorted((json.loads(x) for x in (source/"mano_hand_pose_trajectory.jsonl").read_text().splitlines()), key=lambda x: x["timestamp_ns"])
        pose_times = [int(x["timestamp_ns"]) for x in pose_items]
        poses = [x["hand_poses"] for x in pose_items]
        headset = load_headset_pose_provider_from_csv(str(source/"headset_trajectory.csv"))
        contract = {"format_version": 1, "sequence": sequence, "subject_id": subject,
                    "rgb_semantics": "premultiplied", "storage": storage, "output_size": 512,
                    "frame_alignment": "output_frame_index_equals_source_frame_index",
                    "requested_frames_sha256": hashlib.sha256(json.dumps(sorted(required.items())).encode()).hexdigest(),
                    "checkpoint_sha256": {side: self.hashes[f"{subject}_{side}"] for side in models},
                    "renderer_sha256": fingerprint(), "source_frames": len(timestamps)}
        output = output_root/sequence
        if output.exists():
            success = output/"_SUCCESS.json"
            if not success.exists():
                raise FileExistsError(f"Incomplete cache already exists: {output}")
            saved = json.loads(success.read_text())
            if any(saved.get(k) != v for k, v in contract.items()):
                raise ValueError(f"Stale cache: {output}")
            return saved
        partial = Path(tempfile.mkdtemp(prefix=f".{sequence}.partial.", dir=output_root))
        (partial/"frames").mkdir()
        encoders = []
        if storage == "aligned_video":
            encoders = [ffmpeg_rgb(partial/"reconstruction.mp4",512,30), ffmpeg_alpha(partial/"alpha.mkv",512,30)]
        blank_rgb, blank_alpha = np.zeros((512,512,3),np.uint8), np.zeros((512,512),np.uint8)
        indices = range(len(timestamps)) if encoders else sorted(required)
        diagnostics = []
        started = time.monotonic()
        try:
            for frame in indices:
                rgb, alpha = blank_rgb, blank_alpha
                if frame in required:
                    ts, cal = timestamps[frame], calibrations[frame]
                    hand_poses = nearest_hand_pose(pose_times, poses, ts)
                    camera = world_to_camera(cal, headset, ts)
                    if not hand_poses or camera is None:
                        raise RuntimeError(f"Required frame lacks pose/camera: {sequence}:{frame}")
                    hands, available = [], []
                    for side, raw_key in (("left","0"),("right","1")):
                        pose = hand_poses.get(raw_key)
                        if pose is not None:
                            tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=self.device)
                            hands.append((models[side], tensor(pose["pose"]), tensor(pose["wrist_xform"]["q_wxyz"]), tensor(pose["wrist_xform"]["t_xyz"])))
                            available.append(side)
                    if not hands:
                        raise RuntimeError(f"No posed hand at required frame: {sequence}:{frame}")
                    result = render_scene(hands, torch.as_tensor(camera,dtype=torch.float32,device=self.device),
                                          torch.as_tensor(scaled_intrinsics(cal),device=self.device))
                    if not torch.isfinite(result["rgb"]).all() or not torch.isfinite(result["alpha"]).all():
                        raise RuntimeError(f"Non-finite render: {sequence}:{frame}")
                    rgb = (result["rgb"].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                    alpha = (result["alpha"][...,0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                    support = result["mesh_mask"][...,0].cpu().numpy() > 0
                    assert not rgb[~support].any() and not alpha[~support].any()
                    rgb[alpha == 0] = 0
                    if not alpha.any():
                        raise RuntimeError(f"Empty render at required frame: {sequence}:{frame}")
                    diagnostics.append({"frame_index":frame,"timestamp_ns":ts,"hands":available,
                                        "support_area":int(support.sum()),"alpha_area":int((alpha>0).sum())})
                    if not encoders:
                        for name, image in (("rgb",rgb[...,::-1]),("alpha",alpha),("support",support.astype(np.uint8)*255)):
                            if not cv2.imwrite(str(partial/"frames"/f"{frame:06d}_{name}.png"),image):
                                raise RuntimeError("Sparse cache write failed")
                for encoder, image in zip(encoders,(rgb,alpha)):
                    encoder.stdin.write(np.ascontiguousarray(image).tobytes())
        finally:
            for encoder in encoders:
                encoder.stdin.close()
            errors = [(encoder.stderr.read().decode(),encoder.wait()) for encoder in encoders]
            if any(code for _,code in errors):
                raise RuntimeError(f"Cache encoding failed: {errors}")
        if encoders:
            for name in ("reconstruction.mp4","alpha.mkv"):
                probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                    "-show_entries","stream=nb_read_frames","-of","json",str(partial/name)],text=True))
                if int(probe["streams"][0]["nb_read_frames"]) != len(timestamps):
                    raise RuntimeError("Encoded frame count mismatch")
        (partial/"rendered_frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in diagnostics))
        result = {**contract,"rendered_frames":len(diagnostics),"elapsed_seconds":time.monotonic()-started,
                  "blank_frames":len(timestamps)-len(diagnostics) if encoders else 0}
        (partial/"_SUCCESS.json").write_text(json.dumps(result,indent=2)+"\n")
        partial.rename(output)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model-root",type=Path,default=DEFAULT_MODELS)
    p.add_argument("--storage",choices=("sparse_frames","aligned_video"),required=True)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=1)
    p.add_argument("--sequence",action="append")
    p.add_argument("--preflight-only",action="store_true")
    args = p.parse_args()
    cv2.setNumThreads(1); torch.set_num_threads(1)
    frames = manifest_frames(args.manifest)
    keys = [s for i,s in enumerate(sorted(frames)) if i%args.shards==args.shard and (not args.sequence or s in args.sequence)]
    if args.preflight_only:
        print(json.dumps({"sequences":len(keys),"unique_frames":sum(len(frames[s]) for s in keys),"storage":args.storage}))
        return
    args.output.mkdir(parents=True,exist_ok=True)
    renderer = CacheRenderer(args.model_root,"cuda")
    for seq in keys:
        result = renderer.sequence(seq,frames[seq],args.output,args.storage)
        print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
