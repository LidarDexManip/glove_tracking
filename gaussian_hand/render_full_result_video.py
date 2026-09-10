"""Render complete sequences into one original/Gaussian video; no training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from gaussian_hand.export_diffusion_cache import CacheRenderer, DEFAULT_MODELS, ROOT
from gaussian_hand.make_subject_result_video import fit_panel
from gaussian_hand.render_scene import render_scene, scaled_intrinsics
from gaussian_hand.render_sequence import nearest_hand_pose, world_to_camera, load_headset_pose_provider_from_csv


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence",action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--size",type=int,default=1024)
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(1)
    if args.output.exists():raise FileExistsError(args.output)
    if len({s.split('_')[0] for s in args.sequence}) != len(args.sequence):
        raise ValueError("Expected one sequence per subject")
    torch.cuda.set_per_process_memory_fraction(.05)
    renderer=CacheRenderer(DEFAULT_MODELS,"cuda")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    counts={s:sum(1 for _ in (ROOT/"data/HUGG_ARIA_PINHOLE"/s/"frame_timestamps_214_1.csv").open())-1 for s in args.sequence}
    total=sum(counts.values());size=args.size;header=64
    state={"status":"running","gpu_physical_index":5,"training":False,"expected_frames":total,
           "expected_video_seconds":total/30,"completed_frames":0,"segments":[],
           "layout":"Original full video | both Gaussian hands, black background, same camera",
           "geometry_note":"strict MANO support; no SAM clipping or scene object occlusion"}
    started=time.monotonic();state_path=args.output.with_suffix('.status.json')

    def save():
        elapsed=time.monotonic()-started
        done=state["completed_frames"]
        state.update(elapsed_seconds=elapsed,frames_per_second=done/max(elapsed,1e-6),
                     estimated_remaining_seconds=(total-done)*elapsed/done if done else None)
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+"\n");tmp.replace(state_path)

    save()
    encoder=subprocess.Popen(["ffmpeg","-n","-v","error","-f","rawvideo","-pix_fmt","bgr24",
        "-s",f"{size*2}x{size+header}","-r","30","-i","-","-an","-c:v","libx264",
        "-threads","4","-preset","veryfast","-crf","18","-pix_fmt","yuv420p",
        "-movflags","+faststart",str(args.output)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    gallery=[];error=None
    try:
        for sequence in args.sequence:
            source=ROOT/"data/HUGG_ARIA_PINHOLE"/sequence
            metadata=json.loads((source/"source_metadata.json").read_text())
            subject=sequence.split('_')[0]
            if metadata.get('participant_id')!=subject or not metadata.get('have_hand_object_pose_gt'):
                raise ValueError("Subject/GT metadata mismatch")
            models=renderer.subject(subject)
            with (source/"frame_timestamps_214_1.csv").open() as f:
                timestamps=[int(r['timestamp_ns']) for r in csv.DictReader(f)]
            calibrations=[json.loads(x) for x in (source/"frame_pinhole_calibration_214_1.jsonl").read_text().splitlines()]
            if len(calibrations)!=len(timestamps):raise ValueError("Camera sidecar length mismatch")
            poses=sorted((json.loads(x) for x in (source/"mano_hand_pose_trajectory.jsonl").read_text().splitlines()),key=lambda x:x['timestamp_ns'])
            pose_times=[int(x['timestamp_ns']) for x in poses];pose_hands=[x['hand_poses'] for x in poses]
            headset=load_headset_pose_provider_from_csv(str(source/"headset_trajectory.csv"))
            capture=cv2.VideoCapture(str(source/"rgb_214_1_pinhole.mp4"))
            if not capture.isOpened() or round(capture.get(cv2.CAP_PROP_FRAME_COUNT))!=len(timestamps):
                raise RuntimeError("Source video frame count mismatch")
            segment={"subject":subject,"sequence":sequence,"output_start_frame":state['completed_frames'],
                     "frames":len(timestamps),"rendered_frames":0,"blank_frames":0,"hand_pose_frames":{"left":0,"right":0},
                     "checkpoint_sha256":{side:renderer.hashes[f'{subject}_{side}'] for side in models}}
            state['segments'].append(segment);state['current_sequence']=sequence;save()
            try:
                for frame,(ts,cal) in enumerate(zip(timestamps,calibrations)):
                    ok,original=capture.read()
                    if not ok:raise RuntimeError("Source video stopped before timestamp manifest")
                    hand_poses=nearest_hand_pose(pose_times,pose_hands,ts)
                    camera=world_to_camera(cal,headset,ts)
                    gaussian=np.zeros((size,size,3),np.uint8)
                    hands=[]
                    if hand_poses and camera is not None:
                        tensor=lambda x:torch.tensor(x,device='cuda',dtype=torch.float32)
                        for side,key in (("left","0"),("right","1")):
                            pose=hand_poses.get(key)
                            if pose is not None:
                                hands.append((models[side],tensor(pose['pose']),tensor(pose['wrist_xform']['q_wxyz']),tensor(pose['wrist_xform']['t_xyz'])))
                                segment['hand_pose_frames'][side]+=1
                        if hands:
                            r=render_scene(hands,tensor(camera),tensor(scaled_intrinsics(cal,size)),size=size)
                            if not torch.isfinite(r['rgb']).all() or not torch.isfinite(r['alpha']).all():
                                raise RuntimeError(f"Non-finite rendering at {sequence}:{frame}")
                            rgb=r['rgb'].clamp(0,1).cpu().numpy()
                            alpha=r['alpha'][...,0].cpu().numpy()
                            rgb[alpha<=0]=0
                            gaussian=(rgb[...,::-1]*255).round().astype(np.uint8)
                    if gaussian.any():segment['rendered_frames']+=1
                    else:segment['blank_frames']+=1
                    panel=np.zeros((size+header,size*2,3),np.uint8)
                    panel[header:,:size]=fit_panel(original,size)
                    panel[header:,size:]=gaussian
                    cv2.putText(panel,'Original video',(16,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(panel,'Gaussian rendering',(size+16,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(panel,f'{subject} | {sequence} | {frame/30:.1f}s / {len(timestamps)/30:.1f}s',
                                (16,52),cv2.FONT_HERSHEY_SIMPLEX,.55,(190,190,190),1,cv2.LINE_AA)
                    encoder.stdin.write(panel.tobytes())
                    state['completed_frames']+=1
                    if frame in (len(timestamps)//4,len(timestamps)//2,3*len(timestamps)//4):
                        thumb=cv2.resize(panel,(1024,round((size+header)*512/size)))
                        gallery.append(thumb)
                        cv2.imwrite(str(args.output.with_name(f'{args.output.stem}_{subject}_{frame:06d}.jpg')),thumb)
                    if state['completed_frames']%100==0:
                        save();print(json.dumps({k:state[k] for k in ('current_sequence','completed_frames','elapsed_seconds','frames_per_second','estimated_remaining_seconds')}),flush=True)
            finally:capture.release()
            save()
    except BaseException as exc:
        error=exc
    finally:
        try:encoder.stdin.close()
        except BrokenPipeError:pass
        stderr=encoder.stderr.read().decode();code=encoder.wait()
        if code and error is None:error=RuntimeError(stderr)
    if error is not None:
        state.update(status='failed',error=str(error));save();raise error
    try:
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=width,height,nb_read_frames,duration','-of','json',str(args.output)],text=True))['streams'][0]
        if int(probe['nb_read_frames'])!=total:raise RuntimeError("Final frame count mismatch")
        if gallery:cv2.imwrite(str(args.output.with_suffix('.jpg')),np.concatenate(gallery))
        state.update(status='complete',probe=probe,current_sequence=None);save()
    except BaseException as exc:
        state.update(status='failed',error=str(exc));save();raise
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()
