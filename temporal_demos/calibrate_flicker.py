"""Small paired native-frame tests before changing temporal-demo presets."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from hand_restoration.paired_video import absolute,GAUSSIAN_CONFIG,save_json,sha256
from train_hand_restorer import make_hugg_aria_dataset
from temporal_demos.motion import TemporalRestorer
from temporal_demos.render_clips import metrics,video


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--shared',action='store_true')
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.gpu)
    lock=open(f'/tmp/glove_tracking_paired_gpu{args.gpu}_{os.getuid()}.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    values=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    memory,util=map(int,values.strip().split(','))
    if memory>1024 or util>10:raise RuntimeError('GPU busy')
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.cuda.set_per_process_memory_fraction(.3)
    selection=json.loads(absolute(args.selection).read_text());config=json.loads(GAUSSIAN_CONFIG.read_text())
    chosen=[selection['selected'][0],selection['selected'][4]]
    records=[r for c in chosen for r in c['records'][:45]]
    out=absolute(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest=out/'frames.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    dataset=make_hugg_aria_dataset(config,manifest,include_numpy=True)
    checkpoint=absolute(config['training']['output_dir'])/'controlnet_final.pt'
    start=time.monotonic();model=TemporalRestorer(config,checkpoint)
    variants=[('ddpm_025','original',.25,8),('ddim_025','ddim-stable',.25,8),
              ('ddim_035','ddim-stable',.35,12),('ddim_045','ddim-stable',.45,12)]
    if args.shared:
        variants=[('ddpm_025','original',.25,8),('shared_025','ddpm-shared',.25,8),
                  ('shared_030','ddpm-shared',.30,12),('shared_035','ddpm-shared',.35,12)]
    reports=[]
    for ci,clip in enumerate(chosen):
        conditions=[];masks=[];targets=[];visible=[]
        for j in range(45):
            s=dataset[ci*45+j]
            conditions.append(s['condition_rgb_np']);masks.append(s['condition_mask_np'])
            targets.append((s['target_rgb_np']*255).round().astype(np.uint8))
            visible.append(s['visible_hand_mask_np'].astype(bool))
        encoded=model.encode(conditions,masks);outputs={};result={}
        for label,scheduler,strength,overlap in variants:
            began=time.monotonic()
            frames=model.generate(encoded,strength=strength,scheduler_kind=scheduler,overlap=overlap,steps=20,seed=7)
            outputs[label]=frames;result[label]=metrics(frames,targets,visible)
            print(clip['id'],label,round(time.monotonic()-began,1),result[label],flush=True)
        result['input']=metrics([(c*255).round().astype(np.uint8) for c in conditions],targets,visible)
        panels=[]
        for j in range(45):
            images=[targets[j],*[outputs[v[0]][j] for v in variants]]
            canvas=np.zeros((576,2560,3),np.uint8)
            for col,(im,label) in enumerate(zip(images,['Original',*[v[0] for v in variants]])):
                canvas[64:,col*512:(col+1)*512]=cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE)
                cv2.putText(canvas,label,(col*512+10,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            panels.append(canvas)
        video(out/f"{clip['id']}.mp4",panels,30)
        cv2.imwrite(str(out/f"{clip['id']}.jpg"),np.concatenate([panels[0],panels[22],panels[44]])[...,::-1])
        reports.append(dict(id=clip['id'],metrics=result))
        save_json(out/'partial.json',reports)
    save_json(out/'summary.json',dict(clips=reports,variants=variants,frames_per_clip=45,fps=30,
        checkpoint_sha256=sha256(checkpoint),elapsed_seconds=time.monotonic()-start,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),extra_training=False))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
