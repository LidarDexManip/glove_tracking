"""Both-hand restoration, composited into one frame and exported as a clean demo."""
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
from hand_restoration.hugg_aria_conditioning import paste_crop_with_mask
from train_hand_restorer import make_hugg_aria_dataset
from temporal_demos.motion import TemporalRestorer
from temporal_demos.stabilize import stabilize_brightness
from temporal_demos.select_two_hands import quality
from temporal_demos.render_clips import video


def composite_pair(base,conditions,restorations,masks,boxes):
    if not all(len(x)==2 for x in (conditions,restorations,masks,boxes)):
        raise ValueError('Exactly two hands are required')
    conditioned=base.copy();restored=base.copy();full_masks=[]
    for condition,restoration,mask,box in zip(conditions,restorations,masks,boxes):
        conditioned,full_mask=paste_crop_with_mask(conditioned,condition,mask,box)
        restored,_=paste_crop_with_mask(restored,restoration,mask,box)
        if not full_mask.any():raise ValueError('Missing hand edit mask')
        full_masks.append(full_mask)
    if (full_masks[0]&full_masks[1]).any():raise ValueError('Hand edit regions overlap')
    outside=~(full_masks[0]|full_masks[1])
    if not np.array_equal(conditioned[outside],base[outside]) or not np.array_equal(restored[outside],base[outside]):
        raise AssertionError('Background changed outside both hands')
    return conditioned,restored,full_masks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=6)
    parser.add_argument('--allow-shared-gpu',action='store_true')
    parser.add_argument('--clip',action='append',help='Render only these visually approved candidate IDs')
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise ValueError('CUDA_VISIBLE_DEVICES must match --gpu')
    lock=open(f'/tmp/glove_tracking_paired_gpu{args.gpu}_{os.getuid()}.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    used,free,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
        '--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
    if free<20*1024:raise RuntimeError('Need 16 GiB allocator cap plus 4 GiB headroom')
    if not args.allow_shared_gpu and (used>1024 or util>10):raise RuntimeError('GPU is busy')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(0).total_memory)
    selection=json.loads(absolute(args.selection).read_text())
    if selection['fps']!=30 or sha256(GAUSSIAN_CONFIG)!=selection['config_sha256']:raise ValueError('Input settings changed')
    config=json.loads(GAUSSIAN_CONFIG.read_text());out=absolute(args.output);out.mkdir(parents=True,exist_ok=False)
    metadata=out/'metadata';metadata.mkdir();clips_dir=out/'clips';clips_dir.mkdir()
    queue=selection['selected']+[c for c in selection['candidates'] if c['passes'] and
        c['id'] not in {s['id'] for s in selection['selected']}]
    if args.clip:
        by_id={c['id']:c for c in queue}
        if not set(args.clip)<=by_id.keys():raise ValueError('Requested clip did not pass screening')
        queue=[by_id[name] for name in dict.fromkeys(args.clip)]
    records=[]
    for clip in queue:
        clip['offsets']={}
        for side in ('left','right'):
            clip['offsets'][side]=len(records);records.extend(clip['hands'][side]['records'])
    if not records:raise ValueError('No two-hand clips passed screening')
    manifest=metadata/'frames.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    dataset=make_hugg_aria_dataset(config,manifest,include_numpy=True)
    checkpoint=absolute(config['training']['output_dir'])/'controlnet_final.pt'
    save_json(metadata/'provenance.json',dict(config_sha256=sha256(GAUSSIAN_CONFIG),checkpoint_sha256=sha256(checkpoint),
        selection_sha256=sha256(absolute(args.selection)),gpu=args.gpu,shared_gpu=args.allow_shared_gpu,
        allocator_limit_gib=16,motion_strength=.35,context=24,overlap=16,steps=20,seed=7,fps=30,
        brightness_window=9,brightness_strength=.9,labels=['Gaussian-conditioned input','Restoration'],
        method='Independent left/right temporal crops; both pasted onto the same original frame; outside both edit masks unchanged.',
        extra_training=False,code={p:sha256(absolute(p)) for p in ('temporal_demos/render_two_hands.py','temporal_demos/motion.py','temporal_demos/stabilize.py')}))
    start=time.monotonic();model=None;reports=[];montage=[]
    completed_subjects=set()
    for clip in queue:
        if clip['subject'] in completed_subjects:continue
        hands={};failed=False
        for side in ('left','right'):
            values=dict(conditions=[],masks=[],visible=[],edits=[],quality=[],boxes=[])
            for j in range(150):
                sample=dataset[clip['offsets'][side]+j]
                for key,sample_key in [('conditions','condition_rgb_np'),('masks','condition_mask_np'),
                                       ('visible','visible_hand_mask_np'),('edits','edit_mask_np')]:values[key].append(sample[sample_key])
                values['boxes'].append(sample['crop_box']);values['quality'].append(quality(sample))
            visible=[q['visible'] for q in values['quality']];coverage=[q['gaussian_coverage'] for q in values['quality']]
            passed=min(visible)>=.70 and np.percentile(visible,10)>=.80 and min(coverage)>=.85
            if not passed:failed=True
            print('DENSE',clip['id'],side,'passed',passed,'visibility min/p10',min(visible),np.percentile(visible,10),'coverage',min(coverage),flush=True)
            hands[side]=values
        if failed:
            save_json(metadata/(clip['id']+'_skipped.json'),{side:hands[side]['quality'] for side in hands})
            continue
        for side in ('left','right'):
            values=hands[side]
            if model is None:model=TemporalRestorer(config,checkpoint)
            encoded=model.encode(values['conditions'],values['masks'])
            raw=model.generate(encoded,strength=.35,steps=20,seed=7,context_size=24,overlap=16,
                scheduler_kind='original',progress=lambda step,total:print(clip['id'],side,step,total,flush=True) if step%5==0 else None)
            values['generated'],correction=stabilize_brightness(raw,values['conditions'],values['visible'],values['edits'],window=9,strength=.9)
            save_json(metadata/(clip['id']+'_'+side+'_brightness.json'),correction)
            del encoded,raw
        panels=[];checks=[]
        for j in range(150):
            sample=dataset[clip['offsets']['left']+j]
            base=sample['full_target_rgb_np']
            condition,restored,masks=composite_pair(base,[hands[s]['conditions'][j] for s in ('left','right')],
                [hands[s]['generated'][j].astype(np.float32)/255 for s in ('left','right')],
                [hands[s]['edits'][j] for s in ('left','right')],[hands[s]['boxes'][j] for s in ('left','right')])
            changes=[int((np.abs(restored-base).max(-1)[mask]>1/255).sum()) for mask in masks]
            if min(changes)<100:raise AssertionError('One hand was not restored')
            checks.append(changes)
            panel=np.zeros((1024,1920,3),np.uint8)
            for k,(rgb,label) in enumerate(zip((condition,restored),('Gaussian-conditioned input','Restoration'))):
                im=cv2.rotate((rgb*255).round().clip(0,255).astype(np.uint8),cv2.ROTATE_90_CLOCKWISE)
                panel[64:,k*960:(k+1)*960]=cv2.resize(im,(960,960),interpolation=cv2.INTER_AREA)
                size=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,1.05,2)[0]
                cv2.putText(panel,label,(k*960+(960-size[0])//2,42),cv2.FONT_HERSHEY_SIMPLEX,1.05,(255,255,255),2,cv2.LINE_AA)
            panels.append(panel)
        destination=clips_dir/(clip['id']+'.mp4');video(destination,panels,30)
        sheet=np.concatenate([cv2.resize(panels[j],(960,512)) for j in (0,75,149)])
        cv2.imwrite(str(metadata/(clip['id']+'.jpg')),sheet[...,::-1]);montage.append(cv2.resize(panels[75],(960,512)))
        reports.append(dict(id=clip['id'],frames=150,source_start=clip['start'],video=str(destination),
            minimum_changed_pixels_per_hand=np.min(checks,axis=0).tolist(),
            dense_audit={s:hands[s]['quality'] for s in ('left','right')}))
        completed_subjects.add(clip['subject'])
        save_json(metadata/'completed.json',reports)
        print('COMPLETED',clip['id'],'both hands changed',np.min(checks,axis=0).tolist(),flush=True)
        del hands,panels,sample,base,condition,restored
    peak=torch.cuda.max_memory_allocated();del model;torch.cuda.empty_cache()
    if reports:
        listing=metadata/'concat.txt';listing.write_text(''.join(f"file '{r['video']}'\n" for r in reports))
        subprocess.run(['ffmpeg','-n','-v','error','-f','concat','-safe','0','-i',str(listing),'-map','0:v:0',
            '-an','-c:v','copy','-movflags','+faststart',str(out/'two_hand_demo_30fps.mp4')],check=True)
        cv2.imwrite(str(metadata/'contact_sheet.jpg'),np.concatenate(montage)[...,::-1])
    save_json(metadata/'summary.json',dict(clips=reports,elapsed_seconds=time.monotonic()-start,peak_gpu_bytes=peak,
        frames=len(reports)*150,fps=30,extra_training=False))
    print('COMPLETE',len(reports),'two-hand clips',time.monotonic()-start,'seconds',flush=True)


if __name__=='__main__':main()
