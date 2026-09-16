"""Render selected clips with the original Gaussian ControlNet and frozen AnimateDiff."""
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

from hand_restoration.paired_video import ROOT,GAUSSIAN_CONFIG,absolute,save_json,sha256
from hand_restoration.hugg_aria_conditioning import paste_crop_with_mask
from train_hand_restorer import make_hugg_aria_dataset
from temporal_demos.motion import TemporalRestorer,ADAPTER


def video(path,frames,fps):
    first=frames[0]
    encoder=subprocess.Popen(['ffmpeg','-n','-v','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{first.shape[1]}x{first.shape[0]}','-r',str(fps),'-i','-','-an','-c:v','libx264',
        '-threads','2','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],
        stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for frame in frames:encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
    encoder.stdin.close();error=encoder.stderr.read().decode()
    if encoder.wait():raise RuntimeError(error)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,duration','-of','json',str(path)],text=True))['streams'][0]
    assert int(info['nb_read_frames'])==len(frames)
    assert abs(float(info['duration'])-len(frames)/fps)<.002


def metrics(outputs,targets,masks):
    mae=[];psnr=[];temporal=[]
    for i,(image,target,mask) in enumerate(zip(outputs,targets,masks)):
        diff=(image.astype(np.float32)-target.astype(np.float32))/255
        if mask.any():
            mae.append(float(np.abs(diff[mask]).mean()*255))
            psnr.append(float(-10*np.log10(max(float((diff[mask]**2).mean()),1e-12))))
        if i:
            now=cv2.cvtColor(target,cv2.COLOR_RGB2GRAY)
            before=cv2.cvtColor(targets[i-1],cv2.COLOR_RGB2GRAY)
            flow=cv2.calcOpticalFlowFarneback(now,before,None,.5,3,15,3,5,1.2,0)
            yy,xx=np.mgrid[:now.shape[0],:now.shape[1]].astype(np.float32)
            maps=(xx+flow[...,0],yy+flow[...,1])
            warp=lambda x:cv2.remap(x.astype(np.float32),*maps,cv2.INTER_LINEAR)
            valid=mask & (warp(masks[i-1])>.99)
            valid=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            residual=(image.astype(np.float32)-warp(outputs[i-1]))-(target.astype(np.float32)-warp(targets[i-1]))
            if valid.any():temporal.append(float(np.abs(residual[valid]).mean()))
    return dict(visible_hand_psnr=float(np.mean(psnr)),visible_hand_mae_255=float(np.mean(mae)),
        flow_compensated_temporal_residual_mae_255=float(np.mean(temporal)) if temporal else None,
        note='Optical-flow temporal proxy; not a perceptual-quality guarantee. Selected training clips, not benchmark.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--count',type=int,default=6)
    parser.add_argument('--pilot-frames',type=int,default=0)
    parser.add_argument('--strength',type=float,default=.25)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--scheduler',choices=['original','ddim','ddim-linear'],default='original')
    parser.add_argument('--refine-strength',type=float,default=0.)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise ValueError('Set CUDA_VISIBLE_DEVICES to --gpu')
    lock=open(f'/tmp/glove_tracking_paired_gpu{args.gpu}_{os.getuid()}.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    query=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'],text=True)
    memory,util=map(int,query.strip().split(','))
    if memory>1024 or util>10:raise RuntimeError('Selected GPU is not idle')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(.35)
    selection=json.loads(absolute(args.selection).read_text())
    config=json.loads(GAUSSIAN_CONFIG.read_text())
    if sha256(GAUSSIAN_CONFIG)!=selection['config_sha256']:raise ValueError('Configuration changed')
    checkpoint=absolute(config['training']['output_dir'])/'controlnet_final.pt'
    output=absolute(args.output);output.mkdir(parents=True,exist_ok=False)
    clips=selection['selected'][:args.count]
    records=[]
    for c in clips:
        c['offset']=len(records)
        c['render_records']=c['records'][:args.pilot_frames] if args.pilot_frames else c['records']
        records.extend(c['render_records'])
    manifest=output/'frames.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    dataset=make_hugg_aria_dataset(config,manifest,include_numpy=True)
    started=time.monotonic()
    model=TemporalRestorer(config,checkpoint)
    from huggingface_hub import hf_hub_download
    adapter_path=Path(hf_hub_download(ADAPTER,'diffusion_pytorch_model.safetensors',local_files_only=True)).resolve()
    save_json(output/'provenance.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        selection_sha256=sha256(absolute(args.selection)),adapter=ADAPTER,checkpoint_step=model.checkpoint_step,
        adapter_weights=str(adapter_path),adapter_sha256=sha256(adapter_path),motion_layers=model.motion_layers,
        gpu=args.gpu,steps=args.steps,motion_strength=args.strength,fps=selection['fps'],seed=7,
        dtype='bfloat16',scheduler=dict(model.model.noise_scheduler.config),context=16,overlap=8,
        animatediff_scheduler=args.scheduler,refine_strength=args.refine_strength,
        extra_training=False,initial_noise='shared across frames; identical random draws in addon off/on',
        code={p:sha256(ROOT/p) for p in ('temporal_demos/motion.py','temporal_demos/render_clips.py')},
        scope='Inference-only demo; new fixed temporal crop; not frozen baseline benchmark. Native orientation inference.'))
    reports=[];sheets=[];combined=[]
    for clip in clips:
        directory=output/clip['id'];directory.mkdir()
        conditions=[];masks=[];targets=[];visible=[];fractions=[]
        for j in range(len(clip['render_records'])):
            s=dataset[clip['offset']+j]
            conditions.append(s['condition_rgb_np']);masks.append(s['condition_mask_np'])
            targets.append((s['target_rgb_np']*255).round().astype(np.uint8))
            visible.append(s['visible_hand_mask_np'].astype(bool))
            fractions.append(float(visible[-1].sum()/max(s['mano_mask_np'].sum(),1)))
        if min(fractions)<.70 or np.percentile(fractions,10)<.80:
            save_json(directory/'SKIPPED.json',dict(reason='Dense mask screening failed',visible_ratios=fractions))
            print('SKIPPED',clip['id'],min(fractions),flush=True);continue
        encoded=model.encode(conditions,masks)
        generated={}
        for name,strength in [('gaussian_only',0.),('animatediff',args.strength)]:
            began=time.monotonic()
            generated[name]=model.generate(encoded,strength=strength,steps=args.steps,seed=7,
                scheduler_kind=args.scheduler if name=='animatediff' else 'original',
                initial_rgb=generated.get('gaussian_only') if name=='animatediff' and args.refine_strength else None,
                denoise_strength=args.refine_strength,
                progress=lambda step,total: print(clip['id'],name,step,total,flush=True) if step%5==0 else None)
            print('TIMING',name,time.monotonic()-began,flush=True)
            video(directory/f'{name}_crop.mp4',[cv2.rotate(f,cv2.ROTATE_90_CLOCKWISE) for f in generated[name]],selection['fps'])
        panels=[];full=[]
        for j,(single,animated) in enumerate(zip(generated['gaussian_only'],generated['animatediff'])):
            images=[targets[j],(conditions[j]*255).round().astype(np.uint8),single,animated]
            titles=['Original','Gaussian condition','Gaussian diffusion only','+ AnimateDiff']
            canvas=np.zeros((576,2048,3),np.uint8)
            for k,(im,title) in enumerate(zip(images,titles)):
                canvas[64:,k*512:(k+1)*512]=cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE)
                cv2.putText(canvas,title,(k*512+10,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(canvas,f"{clip['id']} | source {clip['render_records'][j]['frame_index']} | {selection['fps']} fps | motion {args.strength}",
                        (10,51),cv2.FONT_HERSHEY_SIMPLEX,.48,(210,210,210),1,cv2.LINE_AA)
            panels.append(canvas)
            s=dataset[clip['offset']+j]
            restored,_=paste_crop_with_mask(s['full_condition_rgb_np'],animated.astype(np.float32)/255,
                                           s['edit_mask_np'],s['crop_box'])
            full.append(cv2.rotate((restored*255).round().clip(0,255).astype(np.uint8),cv2.ROTATE_90_CLOCKWISE))
            for name in generated:
                cv2.imwrite(str(directory/f'{name}_{j:04d}.png'),generated[name][j][...,::-1])
        video(directory/'comparison.mp4',panels,selection['fps'])
        video(directory/'animatediff_full.mp4',full,selection['fps'])
        sheet=np.concatenate([panels[k] for k in (0,len(panels)//2,len(panels)-1)])
        cv2.imwrite(str(directory/'contact_sheet.jpg'),sheet[...,::-1])
        sheets.append(cv2.resize(panels[len(panels)//2],(1024,288)))
        combined.extend(cv2.resize(p,(1536,432)) for p in panels)
        report=dict(id=clip['id'],sequence=clip['sequence'],subject=clip['subject'],hand=clip['hand'],
            source_start=clip['start'],frames=len(panels),duration=len(panels)/selection['fps'],
            dense_visible_min=min(fractions),dense_visible_p10=float(np.percentile(fractions,10)),
            metrics={name:metrics(frames,targets,visible) for name,frames in generated.items()})
        save_json(directory/'report.json',report);reports.append(report)
        print('COMPLETE_CLIP',json.dumps(report),flush=True)
        del encoded,generated,panels,full
    peak=torch.cuda.max_memory_allocated();del model;torch.cuda.empty_cache()
    if combined:video(output/'review_all.mp4',combined,selection['fps'])
    if sheets:cv2.imwrite(str(output/'contact_sheet.jpg'),np.concatenate(sheets)[...,::-1])
    save_json(output/'summary.json',dict(clips=reports,elapsed_seconds=time.monotonic()-started,
        peak_gpu_bytes=peak,extra_training=False))
    print('COMPLETE',output,'elapsed',time.monotonic()-started,'peakGiB',peak/2**30,flush=True)


if __name__=='__main__':main()
