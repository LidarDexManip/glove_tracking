"""CPU-only screening of stable, well-visible Gaussian-condition demo windows."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from hand_restoration.paired_video import ROOT, GAUSSIAN_CONFIG, absolute, save_json, sha256
from train_hand_restorer import make_hugg_aria_dataset


def fixed_box(rows):
    boxes=np.array([r['adaptive_crop_box_xyxy'] for r in rows])
    low=boxes[:,:2].min(0); high=boxes[:,2:].max(0)
    side=int(np.ceil(max(high-low)))
    center=(low+high)/2
    start=np.floor(center-side/2).astype(int)
    return [*start.tolist(),*(start+side).tolist()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=5)
    parser.add_argument('--fps',type=int,default=10)
    parser.add_argument('--count',type=int,default=6)
    args=parser.parse_args()
    assert 30 % args.fps == 0 and args.seconds >= 2
    cv2.setNumThreads(1)
    output=absolute(args.output); output.mkdir(parents=True,exist_ok=False)
    config=json.loads(GAUSSIAN_CONFIG.read_text())
    manifest=absolute(config['data']['train_manifest'])
    groups=defaultdict(dict)
    for index,line in enumerate(manifest.open()):
        r=json.loads(line)
        r['global_record_index']=index
        groups[(r['sequence_id'],r['adaptive_handedness'])][r['frame_index']]=r
    quality={}
    for path in (ROOT/'outputs/gaussian_hand/pretrain_v2').glob('*/audit/summary.json'):
        data=json.loads(path.read_text())
        side=data['binding']['handedness']
        for r in data['reports']:
            quality[(r['sequence'],side)]=r['metric_percentiles']['visible_mano_ratio']['50']
    # Existing Gaussian diffusion PSNR is only a weak presentation-screening proxy.
    # This intentionally selected showcase batch is not a model-quality benchmark.
    prev=ROOT/'outputs/hand_restoration/paired_quantitative_final_seen_eval_v1/gaussian/frames.jsonl'
    restoration={}
    if prev.exists():
        for line in prev.open():
            r=json.loads(line); m=r['regions']['visible_hand']
            if m: restoration[(r['sequence_id'],r['hand'])]=m['psnr_db']
    candidates=[]
    length=args.seconds*30
    for (seq,hand),mapping in sorted(groups.items()):
        if hand not in ('left','right'):
            continue  # Avoid switching joint/split crop identity within a clip.
        frames=sorted(mapping)
        options=[]
        for start in frames[::30]:
            if not all(f in mapping for f in range(start,start+length)):
                continue
            rows=[mapping[f] for f in range(start,start+length)]
            box=fixed_box(rows)
            # Current aligned MANO projection coordinates are 704x704.
            if min(box)<0 or max(box)>704:
                continue
            boxes=np.array([r['adaptive_crop_box_xyxy'] for r in rows])
            sizes=boxes[:,2]-boxes[:,0]
            zoom=(box[2]-box[0])/np.median(sizes)
            area=np.array([r['adaptive_component_area'] for r in rows])
            if zoom>1.45 or np.percentile(area,10)<800:
                continue
            score=(10*quality.get((seq,hand),.7)+.3*restoration.get((seq,hand),18)
                   -5*(zoom-1)-2*np.std(sizes)/np.mean(sizes)+np.log(np.median(area))/3)
            options.append(dict(sequence=seq,subject=seq.split('_')[0],hand=hand,start=start,
                fixed_crop_box=box,coarse_score=float(score),crop_expansion=float(zoom),
                coarse_visible_median=quality.get((seq,hand)),
                existing_seen_psnr=restoration.get((seq,hand))))
        if options:
            candidates.append(max(options,key=lambda x:x['coarse_score']))
    # Limit expensive pixel inspection, while retaining subject diversity.
    coarse=[]
    for subject in sorted({r['subject'] for r in candidates}):
        coarse.extend(sorted([r for r in candidates if r['subject']==subject],
                             key=lambda r:r['coarse_score'],reverse=True)[:4])
    print('coarse candidates',len(candidates),'pixel-audit windows',len(coarse),flush=True)
    all_rows=[]
    for i,clip in enumerate(coarse):
        clip['id']=f"{clip['sequence']}_{clip['hand']}_{clip['start']:06d}"
        clip['records']=[]
        for frame in range(clip['start'],clip['start']+length,30//args.fps):
            row=dict(groups[(clip['sequence'],clip['hand'])][frame])
            row['original_adaptive_crop_box_xyxy']=row['adaptive_crop_box_xyxy']
            row['adaptive_crop_box_xyxy']=clip['fixed_crop_box']
            clip['records'].append(row)
        clip['dataset_offset']=len(all_rows)
        all_rows.extend(clip['records'])
    path=output/'candidate_frames.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
    dataset=make_hugg_aria_dataset(config,path,include_numpy=True)
    gallery=[]
    for clip in coarse:
        visible,appearance,ious,border,sharp=[],[],[],[],[]
        last=None; panel=None
        inspected=np.linspace(0,len(clip['records'])-1,10,dtype=int).tolist()
        for j in inspected:
            s=dataset[clip['dataset_offset']+j]
            mask=s['visible_hand_mask_np'].astype(bool)
            mano=s['mano_mask_np'].astype(bool)
            visible.append(float(mask.sum()/max(mano.sum(),1)))
            inner=mask & ~s['wrist_mask_np'].astype(bool)
            err=(s['condition_rgb_np']-s['target_rgb_np'])**2
            appearance.append(float(-10*np.log10(max(float(err[inner].mean()),1e-10))) if inner.any() else 0)
            border.append(float((mano[:3].sum()+mano[-3:].sum()+mano[:,:3].sum()+mano[:,-3:].sum())/max(mano.sum(),1)))
            if last is not None:
                ious.append(float((mask&last).sum()/max((mask|last).sum(),1)))
            last=mask
            gray=cv2.cvtColor((s['target_rgb_np']*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
            lap=cv2.Laplacian(gray,cv2.CV_32F)
            sharp.append(float(lap[mask].var()) if mask.any() else 0)
            if j==inspected[len(inspected)//2]:
                pictures=[s['target_rgb_np'],s['condition_rgb_np'],np.repeat(mask[...,None],3,axis=2).astype(np.float32)]
                panel=np.concatenate([cv2.rotate((x[...,::-1]*255).round().astype(np.uint8),cv2.ROTATE_90_CLOCKWISE) for x in pictures],axis=1)
                panel=cv2.resize(panel,(768,256))
        metrics=dict(visible_min=min(visible),visible_p10=float(np.percentile(visible,10)),
            gaussian_psnr_mean=float(np.mean(appearance)),gaussian_psnr_min=min(appearance),
            mask_temporal_iou_median=float(np.median(ious)),border_fraction_max=max(border),
            sharpness_median=float(np.median(sharp)))
        clip['pixel_audit']=metrics
        clip['audit_sample_offsets']=inspected
        clip['score']=(metrics['gaussian_psnr_mean']+12*metrics['visible_p10']
                       +3*metrics['mask_temporal_iou_median']-20*metrics['border_fraction_max'])
        clip['passes_proxy_gate']=(metrics['visible_p10']>=.85 and metrics['visible_min']>=.75
             and metrics['border_fraction_max']<.02 and metrics['gaussian_psnr_mean']>=17)
        if panel is not None:
            title=np.zeros((32,768,3),np.uint8)
            cv2.putText(title,f"{clip['id']} visible={metrics['visible_p10']:.2f} Gaussian PSNR={metrics['gaussian_psnr_mean']:.1f}",
                        (5,21),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
            cv2.imwrite(str(output/f"{clip['id']}_audit.jpg"),np.concatenate([title,panel]))
        print(clip['id'],clip['passes_proxy_gate'],metrics,flush=True)
    del dataset
    ranked=sorted(coarse,key=lambda r:r['score'],reverse=True)
    selected=[]; subjects=set()
    for clip in ranked:
        if clip['passes_proxy_gate'] and clip['subject'] not in subjects:
            selected.append(clip); subjects.add(clip['subject'])
        if len(selected)==args.count: break
    if len(selected)<args.count:
        for clip in ranked:
            if clip['passes_proxy_gate'] and clip not in selected:
                selected.append(clip)
            if len(selected)==args.count: break
    report=dict(config=str(GAUSSIAN_CONFIG),config_sha256=sha256(GAUSSIAN_CONFIG),
        source_manifest_sha256=sha256(manifest),source_fps=30,fps=args.fps,seconds=args.seconds,
        coarse_candidate_count=len(candidates),audited_window_count=len(coarse),selected=selected,candidates=ranked,
        caveat='Presentation-only selected clips; mask/appearance proxies, not manually verified segmentation ground truth.',
        temporal_crop='Fixed union square over all 150 consecutive source frames. Same hand identity. No original manifests modified.')
    save_json(output/'selection.json',report)
    selected_panels=[cv2.imread(str(output/f"{c['id']}_audit.jpg")) for c in selected]
    if selected_panels: cv2.imwrite(str(output/'shortlist.jpg'),np.concatenate(selected_panels))
    print('SELECTED',[r['id'] for r in selected],flush=True)


if __name__=='__main__':main()
