"""Find continuous clips with two independently usable Gaussian hand conditions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from hand_restoration.paired_video import absolute,GAUSSIAN_CONFIG,save_json,sha256
from hand_restoration.hugg_aria_conditioning import paste_crop_with_mask
from train_hand_restorer import make_hugg_aria_dataset
from temporal_demos.select_clips import fixed_box


def quality(sample):
    visible=sample['visible_hand_mask_np'].astype(bool)
    mano=sample['mano_mask_np'].astype(bool)
    inner=visible & ~sample['wrist_mask_np'].astype(bool)
    color=sample['mano_rgb_np'].max(-1)>2/255
    return dict(visible=float(visible.sum()/max(mano.sum(),1)),
        gaussian_coverage=float((color&inner).sum()/max(inner.sum(),1)),
        gaussian_psnr=float(-10*np.log10(max(float(((sample['condition_rgb_np']-sample['target_rgb_np'])[inner]**2).mean()),1e-10))) if inner.any() else 0,
        border=float((mano[:3].sum()+mano[-3:].sum()+mano[:,:3].sum()+mano[:,-3:].sum())/max(mano.sum(),1)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--all-sequences',action='store_true')
    args=parser.parse_args();out=absolute(args.output);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    config=json.loads(GAUSSIAN_CONFIG.read_text())
    previous=json.loads(absolute('outputs/hand_restoration/animatediff_demo_v2_30fps/selection.json').read_text())
    seqs={c['sequence'] for c in previous['selected']};groups=defaultdict(dict)
    for index,line in enumerate(absolute(config['data']['train_manifest']).open()):
        row=json.loads(line)
        if (args.all_sequences or row['sequence_id'] in seqs) and row['adaptive_handedness'] in ('left','right'):
            groups[(row['sequence_id'],row['adaptive_handedness'])][row['frame_index']]=dict(row,global_record_index=index)
    candidates=[]
    references={c['sequence']:c for c in previous['selected']}
    sequences=sorted({key[0] for key in groups})
    quality_prior={}
    for path in absolute('outputs/gaussian_hand/pretrain_v2').glob('*/audit/summary.json'):
        data=json.loads(path.read_text());side=data['binding']['handedness']
        for row in data['reports']:quality_prior[(row['sequence'],side)]=row['metric_percentiles']['visible_mano_ratio']['50']
    for seq in sequences:
        previous_clip=references.get(seq,dict(sequence=seq,subject=seq.split('_')[0],start=-10000,hand='',id=None))
        seq=previous_clip['sequence'];paired=set(groups[(seq,'left')])&set(groups[(seq,'right')])
        starts=set(sorted(paired)[::30]);starts.add(previous_clip['start'])
        options=[]
        for start in sorted(starts):
            if not all(f in paired for f in range(start,start+150)):continue
            hands={};scores=[]
            for side in ('left','right'):
                rows=[dict(groups[(seq,side)][f]) for f in range(start,start+150)]
                box=(previous_clip['fixed_crop_box'] if start==previous_clip['start'] and side==previous_clip['hand'] else fixed_box(rows))
                if min(box)<0 or max(box)>704:break
                area=np.array([r['adaptive_component_area'] for r in rows])
                sizes=np.array([r['adaptive_crop_box_xyxy'][2]-r['adaptive_crop_box_xyxy'][0] for r in rows])
                expansion=(box[2]-box[0])/np.median(sizes)
                if np.percentile(area,10)<600 or expansion>1.65:break
                for row in rows:
                    row['original_adaptive_crop_box_xyxy']=row['adaptive_crop_box_xyxy']
                    row['adaptive_crop_box_xyxy']=box
                hands[side]=dict(records=rows,fixed_box=box)
                scores.append(np.log(np.percentile(area,10))-3*(expansion-1)+4*quality_prior.get((seq,side),.7))
            if len(hands)!=2:continue
            a,b=hands['left']['fixed_box'],hands['right']['fixed_box']
            if min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1]):continue
            options.append(dict(id=f'{seq}_both_{start:06d}',sequence=seq,subject=previous_clip['subject'],
                start=start,hands=hands,coarse_score=float(min(scores)),matches_previous=start==previous_clip['start'],
                previous_id=previous_clip['id'],previous_side=previous_clip['hand']))
        chosen=sorted(options,key=lambda c:(c['matches_previous'],c['coarse_score']),reverse=True)[:1]
        for c in sorted(options,key=lambda c:c['coarse_score'],reverse=True):
            if all(abs(c['start']-x['start'])>=90 for x in chosen):chosen.append(c)
            if len(chosen)>=3:break
        candidates.extend(chosen)
    if args.all_sequences:
        shortlist=[]
        for subject in sorted({c['subject'] for c in candidates}):
            used=set()
            for clip in sorted([c for c in candidates if c['subject']==subject],key=lambda c:c['coarse_score'],reverse=True):
                if clip['sequence'] not in used:shortlist.append(clip);used.add(clip['sequence'])
                if len(used)==3:break
        # Retain the one two-hand window already found usable in the small scan.
        known=next((c for c in candidates if c['id']=='P0003_f4730606_both_001847'),None)
        if known and all(c['id']!=known['id'] for c in shortlist):shortlist.append(known)
        candidates=shortlist
    print('AUDIT_WINDOWS',len(candidates),flush=True)
    records=[];offsets=[0,30,60,90,120,149]
    for clip in candidates:
        clip['audit_offset']=len(records)
        for offset in offsets:
            for side in ('left','right'):records.append(clip['hands'][side]['records'][offset])
    manifest=out/'audit_frames.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    ds=make_hugg_aria_dataset(config,manifest,include_numpy=True)
    sheets=[]
    for clip in candidates:
        hand_quality={side:[] for side in ('left','right')};pictures=[]
        for j,offset in enumerate(offsets):
            samples=[ds[clip['audit_offset']+j*2+k] for k in range(2)]
            for side,s in zip(('left','right'),samples):hand_quality[side].append(quality(s))
            if j in (0,3,5):
                base=samples[0]['full_target_rgb_np'].copy()
                for s in samples:base,_=paste_crop_with_mask(base,s['condition_rgb_np'],s['edit_mask_np'],s['crop_box'])
                pictures.append(cv2.resize(cv2.rotate((base*255).round().clip(0,255).astype(np.uint8),cv2.ROTATE_90_CLOCKWISE),(384,384)))
        clip['audit']=hand_quality
        flat=[q for values in hand_quality.values() for q in values]
        clip['passes']=all(q['visible']>=.75 and q['gaussian_coverage']>=.85 and q['border']<.025 for q in flat)
        clip['score']=min(np.mean([q['gaussian_psnr'] for q in values])+10*np.percentile([q['visible'] for q in values],10) for values in hand_quality.values())
        canvas=np.concatenate(pictures,axis=1)
        title=np.zeros((32,canvas.shape[1],3),np.uint8)
        cv2.putText(title,clip['id']+f" | usable={clip['passes']}",(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        canvas=np.concatenate([title,canvas]);cv2.imwrite(str(out/(clip['id']+'.jpg')),canvas[...,::-1])
        sheets.append(cv2.resize(canvas,(864,312)))
        print(clip['id'],'PASS',clip['passes'],'score',round(clip['score'],2),
              {side:dict(visible_min=round(min(q['visible'] for q in values),3),coverage_min=round(min(q['gaussian_coverage'] for q in values),3)) for side,values in hand_quality.items()},flush=True)
        save_json(out/'candidates.json',candidates)
    selected=[]
    for clip in sorted(candidates,key=lambda x:x['score'],reverse=True):
        if clip['passes'] and clip['subject'] not in {c['subject'] for c in selected}:selected.append(clip)
        if len(selected)==4:break
    save_json(out/'selection.json',dict(config_sha256=sha256(GAUSSIAN_CONFIG),fps=30,seconds=5,
        selected=selected,candidates=candidates,source_manifest_sha256=sha256(absolute(config['data']['train_manifest']))))
    if selected:
        panels=[cv2.imread(str(out/(c['id']+'.jpg'))) for c in selected]
        cv2.imwrite(str(out/'shortlist.jpg'),np.concatenate(panels))
    print('SELECTED',[c['id'] for c in selected],flush=True)


if __name__=='__main__':main()
