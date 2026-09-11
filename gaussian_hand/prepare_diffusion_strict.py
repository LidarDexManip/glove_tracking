"""Prepare the controlled Gaussian input experiment; NEVER launch training."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/gaussian_hand/diffusion_strict_v2'
BASELINE=ROOT/'configs/hand_restoration/hugg_aria_glove2hand_faithful_mano_adaptive_splitcrop_v1.json'
CONFIG=ROOT/'configs/hand_restoration/hugg_aria_gaussian_surface_strict_v2.json'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def setup():
    freeze=json.loads((ROOT/'baselines/adaptive_splitcrop_v1/BASELINE_FREEZE.json').read_text())
    config=json.loads(BASELINE.read_text())
    assert digest(BASELINE)==freeze['configuration']['sha256']
    for key in ('train_manifest','val_manifest','manifest_summary'):
        expected={'train_manifest':'train_manifest_sha256','val_manifest':'seen_eval_manifest_sha256',
                  'manifest_summary':'manifest_summary_sha256'}[key]
        if digest(ROOT/config['data'][key])!=freeze['data'][expected]:raise ValueError(f'Baseline {key} changed')
    output='outputs/hand_restoration/hugg_aria_gaussian_surface_strict_v2'
    config['data'].update(render_kind='gaussian',render_root='outputs/gaussian_hand/diffusion_strict_v2/aligned',
                          render_rgb_semantics='premultiplied',render_storage='aligned_video')
    config['training']['output_dir']=output
    config['comparison']={'baseline_id':freeze['baseline_id'],'expected_world_size':8,
        'effective_batch_size':256,'initialization':'same SD1.5 initialization as baseline; no old ControlNet warm start',
        'required_readiness':'outputs/gaussian_hand/diffusion_strict_v2/READY.json',
        'treatment':'Gaussian RGB and native opacity within frozen baseline MANO support; all masks/crops fixed'}
    OUT.mkdir(parents=True,exist_ok=True)
    CONFIG.write_text(json.dumps(config,indent=2)+'\n')
    grouped=defaultdict(list)
    with (ROOT/config['data']['train_manifest']).open() as f:
        for i,line in enumerate(f):
            r=json.loads(line);r['global_record_index']=i;grouped[r['sequence_id']].append(r)
    (OUT/'manifests').mkdir(exist_ok=True)
    for sequence,rows in grouped.items():
        (OUT/'manifests'/f'{sequence}.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    if len(grouped)!=136 or sum(map(len,grouped.values()))!=524917:raise ValueError('Unexpected sample population')
    meta={'sequences':len(grouped),'train_records':sum(map(len,grouped.values())),
          'config_sha256':digest(CONFIG),'baseline_config_sha256':digest(BASELINE),
          'baseline_freeze':freeze,'training_started':False}
    (OUT/'setup.json').write_text(json.dumps(meta,indent=2)+'\n')
    return config,sorted(grouped)


def finalize(config,sequences):
    freeze=json.loads((OUT/'setup.json').read_text())['baseline_freeze']
    qa=json.loads((OUT/'aligned_qa/report.json').read_text())
    if qa['status']!='passed' or qa['records']!=183:raise ValueError('Final encoded-data QA not passed')
    rows=[]
    for seq in sequences:
        render=OUT/'aligned'/seq/'_SUCCESS.json';images=OUT/'image_sequences'/seq/'_SUCCESS.json'
        r=json.loads(render.read_text());i=json.loads(images.read_text())
        if r['support_policy']!='frozen_baseline_mano_alpha' or i['outside_edit_max_delta']!=0:
            raise ValueError(f'Failed sequence contract: {seq}')
        rows.append({'sequence':seq,'render_success_sha256':digest(render),'images_success_sha256':digest(images),
                     'records':i['records'],'rendered_frames':r['rendered_frames'],'archive_bytes':i['archive_bytes']})
    if sum(r['records'] for r in rows)!=524917 or sum(r['rendered_frames'] for r in rows)!=393684:
        raise ValueError('Incomplete frame/crop coverage')
    checkpoint=ROOT/freeze['training_result']['checkpoint']
    if digest(checkpoint)!=freeze['training_result']['checkpoint_sha256']:raise ValueError('Baseline checkpoint changed')
    manifest_hashes={key:digest(ROOT/config['data'][key]) for key in ('train_manifest','val_manifest','manifest_summary')}
    expected={'train_manifest':freeze['data']['train_manifest_sha256'],'val_manifest':freeze['data']['seen_eval_manifest_sha256'],
              'manifest_summary':freeze['data']['manifest_summary_sha256']}
    if manifest_hashes!=expected:raise ValueError('Baseline manifests changed during preparation')
    result={'status':'ready','training_started':False,'config':str(CONFIG),'config_sha256':digest(CONFIG),
            'baseline_checkpoint_sha256':digest(checkpoint),'manifest_sha256':manifest_hashes,
            'world_size':8,'batch_per_gpu':32,'gradient_accumulation':1,'effective_batch':256,'optimizer_steps':15380,
            'sequence_count':len(rows),'train_records':524917,'seen_eval_records':183,'numeric_eval_subset':136,
            'condition_images':524917,'overlay_images':524917,'image_archive_bytes':sum(r['archive_bytes'] for r in rows),
            'rows':rows,'inference_only_and_input_preparation':True}
    (OUT/'READY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setup-only',action='store_true')
    p.add_argument('--finalize-only',action='store_true')
    p.add_argument('--render-workers',type=int,default=2)
    p.add_argument('--render-gpus',default='5,5')
    p.add_argument('--image-workers',type=int,default=6)
    args=p.parse_args();config,sequences=setup()
    gpus=args.render_gpus.split(',')
    if len(gpus)!=args.render_workers:raise ValueError('One GPU mapping per render worker is required')
    if args.setup_only:print('Setup complete; no GPU use or training');return
    if args.finalize_only:finalize(config,sequences);return
    qa=OUT/'seen_eval_qa/report.json';review=OUT/'seen_visual_review.json'
    if json.loads(qa.read_text())['status']!='passed':raise ValueError('Seen QA not passed')
    if json.loads(review.read_text())['qa_sha256']!=digest(qa):raise ValueError('Seen visual review missing/stale')
    state={'status':'rendering','started_at':time.time(),'training_started':False,'gpu_indices':sorted(set(map(int,gpus))),
           'render_workers':args.render_workers,'image_workers':args.image_workers}
    def save():
        state['updated_at']=time.time();tmp=OUT/'preparation_status.tmp'
        tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(OUT/'preparation_status.json')
    def run(command,log_path,gpu=''):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
        with log_path.open('w') as f:subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    save()
    try:
        if shutil.disk_usage(OUT).free<180*1024**3:raise RuntimeError('Need at least 180 GiB free before image archival')
        def render(shard):
            command=[sys.executable,'-m','gaussian_hand.export_diffusion_cache','--manifest',config['data']['train_manifest'],
                     '--manifest',config['data']['val_manifest'],'--output',str(OUT/'aligned'),'--storage','aligned_video',
                     '--baseline-mano-root',config['data']['loss_mask_root'],'--shard',str(shard),'--shards',str(args.render_workers)]
            run(command,OUT/f'render_{shard}.log',gpus[shard])
        with ThreadPoolExecutor(max_workers=args.render_workers) as pool:list(pool.map(render,range(args.render_workers)))
        state['status']='encoded_condition_qa';save()
        run([sys.executable,'-m','gaussian_hand.audit_diffusion_conditions','--render-root',str(OUT/'aligned'),
             '--storage','aligned_video','--output',str(OUT/'aligned_qa')],OUT/'aligned_qa.log')
        state['status']='exporting_image_sequences';save()
        def images(seq):
            if shutil.disk_usage(OUT).free<50*1024**3:raise RuntimeError('Free disk below 50 GiB safety margin')
            run([sys.executable,'-m','gaussian_hand.export_condition_sequences','--config',str(CONFIG),
                 '--manifest',str(OUT/'manifests'/f'{seq}.jsonl'),'--output',str(OUT/'image_sequences'/seq)],OUT/f'images_{seq}.log')
            print(json.dumps({'image_sequence_complete':seq}),flush=True)
        with ThreadPoolExecutor(max_workers=args.image_workers) as pool:list(pool.map(images,sequences))
        finalize(config,sequences)
        state.update(status='ready',finished_at=time.time());save()
    except BaseException as exc:
        state.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
