"""Paired, hand-focused final-checkpoint evaluation on the frozen seen manifest."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from hand_restoration.paired_video import (
    ARM_NAMES, BASE_CONFIG, GAUSSIAN_CONFIG, absolute, code_fingerprint,
    read_configs, resolve_checkpoint, save_json, sha256,
)
from hand_restoration.inference import build_restorer, run_inference
from train_hand_restorer import make_hugg_aria_dataset, summarize_loss_statistics


REGIONS = dict(visible_hand='visible_hand_mask_np', mano='mano_mask_np',
               edit='edit_mask_np', wrist='wrist_mask_np',
               occluded_mano='sam_excluded_mano_mask_np')


def region_metrics(prediction, target, mask):
    mask = mask.astype(bool)
    if not mask.any():
        return None
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(difference[mask]**2))
    # RGB SSIM, Gaussian 11x11 sigma=1.5, population covariance, L=1.
    # Only windows entirely inside the region count: no background scores.
    x, y = prediction.astype(np.float64), target.astype(np.float64)
    blur = lambda a: cv2.GaussianBlur(a, (11, 11), 1.5)
    ux, uy = blur(x), blur(y)
    vx, vy, cov = blur(x*x)-ux*ux, blur(y*y)-uy*uy, blur(x*y)-ux*uy
    ssim_map = (((2*ux*uy+.01**2)*(2*cov+.03**2)) /
                ((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))).mean(axis=-1)
    interior = cv2.erode(mask.astype(np.uint8), np.ones((11,11),np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return dict(pixels=int(mask.sum()), mse=mse, mae_255=float(np.abs(difference[mask]).mean()*255),
                psnr_db=float(-10*np.log10(max(mse, 1e-20))),
                ssim_interior=float(ssim_map[interior].mean()) if interior.any() else None,
                ssim_pixels=int(interior.sum()))


def aggregate(rows):
    result = {}
    for region in REGIONS:
        entries = [r['regions'][region] for r in rows if r['regions'][region] is not None]
        if not entries:
            result[region] = None
            continue
        pooled_mse = sum(e['pixels']*e['mse'] for e in entries)/sum(e['pixels'] for e in entries)
        ssims = [e['ssim_interior'] for e in entries if e['ssim_interior'] is not None]
        result[region] = dict(n=len(entries), mean_psnr_db=float(np.mean([e['psnr_db'] for e in entries])),
            mean_mse=float(np.mean([e['mse'] for e in entries])),
            mean_mae_255=float(np.mean([e['mae_255'] for e in entries])),
            mean_ssim_interior=float(np.mean(ssims)) if ssims else None, ssim_n=len(ssims),
            pooled_psnr_db=float(-10*np.log10(max(pooled_mse,1e-20))))
    stats = torch.tensor([r['noise_loss_statistics'] for r in rows],dtype=torch.float64).sum(0)
    return dict(regions=result, paired_noise_loss=summarize_loss_statistics(stats))


def log_summary(config):
    path = absolute(config['training']['output_dir'])/'training_log.csv'
    rows = list(csv.DictReader(path.open()))
    keys = ('loss','hand_region_loss','background_region_loss')
    return dict(path=str(path), sha256=sha256(path), steps=len(rows),
        last_1000={k:float(np.mean([float(r[k]) for r in rows[-1000:]])) for k in keys},
        final_validation={k:float(rows[-1][k]) for k in ('validation_loss','validation_hand_region_loss',
                                                       'validation_background_region_loss')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--noise-repeats',type=int,default=4)
    args = parser.parse_args()
    configs = read_configs()
    checkpoint_paths = [resolve_checkpoint(absolute(c['training']['output_dir'])/'controlnet_final.pt',
                                          absolute(c['training']['output_dir'])) for c in configs]
    manifest = absolute(configs[0]['data']['val_manifest'])
    records = [json.loads(l) for l in manifest.open() if l.strip()]
    train_keys = {(r['sequence_id'],r['frame_index'],r['adaptive_handedness'],tuple(r['adaptive_crop_box_xyxy']))
                  for r in map(json.loads,absolute(configs[0]['data']['train_manifest']).open())}
    assert all((r['sequence_id'],r['frame_index'],r['adaptive_handedness'],tuple(r['adaptive_crop_box_xyxy']))
               in train_keys for r in records), 'Seen evaluation must be a training subset.'
    del train_keys
    directory = absolute(args.output)
    directory.mkdir(parents=True,exist_ok=False)
    save_json(directory/'provenance.json',dict(configs=[str(BASE_CONFIG),str(GAUSSIAN_CONFIG)],
        config_hashes=[sha256(BASE_CONFIG),sha256(GAUSSIAN_CONFIG)], checkpoints=list(map(str,checkpoint_paths)),
        checkpoint_hashes=[sha256(p) for p in checkpoint_paths], manifest=str(manifest),
        manifest_sha256=sha256(manifest), records=records, inference_steps=args.steps,seed=args.seed,
        noise_repeats=args.noise_repeats, code=code_fingerprint(), evaluator_sha256=sha256(__file__),
        scope='Seen training data, raw generated RGB before paste-back; same frame/crop/mask/seed; FP32 batch=1.',
        ssim='RGB Gaussian 11x11 sigma1.5 population covariance L=1; centers eroded 5px inside mask.'))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(.12)
    started = time.monotonic()
    model = build_restorer(configs[0], device='cuda')
    signatures, all_results = [], {}
    fields = ('target_rgb_np','mano_mask_np','visible_hand_mask_np','condition_mask_np',
              'edit_mask_np','loss_mask_np','wrist_mask_np','sam_excluded_mano_mask_np')
    for arm, config, checkpoint in zip(ARM_NAMES,configs,checkpoint_paths):
        payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert payload['config'] == asdict(model.config)
        assert payload['global_step'] == 15380
        model.controlnet.load_state_dict(payload['state_dict'],strict=True)
        del payload
        model.controlnet.eval()
        dataset = make_hugg_aria_dataset(config,manifest,include_numpy=True)
        results = []
        folder = directory/arm
        folder.mkdir()
        with (folder/'frames.jsonl').open('w') as output:
            for index, record in enumerate(records):
                sample = dataset[index]
                sig = {k:hashlib.sha256(np.ascontiguousarray(sample[k]).tobytes()).hexdigest() for k in fields}
                sig['crop'] = sample['crop_box'].as_list()
                if arm == 'baseline':
                    signatures.append(sig)
                else:
                    assert sig == signatures[index], f'Mismatch at {index}'
                statistics = torch.zeros(8,device='cuda',dtype=torch.float64)
                with torch.inference_mode():
                    for repeat in range(args.noise_repeats):
                        generator = torch.Generator(device='cuda').manual_seed(args.seed+index*1009+repeat)
                        _, stat = model.training_loss(sample['target_rgb'].unsqueeze(0),
                            sample['condition_rgb'].unsqueeze(0),condition_mask=sample['condition_mask'].unsqueeze(0),
                            edit_mask=sample['loss_mask'].unsqueeze(0),
                            hand_loss_weight=config['training']['hand_loss_weight'],generator=generator,
                            return_statistics=True)
                        statistics += stat.double()
                generated = run_inference(model,sample,config,steps=args.steps,seed=args.seed).generated
                assert np.isfinite(generated).all()
                target = sample['target_rgb_np']
                row = dict(index=index,sequence_id=record['sequence_id'],subject=record['sequence_id'].split('_')[0],
                    frame=record['frame_index'],hand=record['adaptive_handedness'],
                    regions={name:region_metrics(generated,target,sample[key]) for name,key in REGIONS.items()},
                    noise_loss_statistics=statistics.cpu().tolist(),invariants=sig,
                    condition_sha256=hashlib.sha256(sample['condition_rgb_np'].tobytes()).hexdigest())
                if not cv2.imwrite(str(folder/f'{index:04d}_raw.png'),
                                   (generated[...,::-1]*255).round().clip(0,255).astype(np.uint8)):
                    raise OSError('Failed to save prediction')
                if arm == 'baseline':
                    np.savez_compressed(folder/f'{index:04d}_reference.npz',target=target,
                                        **{name:sample[key] for name,key in REGIONS.items()})
                results.append(row)
                output.write(json.dumps(row)+'\n'); output.flush()
                if index % 10 == 0 or index == len(records)-1:
                    print(f'{arm} {index+1}/{len(records)} elapsed={time.monotonic()-started:.1f}s',flush=True)
        all_results[arm] = results
        del dataset
    peak = torch.cuda.max_memory_allocated()
    del model
    torch.cuda.empty_cache()
    report = dict(n=len(records),sequences=len({r['sequence_id'] for r in records}),
        subjects=dict(Counter(r['sequence_id'].split('_')[0] for r in records)),
        overall={a:aggregate(rows) for a,rows in all_results.items()},
        by_subject={subject:{a:aggregate([r for r in rows if r['subject']==subject]) for a,rows in all_results.items()}
                    for subject in sorted({r['sequence_id'].split('_')[0] for r in records})},
        training_logs={a:log_summary(c) for a,c in zip(ARM_NAMES,configs)},peak_gpu_bytes=peak,
        elapsed_seconds=time.monotonic()-started,all_targets_crops_masks_identical=True)
    pairs = defaultdict(list)
    for b,g in zip(all_results['baseline'],all_results['gaussian']):
        if b['regions']['visible_hand'] is not None:
            pairs[b['sequence_id']].append(g['regions']['visible_hand']['psnr_db']-b['regions']['visible_hand']['psnr_db'])
    differences = np.array([np.mean(v) for v in pairs.values()])
    rng = np.random.default_rng(7)
    bootstrap = rng.choice(differences,(10000,len(differences)),replace=True).mean(axis=1)
    report['visible_hand_paired_psnr'] = dict(sequence_macro_delta_db=float(differences.mean()),
        sequence_bootstrap_95ci=np.quantile(bootstrap,[.025,.975]).tolist(),
        positive_sequences=int((differences>0).sum()),total_sequences=len(differences),
        positive_crops=sum(d>0 for v in pairs.values() for d in v),total_crops=sum(map(len,pairs.values())),
        caveat='CI resamples sequences, not subjects; this fixed seen sample is not a generalization test.')
    save_json(directory/'summary.json',report)
    print(json.dumps(report['overall'],indent=2),flush=True)
    print('COMPLETE',directory,flush=True)


if __name__ == '__main__':
    main()
