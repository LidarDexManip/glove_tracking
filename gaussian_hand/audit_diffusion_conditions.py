"""Compare frozen MANO and Gaussian faithful conditions without diffusion training."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import types

import cv2
import numpy as np

from visualize_hugg_aria_faithful_inputs import build_dataset
import hand_restoration.hugg_aria_dataset as dataset_module

ROOT=Path(__file__).resolve().parents[1]


def make_config(config, manifest, render_root=None, storage=None):
    result=copy.deepcopy(config);result['data']['train_manifest']=str(manifest)
    if render_root is not None:
        result['data'].update(render_kind='gaussian',render_root=str(render_root),
                              render_rgb_semantics='premultiplied',render_storage=storage)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--storage',choices=('sparse_frames','aligned_video'),required=True)
    args=p.parse_args();cv2.setNumThreads(1)
    baseline=json.loads((ROOT/'configs/hand_restoration/hugg_aria_glove2hand_faithful_mano_adaptive_splitcrop_v1.json').read_text())
    manifest=ROOT/baseline['data']['val_manifest']
    old=build_dataset(make_config(baseline,manifest))
    new=build_dataset(make_config(baseline,manifest,args.render_root,args.storage))
    frozen=json.loads((ROOT/'baselines/adaptive_splitcrop_v1/BASELINE_FREEZE.json').read_text())
    source=subprocess.check_output(['git','show',f"{frozen['git']['training_commit']}:hand_restoration/hugg_aria_conditioning.py"],cwd=ROOT,text=True)
    module=types.ModuleType('_frozen_baseline_conditioning');sys.modules[module.__name__]=module
    exec(compile(source,'frozen_baseline_conditioning.py','exec'),module.__dict__)
    def frozen_builder(*positional, **keywords):
        if keywords.pop('render_rgb_semantics','straight') != 'straight':
            raise ValueError('Frozen baseline only accepts straight RGB')
        return module.build_faithful_condition(*positional, **keywords)
    current_builder=dataset_module.build_faithful_condition
    args.output.mkdir(parents=True,exist_ok=True);tiles=[];rows=[]
    fields=('target_rgb_np','mano_mask_np','condition_mask_np','edit_mask_np','loss_mask_np',
            'wrist_mask_np','visible_hand_mask_np','sam_excluded_mano_mask_np','dilated_hand_mask_np')
    for i,record in enumerate(old.samples):
        a=old[i];b=new[i]
        for key in fields:
            if not np.array_equal(a[key],b[key]):raise AssertionError(f'{key} differs at {i}')
        if a['crop_box']!=b['crop_box']:raise AssertionError('Crop changed')
        dataset_module.build_faithful_condition=frozen_builder
        try: frozen_sample=old[i]
        finally:dataset_module.build_faithful_condition=current_builder
        drift=float(np.max(np.abs(a['condition_rgb_np']-frozen_sample['condition_rgb_np'])))
        if drift>1e-7:raise AssertionError(f'Baseline implementation drift: {drift}')
        edit=b['edit_mask_np']>0;visible=b['visible_hand_mask_np']>0
        outside_delta=float(np.max(np.abs(b['condition_rgb_np'][~edit]-b['target_rgb_np'][~edit]))) if (~edit).any() else 0.
        if outside_delta>1e-7:raise AssertionError('Background changed outside edit mask')
        protected=(b['sam_excluded_mano_mask_np']>0)&~(b['wrist_mask_np']>0)
        protected_delta=float(np.max(np.abs(b['condition_rgb_np'][protected]-b['target_rgb_np'][protected]))) if protected.any() else 0.
        if protected_delta>1e-7:raise AssertionError('Protected object pixels changed')
        panels=[]
        for name,image in (('Original',b['target_rgb_np']),('Naive MANO condition',a['condition_rgb_np']),
                           ('Gaussian condition',b['condition_rgb_np'])):
            panel=cv2.resize((image[...,::-1]*255).round().clip(0,255).astype(np.uint8),(256,256))
            cv2.putText(panel,name,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,0),1,cv2.LINE_AA);panels.append(panel)
        tile=np.concatenate(panels,axis=1)
        cv2.putText(tile,f"{i}: {record['sequence_id']}:{record['frame_index']} {record['adaptive_handedness']}",
                    (5,249),cv2.FONT_HERSHEY_SIMPLEX,.40,(0,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(args.output/f'{i:03d}.jpg'),tile)
        tiles.append(tile)
        rows.append({'index':i,'sequence':record['sequence_id'],'frame_index':record['frame_index'],
                     'baseline_condition_max_drift':drift,'outside_edit_delta':outside_delta,'protected_delta':protected_delta,
                     'visible_pixels':int(visible.sum()),'condition_difference_mean':float(np.abs(a['condition_rgb_np']-b['condition_rgb_np']).mean()),
                     'wrist_overrides_occlusion_pixels':int(((b['wrist_mask_np']>0)&(b['sam_excluded_mano_mask_np']>0)).sum())})
    for start in range(0,len(tiles),12):
        cv2.imwrite(str(args.output/f'sheet_{start//12:02d}.jpg'),np.concatenate(tiles[start:start+12],axis=0))
    report={'status':'passed','records':len(rows),'storage':args.storage,
            'training_started':False,'baseline_commit':frozen['git']['training_commit'],
            'identical_fields':list(fields)+['crop_box'],'baseline_frozen_function_checked':True,
            'wrist_policy_note':'Existing wrist mask may override occlusion protection; unchanged in both arms',
            'samples':rows}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}))


if __name__=='__main__':main()
