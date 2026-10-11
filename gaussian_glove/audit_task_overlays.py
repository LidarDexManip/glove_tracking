"""Audit rendered cache coverage and decode fixed real samples, without a GPU."""
import argparse, json, tarfile
from pathlib import Path
import numpy as np
import torch
from gaussian_glove.task_overlay_dataset import TaskOverlayDataset
from gaussian_glove.prepare_task_overlays import digest, write_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    ready=json.loads((a.output/'READY.json').read_text())
    ds=TaskOverlayDataset(a.output/'train_manifest.jsonl')
    grouped={}
    for index,row in enumerate(ds.samples): grouped.setdefault(row['sequence_id'],[]).append(index)
    assert len(grouped)==14
    expected_names={'target','render','alpha','condition','overlay','mano_mask','visible_mask','edit_mask','sam_mask'}
    reports=[]; checked=0
    for sequence,indices in grouped.items():
        folder=a.output/sequence
        success=json.loads((folder/'_SUCCESS.json').read_text())
        actual_frames={ds.samples[i]['frame_index'] for i in indices}
        assert actual_frames==set(range(success['source_frames']))
        keys={ds.samples[i]['key'] for i in indices}
        with tarfile.open(folder/'crops.tar','r') as archive:
            members=archive.getmembers(); names={x.name for x in members}
            expected={f'{key}.{member}.png' for key in keys for member in expected_names}
            assert len(names)==len(members) and names==expected
            assert all(x.isfile() and x.size>0 for x in members)
        chosen=sorted({indices[int(j)] for j in np.linspace(0,len(indices)-1,min(8,len(indices)))})
        for i in chosen:
            item=ds[i]
            for k in ('target_rgb','condition_rgb','condition_mask','loss_mask','edit_mask'):
                assert torch.isfinite(item[k]).all()
            outside=item['edit_mask'][0]==0
            assert torch.equal(item['condition_rgb'][:,outside],item['target_rgb'][:,outside])
            for k in ('condition_mask','loss_mask','edit_mask'):
                assert set(item[k].unique().tolist())<={0.,1.}
            checked+=1
        for filename in ('crops.tar','frames.jsonl','render.mkv','alpha.mkv','overlay.mp4'):
            assert digest(folder/filename)==success['artifacts'][filename]['sha256']
        reports.append({'sequence':sequence,'frames':len(actual_frames),'crops':len(indices),
                        'archive_members':len(members),'decoded_samples':len(chosen)})
        print(json.dumps(reports[-1]),flush=True)
    ds.close()
    assert len(ds)==ready['training_crops']
    train_keys={(r['sequence_id'],r['key']) for r in ds.samples}
    seen=[json.loads(x) for x in (a.output/'seen_eval_manifest.jsonl').read_text().splitlines()]
    assert all((r['sequence_id'],r['key']) in train_keys for r in seen)
    result={'status':'passed','sessions':reports,'source_frames':sum(x['frames'] for x in reports),
            'training_crops':len(ds),'decoded_samples':checked,'quality_filtering':False,
            'checks':['all source frames represented','complete unique archive members','artifact SHA256',
                      'real dataset tensors finite','binary masks','PNG background exact outside edit',
                      'seen-eval subset of training'],'diffusion_training_started':False}
    write_json(a.output/'AUDIT.json',result)

if __name__=='__main__': main()
