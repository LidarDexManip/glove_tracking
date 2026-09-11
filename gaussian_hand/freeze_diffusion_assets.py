"""Freeze existing local SD assets/environment and audit Gaussian data exposure."""
import importlib.metadata
import json
from pathlib import Path
import subprocess

from gaussian_hand.prepare_diffusion_strict import ROOT,OUT,digest


def main():
    run=ROOT/'outputs/hand_restoration/hugg_aria_glove2hand_faithful_mano_adaptive_splitcrop_v1/run_metadata.json'
    baseline=json.loads(run.read_text());versions={}
    for name in ('torch','torchvision','diffusers','transformers','accelerate','numpy','opencv-python','smplx'):
        versions[name]=importlib.metadata.version(name)
        if versions[name]!=baseline['packages'][name]:raise ValueError(f'Environment drift: {name}')
    from huggingface_hub.constants import HF_HUB_CACHE
    repository=Path(HF_HUB_CACHE)/'models--runwayml--stable-diffusion-v1-5'
    revision=(repository/'refs/main').read_text().strip();snapshot=repository/'snapshots'/revision
    assets={str(p.relative_to(snapshot)):digest(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
    if not all(f in assets for f in ('unet/diffusion_pytorch_model.safetensors','vae/diffusion_pytorch_model.safetensors',
                                     'text_encoder/model.safetensors','scheduler/scheduler_config.json','tokenizer/vocab.json')):
        raise ValueError('Missing local SD assets')
    code={}
    for name in ('hand_restoration/diffusion.py','hand_restoration/samplers.py'):
        old=subprocess.check_output(['git','show',baseline['git_commit']+':'+name],cwd=ROOT)
        if old!=(ROOT/name).read_bytes():raise ValueError(f'Model/sampler changed: {name}')
        code[name]=digest(ROOT/name)
    train={}
    manifest=ROOT/'data/derived/hugg_aria_diffusion/adaptive_hand_crop_v1/train_manifest.jsonl'
    for line in manifest.open():
        row=json.loads(line);train.setdefault(row['sequence_id'],set()).add(row['frame_index'])
    total=0
    for p in (ROOT/'outputs/gaussian_hand/pretrain_v2').glob('P*/dataset/frames.jsonl'):
        for line in p.open():
            row=json.loads(line);total+=1
            if row['frame_index'] not in train[row['sequence']]:raise ValueError('Gaussian saw frame outside diffusion train pool')
    if total!=54468:raise ValueError('Unexpected Gaussian training population')
    frozen_index=ROOT/'baselines/faithful_mano_v2/ASSET_SHA256SUMS.jsonl'
    expected_index=(ROOT/'baselines/faithful_mano_v2/FREEZE_SHA256SUMS.txt').read_text().splitlines()
    expected_hash=next(line.split()[0] for line in expected_index if line.endswith('  ASSET_SHA256SUMS.jsonl'))
    if digest(frozen_index)!=expected_hash:raise ValueError('Historical asset index changed')
    source_assets=[]
    for line in frozen_index.open():
        asset=json.loads(line);path=ROOT/asset['path'];stat=path.stat()
        if digest(path)!=asset['sha256']:raise ValueError(f"Baseline input asset changed: {asset['path']}")
        source_assets.append({'path':asset['path'],'sha256':asset['sha256'],
                              'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    if len(source_assets)!=1360:raise ValueError('Incomplete original input asset coverage')
    record={'status':'frozen','versions':versions,'snapshot':str(snapshot),'revision':revision,'assets':assets,
            'code':code,'gaussian_training_crops':total,'gaussian_crops_outside_diffusion_train_pool':0,
            'source_asset_index_sha256':digest(frozen_index),'source_assets':source_assets,
            'scope_note':'Existing cached SD snapshot pinned now; baseline did not record initial asset hashes. Core model/sampler code and package versions match the frozen baseline.'}
    (OUT/'ASSET_LOCK.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('assets','code','source_assets')},indent=2))
    print(f'Verified {len(source_assets)} historical source video/MANO/SAM/wrist assets')


if __name__=='__main__':main()
