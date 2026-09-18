"""Read-only gate for controlled training; no model initialization or training."""
from __future__ import annotations

import hashlib
import json
import importlib.metadata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'configs/hand_restoration/hugg_aria_glove2hand_faithful_mano_adaptive_splitcrop_v1.json'
PREPARED=ROOT/'configs/hand_restoration/hugg_aria_gaussian_surface_strict_v2.json'
FOUR_GPU_LAYOUT='four_gpu_batch64_same_data'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_comparison(config):
    baseline=json.loads(BASELINE.read_text())
    freeze=json.loads((ROOT/'baselines/adaptive_splitcrop_v1/BASELINE_FREEZE.json').read_text())
    if digest(BASELINE)!=freeze['configuration']['sha256']:raise ValueError('Frozen baseline config changed')
    if config['model']!=baseline['model'] or config['seed']!=baseline['seed']:
        raise ValueError('Model initialization/seed differs from baseline')
    if config.get('inference')!=baseline.get('inference'):
        raise ValueError('Evaluation inference settings differ from baseline')
    training=dict(config['training']);training['output_dir']=baseline['training']['output_dir']
    layout=config['comparison'].get('execution_layout','original_eight_gpu')
    expected_world=8
    if layout==FOUR_GPU_LAYOUT:
        expected_world=4
        if (training['batch_size'],training['validation_batch_size'],training['gradient_accumulation_steps'])!=(64,16,1):
            raise ValueError('Authorized four-GPU layout requires batch64/validation16/accumulation1')
        training['batch_size']=baseline['training']['batch_size']
        training['validation_batch_size']=baseline['training']['validation_batch_size']
    elif layout!='original_eight_gpu':
        raise ValueError('Unknown execution layout')
    if training!=baseline['training']:raise ValueError('Training hyperparameters differ from baseline')
    data=dict(config['data'])
    if data.pop('render_rgb_semantics',None)!='premultiplied' or data.pop('render_storage',None)!='aligned_video':
        raise ValueError('Unexpected Gaussian cache semantics')
    if data['render_kind']!='gaussian':raise ValueError('Expected Gaussian treatment')
    data['render_kind']=baseline['data']['render_kind'];data['render_root']=baseline['data']['render_root']
    if data!=baseline['data']:raise ValueError('Data/crop/mask configuration differs from baseline')
    if config['comparison']['expected_world_size']!=expected_world:raise ValueError('World size does not match the declared layout')
    if config['training']['batch_size']*expected_world*config['training']['gradient_accumulation_steps']!=256:
        raise ValueError('Effective batch must remain 256')


def validate_ready(config_path, world_size=None, resume=None):
    config_path=Path(config_path);config=json.loads(config_path.read_text());validate_comparison(config)
    path=ROOT/config['comparison']['required_readiness']
    if not path.is_file():raise RuntimeError(f'Data preparation is not ready: {path}')
    ready=json.loads(path.read_text())
    prepared_hash=digest(config_path)
    if config['comparison'].get('execution_layout')==FOUR_GPU_LAYOUT:
        prepared=json.loads(PREPARED.read_text());validate_comparison(prepared)
        prepared_hash=digest(PREPARED)
        if config['comparison'].get('prepared_config_sha256')!=prepared_hash:
            raise ValueError('Four-GPU run is not bound to the audited preparation config')
        if config['data']!=prepared['data'] or config['comparison']['required_readiness']!=prepared['comparison']['required_readiness']:
            raise ValueError('Four-GPU run must use the exact prepared data')
    if ready['status']!='ready' or ready['config_sha256']!=prepared_hash:raise ValueError('Readiness/config mismatch')
    if world_size is not None and world_size!=config['comparison']['expected_world_size']:
        raise ValueError('Actual world size differs from the selected execution layout')
    if resume is not None:
        resume=Path(resume).resolve();output=(ROOT/config['training']['output_dir']).resolve()
        if not resume.is_dir() or resume.parent!=output:raise ValueError('Only exact resume of this new run is allowed')
    for key,value in ready['manifest_sha256'].items():
        if digest(ROOT/config['data'][key])!=value:raise ValueError(f'Prepared manifest changed: {key}')
    root=path.parent
    if ready.get('dataset_audit_sha256')!=digest(root/'dataset_audit.json'):
        raise ValueError('Final archive/dataloader audit missing or changed')
    audit=json.loads((root/'dataset_audit.json').read_text())
    if audit['status']!='passed' or audit['config_sha256']!=prepared_hash:
        raise ValueError('Dataset audit/config mismatch')
    if ready.get('asset_lock_sha256')!=digest(root/'ASSET_LOCK.json'):
        raise ValueError('Prepared asset lock changed')
    assets=json.loads((root/'ASSET_LOCK.json').read_text())
    if digest(ROOT/'baselines/faithful_mano_v2/ASSET_SHA256SUMS.jsonl')!=assets['source_asset_index_sha256']:
        raise ValueError('Historical source-asset index changed')
    for asset in assets['source_assets']:
        source=ROOT/asset['path'];stat=source.stat()
        if stat.st_size!=asset['size_bytes']:raise ValueError(f'Input asset size changed: {source}')
        if stat.st_mtime_ns!=asset['mtime_ns'] and digest(source)!=asset['sha256']:
            raise ValueError(f'Input asset content changed: {source}')
    for name,version in assets['versions'].items():
        if importlib.metadata.version(name)!=version:raise ValueError(f'Environment drift: {name}')
    for name,value in assets['code'].items():
        if digest(ROOT/name)!=value:raise ValueError(f'Core model/sampler changed: {name}')
    snapshot=Path(assets['snapshot'])
    from huggingface_hub.constants import HF_HUB_CACHE
    if Path(HF_HUB_CACHE).resolve()!=snapshot.parents[2].resolve():
        raise ValueError('HF_HUB_CACHE does not point to the frozen pretrained assets')
    if (snapshot.parent.parent/'refs/main').read_text().strip()!=assets['revision']:
        raise ValueError('Cached pretrained revision changed')
    for name,value in assets['assets'].items():
        if digest(snapshot/name)!=value:raise ValueError(f'Pretrained asset changed: {name}')
    from gaussian_hand.archive_retention import validate_archive_storage
    validate_archive_storage(root)
    for r in ready['rows']:
        seq=r['sequence']
        for relative,key in ((f'aligned/{seq}/_SUCCESS.json','render_success_sha256'),
                             (f'image_sequences/{seq}/_SUCCESS.json','images_success_sha256')):
            if digest(root/relative)!=r[key]:raise ValueError(f'Prepared metadata changed: {seq}')
        for relative in (f'aligned/{seq}/reconstruction.mp4',f'aligned/{seq}/alpha.mkv'):
            if not (root/relative).is_file():raise FileNotFoundError(root/relative)
    return ready
