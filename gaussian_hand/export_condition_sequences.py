"""Archive all overlay/condition crop images; retain float online training path."""
from __future__ import annotations

import argparse
import io
import fcntl
import json
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np

from visualize_hugg_aria_faithful_inputs import build_dataset
from gaussian_hand.export_diffusion_cache import sha256


def last_frame_cache(function):
    """Reuse adjacent split-hand decodes, returning copies to avoid aliasing."""
    saved_key=saved=None
    def clone(value):
        return tuple(x.copy() for x in value) if isinstance(value,tuple) else value.copy()
    def call(*args):
        nonlocal saved_key,saved
        if args==saved_key:return clone(saved)
        value=function(*args);saved_key=args;saved=clone(value)
        return value
    return call


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cv2.setNumThreads(1)
    # CPU archival only: avoid 16 decoder threads per open video per worker.
    # This does not alter the training loader or decoded pixels.
    video_capture=cv2.VideoCapture
    cv2.VideoCapture=lambda path:video_capture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    config=json.loads(args.config.read_text());config['data']['train_manifest']=str(args.manifest)
    dataset=build_dataset(config)
    for name in ('_read_triplet','_read_loss_mask_render','_sam_mask'):
        setattr(dataset,name,last_frame_cache(getattr(dataset,name)))
    sequence=dataset.samples[0]['sequence_id']
    if any(r['sequence_id']!=sequence for r in dataset.samples):raise ValueError('Expected one sequence')
    render_success=Path(config['data']['render_root'])/sequence/'_SUCCESS.json'
    contract={'manifest_sha256':sha256(args.manifest),'config_sha256':sha256(args.config),
              'render_success_sha256':sha256(render_success),'sequence':sequence,'records':len(dataset)}
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'.export.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    done=args.output/'_SUCCESS.json'
    if done.exists():
        previous=json.loads(done.read_text())
        if any(previous.get(k)!=v for k,v in contract.items()):raise ValueError('Stale image archive')
        if not (args.output/'images.tar').exists():
            raise FileNotFoundError('Full image archive absent (possibly intentionally compacted). '
                                    'Re-export to a NEW --output directory; keep historical metadata unchanged.')
        print(json.dumps(previous));return
    if (args.output/'images.tar').exists():raise FileExistsError('Unverified image archive exists')
    start=time.monotonic();index=[];max_outside=0.;total_bytes=0
    with tarfile.open(args.output/'images.partial.tar','w') as archive:
        for i,record in enumerate(dataset.samples):
            sample=dataset[i]
            outside=sample['edit_mask_np']==0
            delta=float(np.abs(sample['condition_rgb_np'][outside]-sample['target_rgb_np'][outside]).max()) if outside.any() else 0.
            max_outside=max(max_outside,delta)
            if delta>1e-7:raise AssertionError(f'Outside-edit pixels changed at {i}')
            stem=f"{record['global_record_index']:07d}"
            names=[]
            for kind,extension,options in (('condition','.png',[cv2.IMWRITE_PNG_COMPRESSION,1]),
                                            ('overlay','.jpg',[cv2.IMWRITE_JPEG_QUALITY,95])):
                array=sample[f'{kind}_rgb_np']
                if not np.isfinite(array).all():raise AssertionError('Non-finite condition')
                pixels=(array[...,::-1]*255).round().clip(0,255).astype(np.uint8)
                ok,encoded=cv2.imencode(extension,pixels,options)
                if not ok:raise RuntimeError('Image encoding failed')
                content=encoded.tobytes();name=f'{stem}.{kind}{extension}'
                member=tarfile.TarInfo(name);member.size=len(content)
                archive.addfile(member,io.BytesIO(content));total_bytes+=len(content);names.append(name)
            index.append({**record,'condition_member':names[0],'overlay_member':names[1]})
    (args.output/'images.partial.tar').rename(args.output/'images.tar')
    (args.output/'index.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in index))
    result={**contract,'status':'complete','image_count':2*len(dataset),'encoded_bytes':total_bytes,
            'archive_bytes':(args.output/'images.tar').stat().st_size,'outside_edit_max_delta':max_outside,
            'elapsed_seconds':time.monotonic()-start,
            'training_policy':'Training computes float conditions from aligned render videos; PNG/JPEG archives are visualization assets, not quantized training inputs'}
    done.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
