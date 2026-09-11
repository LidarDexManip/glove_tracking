"""CPU image export overlaps completed render sequences; never uses GPUs."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from gaussian_hand.prepare_diffusion_strict import ROOT,OUT,CONFIG


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--reverse',action='store_true')
    parser.add_argument('--label',default='')
    args=parser.parse_args()
    if args.workers<1 or (args.label and not args.label.isalnum()):raise ValueError('Invalid worker count/label')
    suffix='_'+args.label if args.label else ''
    sequences=sorted(p.stem for p in (OUT/'manifests').glob('P*.jsonl'))
    if args.reverse:sequences.reverse()
    submitted=set();futures={}
    def export(seq):
        if shutil.disk_usage(OUT).free<50*1024**3:raise RuntimeError('Disk safety margin reached')
        with (OUT/f'prefill{suffix}_images_{seq}.log').open('w') as log:
            subprocess.run([sys.executable,'-m','gaussian_hand.export_condition_sequences','--config',str(CONFIG),
                '--manifest',str(OUT/'manifests'/f'{seq}.jsonl'),'--output',str(OUT/'image_sequences'/seq)],cwd=ROOT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                stdout=log,stderr=subprocess.STDOUT,check=True)
        return seq
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(submitted)<len(sequences) or futures:
            for future in list(futures):
                if future.done():print('complete',future.result(),flush=True);del futures[future]
            for seq in sequences:
                if len(futures)>=args.workers:break
                if seq in submitted:continue
                if (OUT/'image_sequences'/seq/'_SUCCESS.json').exists():submitted.add(seq);continue
                if (OUT/'aligned'/seq/'_SUCCESS.json').exists():
                    futures[pool.submit(export,seq)]=seq;submitted.add(seq)
            status={'status':'running','submitted':len(submitted),'in_flight':list(futures.values()),
                    'complete':len(list((OUT/'image_sequences').glob('P*/_SUCCESS.json'))),'training_started':False}
            (OUT/f'image_prefill{suffix}_status.json').write_text(json.dumps(status,indent=2)+'\n')
            if len(submitted)<len(sequences) or futures:time.sleep(15)
    (OUT/f'image_prefill{suffix}_status.json').write_text(json.dumps({'status':'complete','sequences':len(sequences),'training_started':False})+'\n')


if __name__=='__main__':main()
