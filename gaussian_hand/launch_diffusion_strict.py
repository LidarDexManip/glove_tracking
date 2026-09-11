"""Validate preparation; launch training ONLY with an explicit --execute flag."""
import argparse
import json
import os
import shlex
import subprocess
import sys

from gaussian_hand.diffusion_readiness import ROOT, validate_ready


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--four-gpu',action='store_true',help='User-selected GPU 0-3, batch64 per GPU; same prepared data')
    args=p.parse_args()
    name='hugg_aria_gaussian_surface_v2_4gpu.json' if args.four_gpu else 'hugg_aria_gaussian_surface_strict_v2.json'
    config=ROOT/'configs/hand_restoration'/name
    ready=validate_ready(config)
    world=4 if args.four_gpu else 8
    command=[sys.executable,'-m','accelerate.commands.launch','--multi_gpu','--num_processes',str(world),
             '--num_machines','1','--mixed_precision','bf16','--main_process_port','29671',
             str(ROOT/'train_hand_restorer.py'),'--config',str(config)]
    print(f"READY: {ready['train_records']} crops; {world} GPUs; global batch 256; 15,380 optimizer steps")
    print(shlex.join(command),flush=True)
    if not args.execute:
        print('Validation only. No training started.');return
    runtime=json.loads(config.read_text())
    output=ROOT/runtime['training']['output_dir']
    if output.exists() and any(p.name not in {'launch.log','launch_status.json'} for p in output.iterdir()):
        raise FileExistsError('Training output already contains a run; refuse to overwrite or silently restart')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,range(world))),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    subprocess.run(command,cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
