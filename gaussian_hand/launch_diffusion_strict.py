"""Validate preparation; launch training ONLY with an explicit --execute flag."""
import argparse
import os
import shlex
import subprocess
import sys

from gaussian_hand.diffusion_readiness import ROOT, validate_ready


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    config=ROOT/'configs/hand_restoration/hugg_aria_gaussian_surface_strict_v2.json'
    ready=validate_ready(config)
    command=[sys.executable,'-m','accelerate.commands.launch','--multi_gpu','--num_processes','8',
             '--num_machines','1','--mixed_precision','bf16','--main_process_port','29671',
             str(ROOT/'train_hand_restorer.py'),'--config',str(config)]
    print(f"READY: {ready['train_records']} crops; 8 GPUs; 15,380 optimizer steps")
    print(shlex.join(command),flush=True)
    if not args.execute:
        print('Validation only. No training started.');return
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    subprocess.run(command,cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
