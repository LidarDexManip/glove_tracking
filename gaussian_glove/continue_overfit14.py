"""Continue the authorized 5000-step run for approximately one hour."""
import csv, json, math, os, shutil, subprocess, time, traceback
from pathlib import Path
ROOT = Path('/home/szeng87/glove_tracking')
OUT = ROOT / 'outputs/hand_restoration/overfit14_gaussian_sam31_v1'
STATUS = OUT / 'extension_status.json'
PYTHON = '/home/szeng87/miniconda3/envs/glove-hot3d/bin/python'
def status(state, **kwargs):
    payload = dict(state=state, requested_extra_seconds=3600, **kwargs)
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(STATUS)
def main():
    if STATUS.exists() and json.loads(STATUS.read_text()).get('state') in ('running', 'completed'):
        raise RuntimeError('Continuation already started; inspect before retrying')
    status('waiting_for_5000_steps')
    while not (OUT / 'exit_code.txt').exists():
        time.sleep(20)
    if int((OUT / 'exit_code.txt').read_text()) != 0:
        raise RuntimeError('Initial training did not exit successfully')
    checkpoint = OUT / 'trainer_state_step005000'
    for name in ['model.safetensors', 'optimizer.bin', 'scheduler.bin', 'custom_checkpoint_0.pkl']:
        if not (checkpoint / name).is_file():
            raise RuntimeError('Missing exact-resume checkpoint member: ' + name)
    with (OUT / 'training_log.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    rows = [r for r in rows if int(r['total_step']) <= 5000]
    last = rows[-1]
    if int(last['total_step']) != 5000:
        raise RuntimeError('Initial run did not reach step 5000')
    recent = [r for r in rows if r['run_id'] == last['run_id'] and int(r['total_step']) >= 4000]
    if len(recent) < 100:
        raise RuntimeError('Insufficient recent timing observations')
    first = recent[0]
    seconds_per_step = (float(last['elapsed_seconds']) - float(first['elapsed_seconds'])) / (int(last['total_step']) - int(first['total_step']))
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise RuntimeError('Invalid observed training speed')
    extra = max(1, round(3600 / seconds_per_step))
    if shutil.disk_usage(OUT).free < 20 * 1024**3:
        raise RuntimeError('Less than 20 GiB free; inspect storage before continuation')
    config = json.loads((ROOT / 'gaussian_glove/configs/overfit14_gaussian_sam31_v1.json').read_text())
    config['training']['max_train_steps'] = 5000 + extra
    config_path = OUT / 'extension_config.json'
    config_path.write_text(json.dumps(config, indent=2))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', NCCL_IB_DISABLE='1', NCCL_DEBUG='INFO', NCCL_CUMEM_ENABLE='0', NCCL_CUMEM_HOST_ENABLE='0', NCCL_NVLS_ENABLE='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH', '')
    details = dict(resume=str(checkpoint), extra_steps=extra, target_steps=5000+extra, observed_seconds_per_step=seconds_per_step, config=str(config_path))
    status('running', **details)
    started = time.monotonic()
    with (OUT / 'extension_train.log').open('w') as log:
        result = subprocess.run([PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=4', '--module', 'gaussian_glove.train_task_restorer', '--config', str(config_path), '--resume', str(checkpoint)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    status('completed' if result.returncode == 0 else 'failed', exit_code=result.returncode, elapsed_seconds=time.monotonic()-started, **details)
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('blocked', error=str(exc))
        traceback.print_exc()
        raise
