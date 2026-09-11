#!/usr/bin/env python3
"""Local web UI: paired video inference using each checkpoint's own render input."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import gradio as gr

from hand_restoration.paired_video import (
    ROOT, BASE_CONFIG, GAUSSIAN_CONFIG, VIDEO_NAMES, FrameCatalog, absolute,
    cache_key, checkpoint_choices, code_fingerprint, read_configs,
    resolve_checkpoint, save_json, sha256,
)


class ComparisonApp:
    def __init__(self, baseline_config=BASE_CONFIG, gaussian_config=GAUSSIAN_CONFIG,
                 output=ROOT/'outputs/hand_restoration/paired_video_web', gpu=0,
                 memory_fraction=.12, allow_shared_gpu=False):
        if gpu < 0 or not .05 <= memory_fraction <= .5:
            raise ValueError('Invalid GPU index or memory fraction (expected .05–.5).')
        self.config_paths = [absolute(baseline_config), absolute(gaussian_config)]
        self.configs = read_configs(*self.config_paths)
        self.config_hashes = [sha256(p) for p in self.config_paths]
        manifest = absolute(self.configs[0]['data']['train_manifest'])
        self.manifest_hash = sha256(manifest)
        self.catalog = FrameCatalog([json.loads(line) for line in manifest.open() if line.strip()])
        self.checkpoint_roots = [absolute(c['training']['output_dir']) for c in self.configs]
        self.output = absolute(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.gpu, self.memory_fraction, self.allow_shared_gpu = gpu, memory_fraction, allow_shared_gpu
        self.job_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.process = None
        self.owner = None
        self.cancelled = False

    def choices(self, arm):
        return [(p.name, str(p)) for p in checkpoint_choices(self.checkpoint_roots[arm])]

    def summary(self, sequence, start, duration, fps, hand):
        rows = self.catalog.plan(sequence, start, duration, int(fps), hand)
        return (f"{sequence} · 源帧 {rows[0]['frame_index']}–{rows[-1]['frame_index']} · "
                f"{len(rows)} 帧/模型，共 {2*len(rows)} 次单帧推理 · {duration:g} 秒视频，{int(fps)} FPS。\n\n"
                "两边使用相同源帧、相同 crop/mask、各自渲染输入。首次生成粗估 "
                f"{max(1, round(2*len(rows)*2/60))} 分钟，实际以 GPU 速度为准。自动 crop 可能随原 manifest 切换；不做额外平滑。")

    def cancel(self, owner):
        with self.state_lock:
            if self.owner != owner:
                return '当前会话没有正在运行的任务。'
            self.cancelled = True
            if self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            return '已请求取消；只停止本网页当前任务，不影响其他训练/推理进程。'

    def _assets(self, sequence):
        data = self.configs[0]['data']
        paths = [absolute(data['pinhole_root'])/sequence/'rgb_214_1_pinhole.mp4',
                 absolute(data['mask_root'])/sequence/'masks.sqlite',
                 absolute(data['wrist_ring_root'])/sequence/'wrist_ring_points.npz']
        wal = absolute(data['mask_root'])/sequence/'masks.sqlite-wal'
        if wal.exists():
            paths.append(wal)
        for config in self.configs:
            root = absolute(config['data']['render_root'])/sequence
            paths.extend([root/'reconstruction.mp4', root/'alpha.mkv', root/'_SUCCESS.json'])
        return {str(p): sha256(p) for p in sorted(set(paths))}

    @staticmethod
    def results(directory, cached=False):
        manifest = json.loads((directory/'manifest.json').read_text())
        step = manifest['checkpoint_steps']
        status = (f"{'复用已验证缓存' if cached else '生成完成'} · baseline step {step['baseline']} / "
                  f"Gaussian step {step['gaussian']} · {manifest['frames_per_arm']} 帧/模型 · "
                  f"{manifest['duration_seconds']:g} 秒。所有目标图、crop 和 mask 一致。\n\n"
                  f"原始生成视频用于观察模型本身；贴回视频会保留编辑区域外的原图。结果目录：`{directory}`")
        return (*(str(directory/name) for name in VIDEO_NAMES), str(directory/'contact_sheet.jpg'),
                str(directory/'manifest.json'), status)

    def generate(self, sequence, start, duration, fps, hand, baseline_checkpoint, gaussian_checkpoint,
                 steps, seed, rotate, owner='local', progress=lambda *args, **kwargs: None):
        if not self.job_lock.acquire(blocking=False):
            raise RuntimeError('另一个比较任务正在运行。')
        with self.state_lock:
            self.owner, self.cancelled = owner, False
        gpu_lock = None
        directory = None
        try:
            if not 1 <= steps <= 50 or int(steps) != steps or int(seed) != seed or not 0 <= seed < 2**32:
                raise ValueError('Denoising steps 必须为 1–50 整数；seed 必须为 0–2^32-1 整数。')
            rows = self.catalog.plan(sequence, start, duration, int(fps), hand)
            progress(.01, desc='核对 checkpoint、源数据与缓存')
            if [sha256(p) for p in self.config_paths] != self.config_hashes:
                raise ValueError('配置文件在网页启动后发生改变，请重启网页。')
            checkpoints = [resolve_checkpoint(v, root) for v, root in
                           zip((baseline_checkpoint, gaussian_checkpoint), self.checkpoint_roots)]
            if sha256(absolute(self.configs[0]['data']['train_manifest'])) != self.manifest_hash:
                raise ValueError('训练 manifest 在网页启动后发生改变，请重启网页。')
            recipe = {'config_paths': list(map(str,self.config_paths)),
                      'config_hashes': self.config_hashes,
                      'checkpoints': list(map(str,checkpoints)), 'checkpoint_roots': list(map(str,self.checkpoint_roots)),
                      'checkpoint_hashes': [sha256(p) for p in checkpoints],
                      'train_manifest_sha256': self.manifest_hash, 'records': rows,
                      'duration': float(duration), 'fps': int(fps), 'steps': int(steps), 'seed': int(seed),
                      'rotate': bool(rotate), 'gpu': self.gpu, 'memory_fraction': self.memory_fraction,
                      'allow_shared_gpu': self.allow_shared_gpu, 'input_assets': self._assets(sequence),
                      'code': code_fingerprint()}
            key = cache_key(recipe)
            cache = self.output/f'cache_{key}.json'
            if cache.exists():
                candidate = Path(json.loads(cache.read_text())['directory']).resolve()
                if candidate.parent == self.output and (candidate/'_SUCCESS.json').exists():
                    success = json.loads((candidate/'_SUCCESS.json').read_text())
                    if success['recipe_sha256'] == key and all((candidate/n).is_file() and sha256(candidate/n) == h
                            for n,h in success['files'].items()):
                        return self.results(candidate, cached=True)
            if shutil.disk_usage(self.output).free < 10 * 1024**3:
                raise RuntimeError('可用磁盘不足 10 GiB。')
            gpu_lock = open(f'/tmp/glove_tracking_paired_gpu{self.gpu}_{os.getuid()}.lock','a')
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('另一个 paired-video 网页实例正在使用这张 GPU。') from None
            directory = Path(tempfile.mkdtemp(prefix=f'{sequence}_{key[:10]}_',dir=self.output))
            save_json(directory/'request.json', {'recipe': recipe})
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(self.gpu), 'HF_HUB_OFFLINE':'1',
                   'TRANSFORMERS_OFFLINE':'1', 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'PYTHONUNBUFFERED':'1'}
            started = time.monotonic()
            with (directory/'worker.log').open('w') as log:
                with self.state_lock:
                    if self.cancelled:
                        raise RuntimeError('任务已取消。')
                    self.process = subprocess.Popen([sys.executable,'-u','-m','hand_restoration.paired_video',
                        '--request',str(directory/'request.json')], cwd=ROOT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                while self.process.poll() is None:
                    if time.monotonic()-started > 3600:
                        self.cancel(owner)
                        raise RuntimeError('超过 1 小时时限，已停止当前任务。')
                    status = directory/'status.json'
                    if status.exists():
                        state = json.loads(status.read_text())
                        progress(min(.98, state.get('completed',0)/max(state.get('total',1),1)),
                                 desc=f"{state.get('phase','准备')} · {state.get('completed',0)}/{state.get('total',len(rows)*2)}")
                    time.sleep(.5)
                if self.cancelled:
                    raise RuntimeError('任务已取消；未完成结果不会当作缓存使用。')
                if self.process.returncode != 0:
                    state_path = directory/'status.json'
                    error = json.loads(state_path.read_text()).get('error','') if state_path.exists() else ''
                    raise RuntimeError(f'推理失败：{error or "请检查 worker.log"}。日志：{directory/"worker.log"}')
            if not (directory/'_SUCCESS.json').exists():
                raise RuntimeError('缺少完成标记，不返回不完整视频。')
            save_json(cache, {'directory':str(directory)})
            progress(1, desc='已完成')
            return self.results(directory)
        except BaseException:
            with self.state_lock:
                if self.process is not None and self.process.poll() is None:
                    try:
                        os.killpg(self.process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.wait(timeout=10)
            raise
        finally:
            with self.state_lock:
                self.process = None
                self.owner = None
            if gpu_lock is not None:
                gpu_lock.close()
            self.job_lock.release()


def build_ui(app):
    first = 'P0001_f6cc0cc8' if 'P0001_f6cc0cc8' in app.catalog.sequences else app.catalog.sequences[0]
    first_start = app.catalog.next_start(first, 1018 if first=='P0001_f6cc0cc8' else 0, 3, 5, 'auto')
    checkpoints = [app.choices(i) for i in range(2)]
    with gr.Blocks(title='MANO vs Gaussian · paired video inference') as demo:
        gr.Markdown('# MANO baseline ↔ Gaussian surface\n'
                    '同一段源视频、相同 crop / mask / seed，分别使用各自的渲染输入和 checkpoint。'
                    '**这是训练集内对比；没有视频时序模型或额外平滑。**')
        with gr.Row():
            sequence = gr.Dropdown(app.catalog.sequences, value=first, label='Sequence（训练 manifest）')
            hand = gr.Dropdown([('自动：联合双手优先，否则较大手','auto'),('左手 crop','left'),
                                ('右手 crop','right'),('联合双手 crop','both')], value='auto', label='选择同一 crop')
        with gr.Row():
            start = gr.Number(value=first_start, precision=0, minimum=0, label='起始源帧（源视频 30 FPS）')
            duration = gr.Slider(1,10,value=3,step=1,label='视频时长（秒）')
            fps = gr.Dropdown([1,5,10,15,30],value=5,label='推理/播放 FPS（低 FPS 不改变片段时长）')
            find = gr.Button('寻找下一可用片段')
        with gr.Row():
            baseline = gr.Dropdown(checkpoints[0],value=checkpoints[0][0][1] if checkpoints[0] else None,label='Baseline checkpoint · naive MANO 输入')
            gaussian = gr.Dropdown(checkpoints[1],value=checkpoints[1][0][1] if checkpoints[1] else None,label='Gaussian surface checkpoint · Gaussian 输入')
            refresh = gr.Button('刷新 checkpoints')
        with gr.Row():
            steps = gr.Slider(1,50,value=20,step=1,label='两边相同的 denoising steps')
            seed = gr.Number(value=7,precision=0,label='两边/每帧相同 seed')
            rotate = gr.Checkbox(value=True,label='内部画面顺时针旋转 90°（仅展示，不改变模型输入）')
        plan = gr.Markdown(app.summary(first,first_start,3,5,'auto'))
        gr.Markdown(f'GPU **{app.gpu}** · 单任务队列 · 按需加载，任务结束释放 GPU · '
                    f'{"允许共享 GPU" if app.allow_shared_gpu else "GPU 繁忙时拒绝新推理"} · 相同请求可复用验证过的缓存。')
        with gr.Row():
            run = gr.Button('生成同步对比视频',variant='primary')
            cancel = gr.Button('取消当前会话的任务')
        status = gr.Markdown('就绪：启动网页和选择片段不占 GPU。')
        with gr.Tabs():
            with gr.Tab('原始模型输出（主要比较）'):
                raw_video = gr.Video(label='原图 crop | baseline 原始输出 | Gaussian 原始输出',interactive=False)
            with gr.Tab('各自输入（检查条件差异）'):
                input_video = gr.Video(label='原图 crop | naive MANO condition | Gaussian condition',interactive=False)
            with gr.Tab('贴回原视频（场景效果）'):
                full_video = gr.Video(label='原视频 | baseline 贴回 | Gaussian 贴回',interactive=False)
        sheet = gr.Image(label='同一中间帧：输出 / 输入 / 贴回',interactive=False)
        manifest = gr.File(label='下载对比参数、checkpoint 哈希和逐帧索引',interactive=False)
        def describe(seq,frame,seconds,rate,side):
            try:
                return app.summary(seq,frame,seconds,int(rate),side)
            except ValueError as error:
                return f'⚠️ {error}'
        def find_window(seq,frame,seconds,rate,side):
            try:
                frame = app.catalog.next_start(seq,frame,seconds,int(rate),side)
                return frame, app.summary(seq,frame,seconds,int(rate),side)
            except ValueError as error:
                raise gr.Error(str(error)) from error
        def refresh_choices():
            result = []
            for i in range(2):
                values = app.choices(i)
                result.append(gr.Dropdown(choices=values,value=values[0][1] if values else None))
            return tuple(result)
        def generate(seq,frame,seconds,rate,side,base,gauss,nsteps,random_seed,clockwise,
                     request: gr.Request, progress=gr.Progress()):
            try:
                return app.generate(seq,frame,seconds,int(rate),side,base,gauss,nsteps,random_seed,clockwise,
                                    owner=request.session_hash,progress=progress)
            except Exception as error:
                raise gr.Error(str(error)) from error
        def stop(request: gr.Request):
            return app.cancel(request.session_hash)
        controls = [sequence,start,duration,fps,hand]
        for control in controls:
            control.input(describe,inputs=controls,outputs=plan,queue=False,api_name=False)
        find.click(find_window,inputs=controls,outputs=[start,plan],queue=False,api_name='find_window')
        refresh.click(refresh_choices,outputs=[baseline,gaussian],queue=False,api_name='refresh_checkpoints')
        run.click(generate,inputs=controls+[baseline,gaussian,steps,seed,rotate],
                  outputs=[raw_video,input_video,full_video,sheet,manifest,status],
                  concurrency_limit=1,concurrency_id='paired-gpu',api_name='compare_video')
        cancel.click(stop,outputs=status,queue=False,api_name='cancel_comparison')
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-config',type=Path,default=BASE_CONFIG)
    parser.add_argument('--gaussian-config',type=Path,default=GAUSSIAN_CONFIG)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/hand_restoration/paired_video_web')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--memory-fraction',type=float,default=.12)
    parser.add_argument('--allow-shared-gpu',action='store_true')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=7862)
    args = parser.parse_args()
    username,password = os.getenv('HAND_COMPARE_USERNAME'),os.getenv('HAND_COMPARE_PASSWORD')
    if args.host not in ('127.0.0.1','localhost') and not (username and password):
        raise SystemExit('非本机访问必须设置 HAND_COMPARE_USERNAME 和 HAND_COMPARE_PASSWORD；建议使用 SSH 端口转发。')
    app = ComparisonApp(args.baseline_config,args.gaussian_config,args.output_dir,args.gpu,
                        args.memory_fraction,args.allow_shared_gpu)
    demo = build_ui(app)
    blocked = [ROOT/'data', ROOT/'.git', Path.home()/'.ssh', Path.home()/'.codex', Path.home()/'.cache', *app.checkpoint_roots]
    try:
        demo.queue(default_concurrency_limit=1,max_size=4).launch(
            server_name=args.host,server_port=args.port,share=False,inbrowser=False,show_error=True,
            auth=(username,password) if username and password else None,
            allowed_paths=[str(app.output)],blocked_paths=[str(p.resolve()) for p in blocked])
    finally:
        if app.owner is not None:
            app.cancel(app.owner)


if __name__ == '__main__':
    main()
