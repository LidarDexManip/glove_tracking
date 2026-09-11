"""Matched-frame MANO/Gaussian video comparison. GPU work lives in a subprocess."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / 'configs/hand_restoration/hugg_aria_glove2hand_faithful_mano_adaptive_splitcrop_v1.json'
GAUSSIAN_CONFIG = ROOT / 'configs/hand_restoration/hugg_aria_gaussian_surface_v2_4gpu.json'
ARM_NAMES = ('baseline', 'gaussian')
VIDEO_NAMES = ('outputs_comparison.mp4', 'inputs_comparison.mp4', 'full_frame_comparison.mp4')


def absolute(path):
    return (ROOT / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_configs(baseline_path=BASE_CONFIG, gaussian_path=GAUSSIAN_CONFIG):
    configs = [json.loads(Path(p).read_text()) for p in (baseline_path, gaussian_path)]
    baseline, gaussian = configs
    if baseline['model'] != gaussian['model'] or baseline['inference'] != gaussian['inference']:
        raise ValueError('Model architecture/prompts and inference settings must match.')
    if baseline.get('condition', {}) != gaussian.get('condition', {}):
        raise ValueError('Condition-building settings must match.')
    if baseline['data'].get('output_size', 512) != 512:
        raise ValueError('This comparison layout requires the trained 512px crop size.')
    if baseline['data']['render_kind'] != 'mano' or gaussian['data']['render_kind'] != 'gaussian':
        raise ValueError('Expected MANO baseline and Gaussian surface inputs, in that order.')
    if gaussian['data'].get('render_rgb_semantics') != 'premultiplied':
        raise ValueError('Gaussian input must explicitly use premultiplied RGB.')
    def common(data):
        return {k: v for k, v in data.items() if k not in
                ('render_kind', 'render_root', 'render_rgb_semantics', 'render_storage')}
    if common(baseline['data']) != common(gaussian['data']):
        raise ValueError('The two arms must use identical manifests/crops/masks/targets.')
    return configs


class FrameCatalog:
    def __init__(self, records):
        self.records = records
        self.frames = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(records):
            self.frames[row['sequence_id']][int(row['frame_index'])].append(index)
        self.sequences = sorted(self.frames)

    def selected(self, sequence, hand):
        if sequence not in self.frames or hand not in ('auto', 'left', 'right', 'both'):
            raise ValueError('请选择有效 sequence 和 crop 类型。')
        selected = {}
        for frame, indices in self.frames[sequence].items():
            candidates = [i for i in indices if hand == 'auto' or self.records[i]['adaptive_handedness'] == hand]
            if candidates:
                # Deterministic and independent of model results. No per-arm selection.
                selected[frame] = max(candidates, key=lambda i: (
                    self.records[i]['adaptive_handedness'] == 'both',
                    self.records[i].get('adaptive_component_area', 0), -i))
        return selected

    @staticmethod
    def offsets(duration, fps):
        if fps not in (1, 5, 10, 15, 30) or not math.isfinite(duration) or not 1 <= duration <= 10:
            raise ValueError('时长限 1–10 秒；采样 FPS 必须为 1/5/10/15/30。')
        count = duration * fps
        if abs(count - round(count)) > 1e-6:
            raise ValueError('时长 × FPS 必须是整数。')
        return [i * (30 // fps) for i in range(round(count))]

    def plan(self, sequence, start, duration, fps, hand):
        if not math.isfinite(start) or start < 0 or int(start) != start:
            raise ValueError('起始源帧必须为非负整数。')
        offsets = self.offsets(duration, fps)
        selected = self.selected(sequence, hand)
        frames = [int(start) + offset for offset in offsets]
        missing = [f for f in frames if f not in selected]
        if missing:
            raise ValueError(f'{len(missing)}/{len(frames)} 个采样帧无对应训练 crop（如 {missing[:5]}）。'
                             '请点击“寻找下一可用片段”；不会跳帧、复制缺失帧或使用空白渲染。')
        return [dict(self.records[selected[f]], global_record_index=selected[f]) for f in frames]

    def next_start(self, sequence, start, duration, fps, hand):
        offsets = self.offsets(duration, fps)
        selected = self.selected(sequence, hand)
        if not selected:
            raise ValueError('该 sequence 没有所选类型的 crop。')
        frames = sorted(selected)
        # Prefer subsequent windows, then wrap to the beginning.
        for candidate in [f for f in frames if f >= start] + [f for f in frames if f < start]:
            if all(candidate + offset in selected for offset in offsets):
                return candidate
        raise ValueError('找不到完整片段。请缩短时长、降低 FPS 或改选自动 crop。')


def checkpoint_choices(root, minimum_size=1_000_000_000, minimum_age=10):
    root = Path(root).resolve()
    choices = []
    for path in root.glob('controlnet*.pt'):
        if path.resolve().parent != root or not path.is_file():
            continue
        stat = path.stat()
        if stat.st_size < minimum_size or time.time() - stat.st_mtime < minimum_age:
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                if not any(n.endswith('/data.pkl') for n in archive.namelist()):
                    continue
        except (OSError, zipfile.BadZipFile):
            continue  # An incomplete torch.save is never offered in the UI.
        if path.name == 'controlnet_final.pt':
            key = (1, 0)
        elif path.stem.startswith('controlnet_step') and path.stem[15:].isdigit():
            key = (0, int(path.stem[15:]))
        else:
            continue
        choices.append((key, path))
    return [path for _, path in sorted(choices, reverse=True)]


def resolve_checkpoint(value, root):
    path = Path(value).resolve()
    if path not in checkpoint_choices(root):
        raise ValueError('Checkpoint 不属于该模型的允许目录，或尚未完整写入。')
    return path


def code_fingerprint():
    names = ['hand_restoration/paired_video.py', 'hand_restoration/diffusion.py',
             'hand_restoration/inference.py', 'hand_restoration/hugg_aria_dataset.py',
             'hand_restoration/hugg_aria_conditioning.py', 'train_hand_restorer.py']
    return {name: sha256(ROOT / name) for name in names}


def cache_key(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def worker(request_path):
    import cv2
    import numpy as np
    import torch
    from hand_restoration.inference import build_restorer, run_inference
    from train_hand_restorer import make_hugg_aria_dataset

    request_path = Path(request_path)
    directory = request_path.parent
    request = json.loads(request_path.read_text())
    recipe = request['recipe']
    configs = read_configs(*recipe['config_paths'])
    for path, expected in zip(recipe['config_paths'], recipe['config_hashes']):
        if sha256(path) != expected:
            raise ValueError('Config changed after submission.')
    if code_fingerprint() != recipe['code']:
        raise ValueError('Inference code changed after submission.')
    for config in configs:
        if sha256(absolute(config['data']['train_manifest'])) != recipe['train_manifest_sha256']:
            raise ValueError('Training manifest changed.')
    for path, expected in recipe['input_assets'].items():
        if sha256(path) != expected:
            raise ValueError(f'Input asset changed: {path}')
    # Revalidate trusted checkpoint locations and content in the GPU process.
    for path, root, expected in zip(recipe['checkpoints'], recipe['checkpoint_roots'], recipe['checkpoint_hashes']):
        resolve_checkpoint(path, root)
        if sha256(path) != expected:
            raise ValueError('Checkpoint changed after submission.')
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if not recipe['allow_shared_gpu']:
        row = subprocess.check_output(['nvidia-smi', '-i', str(recipe['gpu']),
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        memory, utilization = map(int, row.strip().split(','))
        if memory > 1024 or utilization > 10:
            raise RuntimeError('GPU 繁忙；未启动推理。可换 GPU，或明确用 --allow-shared-gpu 启动网页。')
    torch.cuda.set_per_process_memory_fraction(recipe['memory_fraction'])
    free, _ = torch.cuda.mem_get_info()
    if free < 16 * 1024**3:
        raise RuntimeError('GPU 可用显存不足 16 GiB；未加载模型。')
    records = recipe['records']
    manifest = directory / 'selected_frames.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
    started = time.monotonic()
    state = {'status': 'running', 'completed': 0, 'total': 2 * len(records), 'phase': 'loading_model'}
    def update(**changes):
        state.update(changes, elapsed_seconds=time.monotonic()-started)
        save_json(directory / 'status.json', state)
    def pixels(value):
        return (value[..., ::-1] * 255).round().clip(0, 255).astype(np.uint8)
    def write(path, array):
        if not cv2.imwrite(str(path), array):
            raise OSError(path)
    def hash_array(value):
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    invariant_fields = ('target_rgb_np', 'mano_mask_np', 'condition_mask_np', 'edit_mask_np',
                        'loss_mask_np', 'wrist_mask_np', 'visible_hand_mask_np', 'sam_excluded_mano_mask_np')
    signatures, metrics, steps = [], {}, {}
    update()
    model = build_restorer(configs[0], device='cuda')
    try:
        for arm, config, checkpoint in zip(ARM_NAMES, configs, recipe['checkpoints']):
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if payload['config'] != asdict(model.config):
                raise ValueError(f'{arm}: checkpoint architecture/config does not match.')
            model.controlnet.load_state_dict(payload['state_dict'], strict=True)
            steps[arm] = int(payload['global_step'])
            del payload
            model.controlnet.eval()
            dataset = make_hugg_aria_dataset(config, manifest, include_numpy=True)
            folder = directory / arm
            folder.mkdir()
            metrics[arm] = []
            update(phase=arm)
            for index, record in enumerate(records):
                sample = dataset[index]
                signature = {key: hash_array(sample[key]) for key in invariant_fields}
                signature['crop'] = sample['crop_box'].as_list()
                if arm == 'baseline':
                    signatures.append(signature)
                elif signature != signatures[index]:
                    raise AssertionError(f'Target/crop/mask mismatch at source frame {record["frame_index"]}')
                result = run_inference(model, sample, config, steps=recipe['steps'], seed=recipe['seed'])
                if not np.isfinite(result.generated).all():
                    raise ValueError('Non-finite model output.')
                write(folder / f'{index:04d}_condition.png', pixels(sample['condition_rgb_np']))
                write(folder / f'{index:04d}_raw.png', pixels(result.generated))
                write(folder / f'{index:04d}_full.png', cv2.resize(pixels(result.restored_full), (512, 512)))
                if arm == 'baseline':
                    write(folder / f'{index:04d}_target.png', pixels(sample['target_rgb_np']))
                    write(folder / f'{index:04d}_source.png', cv2.resize(pixels(sample['full_target_rgb_np']), (512, 512)))
                metrics[arm].append({'frame': record['frame_index'], 'raw_psnr_full': result.generated_psnr_full,
                                    'raw_psnr_edit_mask': result.generated_psnr_masked})
                update(completed=state['completed']+1)
            del dataset
        peak = torch.cuda.max_memory_allocated()
    finally:
        del model
        torch.cuda.empty_cache()
    update(phase='encoding')
    layouts = [
        ('outputs_comparison.mp4', [('baseline', 'target', 'Original target'), ('baseline', 'raw', 'Baseline raw output'), ('gaussian', 'raw', 'Gaussian raw output')]),
        ('inputs_comparison.mp4', [('baseline', 'target', 'Original target'), ('baseline', 'condition', 'Naive MANO input'), ('gaussian', 'condition', 'Gaussian input')]),
        ('full_frame_comparison.mp4', [('baseline', 'source', 'Original video'), ('baseline', 'full', 'Baseline paste-back'), ('gaussian', 'full', 'Gaussian paste-back')]),
    ]
    gallery = []
    for filename, columns in layouts:
        process = subprocess.Popen(['ffmpeg', '-n', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', '1536x576', '-r', str(recipe['fps']), '-i', '-', '-an', '-c:v', 'libx264',
            '-threads', '2', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(directory / filename)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for index, record in enumerate(records):
                panel = np.zeros((576, 1536, 3), np.uint8)
                for column, (arm, kind, title) in enumerate(columns):
                    frame = cv2.imread(str(directory / arm / f'{index:04d}_{kind}.png'))
                    if frame is None:
                        raise OSError('Missing inference output frame.')
                    if recipe['rotate']:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    panel[64:, column*512:(column+1)*512] = frame
                    cv2.putText(panel, title, (column*512+12, 25), cv2.FONT_HERSHEY_SIMPLEX, .60, (255,255,255), 1, cv2.LINE_AA)
                label = f"{record['sequence_id']} | frame {record['frame_index']} | {record['adaptive_handedness']} | {recipe['fps']} fps | steps {steps['baseline']} / {steps['gaussian']}"
                cv2.putText(panel, label, (12, 51), cv2.FONT_HERSHEY_SIMPLEX, .46, (190,190,190), 1, cv2.LINE_AA)
                process.stdin.write(panel.tobytes())
                if index == len(records)//2:
                    gallery.append(panel)
        finally:
            process.stdin.close()
            error = process.stderr.read().decode()
            if process.wait():
                raise RuntimeError(error)
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames,duration','-of','json',str(directory / filename)], text=True))['streams'][0]
        if int(info['nb_read_frames']) != len(records) or abs(float(info['duration'])-recipe['duration']) > .001:
            raise AssertionError('Output frame count or playback duration mismatch.')
    write(directory / 'contact_sheet.jpg', np.concatenate(gallery, axis=0))
    report = {'status': 'complete', 'recipe': recipe, 'checkpoint_steps': steps, 'metrics': metrics,
              'all_targets_crops_masks_identical': True, 'invariant_fields': list(invariant_fields),
              'frames_per_arm': len(records), 'duration_seconds': recipe['duration'], 'peak_gpu_bytes': peak,
              'elapsed_seconds': time.monotonic()-started,
              'scope': 'Seen training data. Single-frame models, fixed identical seed per frame and per arm; no temporal smoothing/interpolation.',
              'raw_video': 'Raw generated images; not composited with target.',
              'full_frame_video': 'Same edit mask for both models; generated crops pasted back into original frames and resized for display.'}
    save_json(directory / 'manifest.json', report)
    save_json(directory / '_SUCCESS.json', {'recipe_sha256': cache_key(recipe), 'files': {
        name: sha256(directory / name) for name in (*VIDEO_NAMES, 'manifest.json', 'contact_sheet.jpg')}})
    update(status='complete', phase='complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    try:
        worker(args.request)
    except Exception as error:
        save_json(args.request.parent / 'status.json', {'status': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
