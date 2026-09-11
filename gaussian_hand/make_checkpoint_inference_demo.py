"""Five fixed training-frame inference previews; does not modify training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from gaussian_hand.make_presentation_video import probe
from hand_restoration.inference import build_restorer, run_inference, save_rgb
from train_hand_restorer import make_hugg_aria_dataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.10)
    started = time.monotonic()
    checkpoint_stat = args.checkpoint.stat()
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    step = int(payload['global_step'])
    assert payload['state_dict'] and step > 0
    if args.checkpoint.stat().st_size != checkpoint_stat.st_size or args.checkpoint.stat().st_mtime_ns != checkpoint_stat.st_mtime_ns:
        raise RuntimeError('Checkpoint changed while validating')
    del payload
    config_path = args.checkpoint.parent / 'config.json'
    config = json.loads(config_path.read_text())
    data_demo = ROOT / 'outputs/gaussian_hand/result_videos/diffusion_training_data_five_subjects_5s_clockwise90.json'
    selection = json.loads(data_demo.read_text())
    assert selection['status'] == 'complete'
    dataset = make_hugg_aria_dataset(config, ROOT / config['data']['train_manifest'], include_numpy=True)
    restorer = build_restorer(config, device='cuda')
    restorer.load_controlnet(args.checkpoint)
    for module in (restorer.vae, restorer.text_encoder, restorer.unet, restorer.controlnet):
        module.eval()
    args.output.mkdir(parents=True)
    panels, rows = [], []
    for ordinal, segment in enumerate(selection['segments']):
        index = segment['training_manifest_indices'][75]
        record = dataset.samples[index]
        assert record['sequence_id'] == segment['sequence']
        assert record['frame_index'] == segment['source_start_frame'] + 75
        sample = dataset[index]
        t = time.monotonic()
        result = run_inference(restorer, sample, config, steps=20, seed=7)
        torch.cuda.synchronize()
        assert np.isfinite(result.generated).all()
        folder = args.output / f'{ordinal:02d}_{segment["subject"]}'
        for name, image in [('condition', sample['condition_rgb_np']), ('raw_model_output', result.generated),
                            ('target', sample['target_rgb_np']), ('restored_crop', result.restored)]:
            save_rgb(folder / f'{name}.png', image)
        if result.restored_full is not None:
            save_rgb(folder / 'restored_full.png', result.restored_full)
        panel = np.zeros((576, 1536, 3), np.uint8)
        for column, (name, image) in enumerate([
                ('Gaussian condition', sample['condition_rgb_np']),
                ('Raw diffusion output', result.generated), ('Original target', sample['target_rgb_np'])]):
            pixels = (image[..., ::-1] * 255).round().clip(0, 255).astype(np.uint8)
            panel[64:, column*512:(column+1)*512] = cv2.rotate(pixels, cv2.ROTATE_90_CLOCKWISE)
            cv2.putText(panel, name, (column*512+12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (255, 255, 255), 1, cv2.LINE_AA)
        label = f"step {step} | {record['sequence_id']} | frame {record['frame_index']} | 20 steps | seed 7 | still preview"
        cv2.putText(panel, label, (12, 51), cv2.FONT_HERSHEY_SIMPLEX, .48, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.imwrite(str(folder / 'comparison.png'), panel)
        panels.append(panel)
        rows.append({'subject': segment['subject'], 'sequence': record['sequence_id'], 'frame_index': record['frame_index'],
                     'training_manifest_index': index, 'handedness': record['adaptive_handedness'],
                     'inference_seconds': time.monotonic()-t,
                     'raw_psnr_full': result.generated_psnr_full, 'raw_psnr_edit_mask': result.generated_psnr_masked})
        print(json.dumps(rows[-1]), flush=True)
    peak_gpu_bytes = torch.cuda.max_memory_allocated()
    del restorer
    torch.cuda.empty_cache()
    cv2.imwrite(str(args.output / 'contact_sheet.jpg'), np.concatenate(panels, axis=0))
    video = args.output / 'five_subjects_inference_preview.mp4'
    encoder = subprocess.Popen([
        'ffmpeg', '-n', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1536x576', '-r', '30',
        '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for panel in panels:
            for _ in range(90):
                encoder.stdin.write(panel.tobytes())
    finally:
        encoder.stdin.close()
        error = encoder.stderr.read().decode()
        code = encoder.wait()
        if code:
            raise RuntimeError(error)
    info = probe(video)
    assert int(info['nb_read_frames']) == 450 and float(info['duration']) == 15
    report = {'status': 'complete', 'checkpoint': str(args.checkpoint), 'checkpoint_sha256': checkpoint_sha,
              'checkpoint_step': step, 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'sample_selection': 'Fixed middle frame of each of the preceding five training-data demo segments',
              'selection_manifest_sha256': hashlib.sha256(data_demo.read_bytes()).hexdigest(),
              'scope': 'Five seen training-frame previews, not held-out evaluation or a baseline A/B result',
              'inference_steps': 20, 'seed': 7, 'guidance_scale': 1.0, 'controlnet_scale': 1.0,
              'precision': 'float32 (existing inference path)', 'gpu_physical_index': 0,
              'peak_allocated_gpu_bytes': peak_gpu_bytes, 'elapsed_seconds': time.monotonic()-started,
              'presentation': 'Five still results, three seconds each; not continuous motion. Each panel rotated clockwise90.',
              'center_panel': 'Raw generated output, NOT target paste-back. Separate restored images also saved.',
              'video': str(video), 'probe': info, 'samples': rows}
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}), flush=True)


if __name__ == '__main__':
    main()
