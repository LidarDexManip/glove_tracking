"""Preview actual diffusion target/condition crops, CPU only, no model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

import cv2
import numpy as np
import torch

from gaussian_hand.make_presentation_video import probe
from train_hand_restorer import make_hugg_aria_dataset

ROOT = Path(__file__).resolve().parents[1]
SELECTION = [
    ('P0001_f6cc0cc8', 1018, 'right'),
    ('P0002_65085bfc', 2553, 'left'),
    ('P0009_0e02bf39', 1453, 'both'),
    ('P0012_0323d770', 2262, 'both'),
    ('P0014_24cb3bf0', 997, 'both'),
]


def u8(image):
    return (image[..., ::-1] * 255).round().clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'outputs/hand_restoration/hugg_aria_gaussian_surface_v2_4gpu/config.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    dataset = make_hugg_aria_dataset(config, ROOT / config['data']['train_manifest'], include_numpy=True)
    lookup = {}
    for index, row in enumerate(dataset.samples):
        key = (row['sequence_id'], row['frame_index'], row['adaptive_handedness'])
        if key in lookup:
            raise ValueError(f'Ambiguous training crop identity: {key}')
        lookup[key] = index
    selections = []
    for sequence, start, side in SELECTION:
        indices = [lookup[(sequence, frame, side)] for frame in range(start, start + 150)]
        selections.append((sequence, start, side, indices))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gallery, references, segments = [], {}, []
    with tempfile.TemporaryDirectory(prefix='.training_demo_', dir=args.output.parent) as directory:
        candidate = Path(directory) / 'video.mp4'
        encoder = subprocess.Popen([
            'ffmpeg', '-n', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1024x576',
            '-r', '30', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'veryfast',
            '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(candidate)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        count = 0
        try:
            for sequence, start, side, indices in selections:
                modes = set()
                archive_path = ROOT / 'outputs/gaussian_hand/diffusion_strict_v2/image_sequences' / sequence / 'images.tar'
                with tarfile.open(archive_path) as archive:
                    for offset, index in enumerate(indices):
                        row = dataset.samples[index]
                        sample = dataset[index]
                        target, condition = u8(sample['target_rgb_np']), u8(sample['condition_rgb_np'])
                        stored = archive.extractfile(f'{index:07d}.condition.png').read()
                        decoded = cv2.imdecode(np.frombuffer(stored, np.uint8), cv2.IMREAD_COLOR)
                        if not np.array_equal(condition, decoded):
                            raise AssertionError(f'Archived/live condition mismatch: {sequence}:{start+offset}')
                        panel = np.zeros((576, 1024, 3), np.uint8)
                        panel[64:, :512] = cv2.rotate(target, cv2.ROTATE_90_CLOCKWISE)
                        panel[64:, 512:] = cv2.rotate(condition, cv2.ROTATE_90_CLOCKWISE)
                        cv2.putText(panel, 'Original target crop', (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    .60, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(panel, 'Gaussian training condition', (524, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    .60, (255, 255, 255), 1, cv2.LINE_AA)
                        mode = row['adaptive_crop_mode']; modes.add(mode)
                        label = f'{sequence} | {side} | frame {start+offset} | {mode}'
                        cv2.putText(panel, label, (12, 51), cv2.FONT_HERSHEY_SIMPLEX,
                                    .46, (190, 190, 190), 1, cv2.LINE_AA)
                        encoder.stdin.write(panel.tobytes())
                        if offset in (0, 75, 149):
                            references[count] = panel[64:].copy()
                        if offset == 75:
                            gallery.append(panel.copy())
                        count += 1
                segments.append({'sequence': sequence, 'subject': sequence.split('_')[0], 'handedness': side,
                    'source_start_frame': start, 'source_end_frame_exclusive': start + 150,
                    'output_start_frame': count - 150, 'frames': 150, 'seconds': 5,
                    'crop_modes': sorted(modes), 'training_manifest_indices': indices})
                print(json.dumps({k: v for k, v in segments[-1].items() if k != 'training_manifest_indices'}), flush=True)
        finally:
            encoder.stdin.close()
            error = encoder.stderr.read().decode()
            code = encoder.wait()
            if code:
                raise RuntimeError(error)
        info = probe(candidate)
        assert count == 750 and int(info['nb_read_frames']) == count and float(info['duration']) == 25
        assert (info['width'], info['height'], info['r_frame_rate']) == (1024, 576, '30/1')
        capture = cv2.VideoCapture(str(candidate))
        checks = []
        try:
            for index, expected in references.items():
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, actual = capture.read()
                assert ok
                mae = float(np.abs(expected.astype(np.float32) - actual[64:].astype(np.float32)).mean())
                assert mae < 5, (index, mae)
                checks.append({'output_frame': index, 'reencode_mae_u8': mae})
        finally:
            capture.release()
        candidate.rename(args.output)
    cv2.imwrite(str(args.output.with_suffix('.jpg')), np.concatenate(gallery, axis=0))
    report = {'status': 'complete', 'video': str(args.output), 'config': str(args.config),
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'train_manifest_sha256': hashlib.sha256((ROOT / config['data']['train_manifest']).read_bytes()).hexdigest(),
        'probe': info, 'segments': segments, 'gpu_used': False, 'diffusion_inference': False,
        'rotation': '90 degrees clockwise per crop panel; upright titles and unchanged left/right layout',
        'meaning': 'Actual training target vs actual online Gaussian condition, including original crop/mask/wrist rules; not a trained restorer output',
        'continuous_source_frames_per_sequence': True, 'archived_condition_exact_matches': count,
        'post_encode_alignment_checks': checks,
        'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'video': str(args.output), 'probe': info}), flush=True)


if __name__ == '__main__':
    main()
