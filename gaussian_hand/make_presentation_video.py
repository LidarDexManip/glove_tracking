"""CPU-only five-subject, 10 s/subject demo with each panel rotated clockwise."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate,duration',
        '-of', 'json', str(path)], text=True))['streams'][0]


def rotated_content(frame):
    """Remove the old header, rotate panels individually, keep left/right order."""
    return np.concatenate([
        cv2.rotate(frame[64:, :1024], cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(frame[64:, 1024:], cv2.ROTATE_90_CLOCKWISE)], axis=1)


def read_frame(capture, index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f'Cannot decode frame {index}')
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Refusing to overwrite an existing presentation')
    status = json.loads(args.source.with_suffix('.status.json').read_text())
    assert status['status'] == 'complete' and len(status['segments']) == 5
    assert (status['probe']['width'], status['probe']['height']) == (2048, 1088)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    segments = []
    with tempfile.TemporaryDirectory(prefix='.presentation_', dir=args.output.parent) as temporary:
        work = Path(temporary)
        clips = []
        for ordinal, segment in enumerate(status['segments']):
            # Midpoints were visually inspected. P0014's midpoint is distant;
            # its quarter-point shows both hands manipulating a juice carton.
            center = segment['frames'] // (4 if segment['subject'] == 'P0014' else 2)
            start = center - 150
            assert 0 <= start and start + 300 <= segment['frames']
            absolute_start = segment['output_start_frame'] + start
            seek_seconds, skip_frames = divmod(absolute_start, 30)
            label = f"{segment['subject']} | {segment['sequence']} | {start/30:.1f}-{(start+300)/30:.1f}s"
            filters = (
                f'[0:v]trim=start_frame={skip_frames}:end_frame={skip_frames+300},'
                'setpts=PTS-STARTPTS,split=2[left][right];'
                '[left]crop=1024:1024:0:64,transpose=clock[l];'
                '[right]crop=1024:1024:1024:64,transpose=clock[r];'
                '[l][r]hstack=inputs=2,pad=iw:ih+64:0:64:black,'
                f"drawtext=fontfile={font}:text='Original video':x=16:y=7:fontsize=24:fontcolor=white,"
                f"drawtext=fontfile={font}:text='Gaussian rendering':x=1040:y=7:fontsize=24:fontcolor=white,"
                f"drawtext=fontfile={font}:text='{label}':x=16:y=38:fontsize=18:fontcolor=0xbebebe,"
                'setsar=1[out]')
            clip = work / f'{ordinal:02d}.mp4'
            subprocess.run([
                'ffmpeg', '-n', '-v', 'error', '-threads', '2', '-ss', str(seek_seconds),
                '-i', str(args.source), '-filter_complex_threads', '2', '-filter_complex', filters,
                '-map', '[out]', '-frames:v', '300', '-an', '-c:v', 'libx264', '-threads', '4',
                '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', str(clip)], check=True)
            assert int(probe(clip)['nb_read_frames']) == 300
            clips.append(clip)
            segments.append({'subject': segment['subject'], 'sequence': segment['sequence'],
                             'source_sequence_start_frame': start, 'source_sequence_end_frame_exclusive': start + 300,
                             'long_video_start_frame': absolute_start, 'output_start_frame': ordinal * 300,
                             'frames': 300, 'duration_seconds': 10})
            print(json.dumps(segments[-1]), flush=True)
        concat = work / 'concat.txt'
        concat.write_text(''.join(f"file '{p.name}'\n" for p in clips))
        candidate = work / 'complete.mp4'
        subprocess.run(['ffmpeg', '-n', '-v', 'error', '-f', 'concat', '-safe', '1', '-i', str(concat),
                        '-c', 'copy', '-an', '-movflags', '+faststart', str(candidate)], check=True)
        info = probe(candidate)
        assert int(info['nb_read_frames']) == 1500 and abs(float(info['duration']) - 50) < 1e-5
        assert (info['width'], info['height'], info['r_frame_rate']) == (2048, 1088, '30/1')
        original = cv2.VideoCapture(str(args.source))
        output = cv2.VideoCapture(str(candidate))
        checks, gallery = [], []
        try:
            for segment in segments:
                for offset in (0, 75, 150, 225, 299):
                    before = read_frame(original, segment['long_video_start_frame'] + offset)
                    after = read_frame(output, segment['output_start_frame'] + offset)
                    expected = rotated_content(before)
                    mae = float(np.abs(expected.astype(np.float32) - after[64:].astype(np.float32)).mean())
                    if mae > 5:
                        raise AssertionError(f'Frame alignment/rotation check failed: {segment["subject"]}:{offset}, MAE={mae}')
                    checks.append({'subject': segment['subject'], 'offset': offset, 'reencode_mae_u8': mae})
                    if offset == 150:
                        gallery.append(cv2.resize(after, (1024, 544)))
        finally:
            original.release()
            output.release()
        candidate.rename(args.output)
    cv2.imwrite(str(args.output.with_suffix('.jpg')), np.concatenate(gallery, axis=0))
    result = {'status': 'complete', 'source': str(args.source), 'video': str(args.output),
              'gpu_used': False, 'training_started': False, 'selection': 'middle 10 s; P0014 quarter-point 10 s',
              'rotation': 'each image panel 90 degrees clockwise before horizontal composition; header stays upright',
              'layout': 'left original video; right both Gaussian hands on black; same as full demo',
              'segments': segments, 'probe': info, 'alignment_rotation_checks': checks,
              'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'video': str(args.output), 'probe': info}), flush=True)


if __name__ == '__main__':
    main()
