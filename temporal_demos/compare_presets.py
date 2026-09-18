"""CPU-only, synchronized old/new preset comparisons for identical source frames."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hand_restoration.paired_video import sha256,save_json
from temporal_demos.render_clips import video


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--current',type=Path,required=True)
    args=parser.parse_args();old=args.previous.resolve();new=args.current.resolve()
    a=json.loads((old/'provenance.json').read_text())
    b=json.loads((new/'provenance.json').read_text())
    for key in ('checkpoint_sha256','selection_sha256','fps','seed','steps','animatediff_scheduler','postprocess'):
        if a[key]!=b[key]:raise ValueError(f'Unmatched comparison setting: {key}')
    if sha256(old/'frames.jsonl')!=sha256(new/'frames.jsonl'):
        raise ValueError('Source frame manifests differ')
    old_clips={r['id']:r for r in json.loads((old/'summary.json').read_text())['clips']}
    clips=json.loads((new/'summary.json').read_text())['clips']
    if set(old_clips)!={r['id'] for r in clips}:raise ValueError('Clip IDs differ')
    paths=[new/r['id']/'preset_comparison.mp4' for r in clips]+[new/'preset_comparison.json']
    if any(p.exists() for p in paths):raise FileExistsError('Preset comparison already exists')
    labels=[f"{name}: s={p['motion_strength']} / context={p['context']} / overlap={p['overlap']}"
            for name,p in [('Previous',a),('New',b)]]
    reports=[];cv2.setNumThreads(1)
    for clip in clips:
        captures=[cv2.VideoCapture(str(root/clip['id']/'animatediff_crop.mp4')) for root in (old,new)]
        frames=[]
        try:
            for i in range(clip['frames']):
                panel=np.zeros((576,1024,3),np.uint8)
                for j,cap in enumerate(captures):
                    ok,im=cap.read()
                    if not ok or im.shape!=(512,512,3):raise ValueError('Missing or mismatched video frame')
                    panel[64:,j*512:(j+1)*512]=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                    cv2.putText(panel,labels[j],(j*512+8,25),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(panel,f"{clip['id']} | source {clip['source_start']+i} | {b['fps']} FPS | same brightness filter",
                            (8,51),cv2.FONT_HERSHEY_SIMPLEX,.45,(210,210,210),1,cv2.LINE_AA)
                frames.append(panel)
            if any(cap.read()[0] for cap in captures):raise ValueError('Extra video frames')
        finally:
            for cap in captures:cap.release()
        video(new/clip['id']/'preset_comparison.mp4',frames,b['fps'])
        reports.append(dict(id=clip['id'],previous=old_clips[clip['id']]['metrics']['animatediff'],
                            current=clip['metrics']['animatediff']))
        print('COMPARED',clip['id'],flush=True)
    save_json(new/'preset_comparison.json',dict(previous_directory=str(old),labels=labels,
        frames_sha256=sha256(new/'frames.jsonl'),clips=reports,
        note='Same frames/checkpoint/postprocessing; motion strength, context and overlap changed together.'))


if __name__=='__main__':main()
