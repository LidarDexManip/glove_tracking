"""Expand the existing selected windows to every source frame; no interpolation."""
import argparse
import copy
from fractions import Fraction
import json
from pathlib import Path
import subprocess

from hand_restoration.paired_video import absolute,save_json,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    previous=json.loads(absolute(args.selection).read_text())
    config=json.loads(Path(previous['config']).read_text())
    manifest=absolute(config['data']['train_manifest'])
    assert sha256(manifest)==previous['source_manifest_sha256']
    clips=copy.deepcopy(previous['selected'])
    needed=set()
    for clip in clips:
        source=absolute(config['data']['pinhole_root'])/clip['sequence']/'rgb_214_1_pinhole.mp4'
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=r_frame_rate,avg_frame_rate','-of','json',str(source)],text=True))['streams'][0]
        fps=Fraction(info['avg_frame_rate'])
        if fps!=30:raise ValueError(f'Expected 30fps source, got {fps}: {source}')
        clip['source_video_probe']=info
        clip['records']=[]
        needed.update((clip['sequence'],f,clip['hand']) for f in range(clip['start'],clip['start']+previous['seconds']*30))
    found={}
    for index,line in enumerate(manifest.open()):
        row=json.loads(line);key=(row['sequence_id'],row['frame_index'],row['adaptive_handedness'])
        if key in needed:found[key]=dict(row,global_record_index=index)
    if needed-found.keys():raise ValueError(f'Missing source frames: {len(needed-found.keys())}')
    for clip in clips:
        for frame in range(clip['start'],clip['start']+previous['seconds']*30):
            row=found[(clip['sequence'],frame,clip['hand'])]
            row['original_adaptive_crop_box_xyxy']=row['adaptive_crop_box_xyxy']
            row['adaptive_crop_box_xyxy']=clip['fixed_crop_box']
            clip['records'].append(row)
    output=absolute(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    save_json(output,dict(config=previous['config'],config_sha256=previous['config_sha256'],
        source_manifest_sha256=previous['source_manifest_sha256'],source_selection_sha256=sha256(absolute(args.selection)),
        source_fps=30,fps=30,seconds=previous['seconds'],selected=clips,
        frame_policy='All consecutive source frames; same windows, subjects, hands and fixed crop as v1; no interpolation or repeats.'))
    print(output,'clips',len(clips),'frames',sum(len(c['records']) for c in clips))


if __name__=='__main__':main()
