"""Export a clean curated 30-FPS showcase; optionally retire audited 10-FPS demos."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess

from hand_restoration.paired_video import absolute,save_json,sha256

CLIPS=['P0014_9b7a0725_left_001790','P0003_f4730606_right_001847',
       'P0010_573b2de6_left_000979','P0015_1eb4f17f_left_000374']
ROOT_NAMES=('animatediff_demo_v1','animatediff_demo_v2_30fps','animatediff_demo_v3_stronger')


def probe(path,count=False):
    command=['ffprobe','-v','error','-select_streams','v:0']
    if count:command+=['-count_frames']
    command+=['-show_entries','stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames,duration',
              '-of','json',str(path)]
    return json.loads(subprocess.check_output(command,text=True))['streams'][0]


def inventory(root):
    paths=[]
    for name in ROOT_NAMES:
        base=root/name
        for folder,dirs,files in os.walk(base,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
            for file in files:
                p=Path(folder)/file
                if p.suffix!='.mp4' or p.is_symlink():continue
                if not p.resolve().is_relative_to(base):raise ValueError('Path escaped demo root')
                paths.append(p)
    def inspect(p):
        return dict(path=str(p),fps=str(Fraction(probe(p)['avg_frame_rate'])),
                    sha256=sha256(p),bytes=p.stat().st_size)
    return list(ThreadPoolExecutor(max_workers=4).map(inspect,paths))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--retire-low-fps',action='store_true')
    args=parser.parse_args()
    root=absolute('outputs/hand_restoration')
    output=absolute(args.output)
    if not output.is_relative_to(root):raise ValueError('Delivery must stay in hand_restoration outputs')
    audit=inventory(root)
    if any(r['fps'] not in ('10','30') for r in audit):raise ValueError('Unexpected source frame rate')
    source=root/'animatediff_demo_v3_stronger/videos'
    inputs={kind:[source/clip/f'animatediff_{kind}.mp4' for clip in CLIPS] for kind in ('crop','full')}
    for paths in inputs.values():
        for p in paths:
            info=probe(p,True)
            if info['avg_frame_rate']!='30/1' or int(info['nb_read_frames'])!=150:
                raise ValueError(f'Invalid showcase input: {p}')
    output.mkdir(parents=True,exist_ok=False)
    metadata=output/'metadata';metadata.mkdir()
    individual=output/'clips';individual.mkdir()
    results=[]
    for kind,paths in inputs.items():
        listing=metadata/f'{kind}_concat.txt'
        listing.write_text(''.join(f"file '{p}'\n" for p in paths))
        name='hand_demo_30fps.mp4' if kind=='crop' else 'hand_demo_full_view_30fps.mp4'
        target=output/name
        subprocess.run(['ffmpeg','-n','-v','error','-f','concat','-safe','0','-i',str(listing),
            '-map','0:v:0','-an','-c:v','copy','-movflags','+faststart',str(target)],check=True)
        info=probe(target,True)
        assert info['avg_frame_rate']==info['r_frame_rate']=='30/1'
        assert int(info['nb_read_frames'])==600 and abs(float(info['duration'])-20)<.002
        results.append(dict(file=name,**info))
        for clip,p in zip(CLIPS,paths):
            shutil.copy2(p,individual/f'{clip}_{kind}_30fps.mp4')
    save_json(metadata/'delivery.json',dict(selected=CLIPS,outputs=results,source=str(source),
        note='Curated visual showcase, not an evaluation. Four complete five-second clips, original 30 FPS. '
             'No metrics, titles, interpolation, slow motion, web page, or extra model inference. '
             'Crop presentation and original-view composites are supplied separately. Video packets copied without re-encoding.'))
    if args.retire_low_fps:
        retired=[r for r in audit if r['fps']=='10']
        retained=[r for r in audit if r['fps']=='30']
        trash=root/'.trash'/('low_fps_demos_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        trash.mkdir(parents=True,exist_ok=False)
        for r in retired:r['recovery_path']=str(trash/Path(r['path']).relative_to(root))
        save_json(trash/'manifest.json',dict(files=retired,note='Moved, not permanently deleted. Restore each recovery_path to path.'))
        for r in retired:
            p=Path(r['path']);destination=Path(r['recovery_path'])
            if p.is_symlink() or sha256(p)!=r['sha256'] or Fraction(probe(p)['avg_frame_rate'])!=10:
                raise ValueError('Retirement target changed after audit')
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(p),str(destination))
            assert sha256(destination)==r['sha256'] and not p.exists()
        for r in retained:assert sha256(Path(r['path']))==r['sha256']
        save_json(metadata/'cleanup.json',dict(retired_count=len(retired),retained_30fps=len(retained),
            recovery_directory=str(trash),retired_bytes=sum(r['bytes'] for r in retired)))
        print('RETIRED',len(retired),'10-FPS videos; preserved',len(retained),'30-FPS videos; recoverable at',trash,flush=True)
    print('EXPORTED',output,json.dumps(results),flush=True)


if __name__=='__main__':main()
