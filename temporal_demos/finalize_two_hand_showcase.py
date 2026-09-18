"""Assemble only visually approved, fully rendered two-hand clips on CPU."""
import argparse
import json
from pathlib import Path
import subprocess

from hand_restoration.paired_video import absolute,save_json
from temporal_demos.export_showcase import probe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--clip',action='append',required=True)
    args=parser.parse_args();root=absolute(args.directory);metadata=root/'metadata'
    completed={r['id']:r for r in json.loads((metadata/'completed.json').read_text())}
    selected=[completed[name] for name in args.clip]
    for row in selected:
        if min(row['minimum_changed_pixels_per_hand'])<100:raise ValueError('Missing hand restoration')
        info=probe(Path(row['video']),True)
        assert int(info['nb_read_frames'])==150 and info['avg_frame_rate']=='30/1'
    listing=metadata/'curated_concat.txt';listing.write_text(''.join(f"file '{r['video']}'\n" for r in selected))
    destination=root/'two_hand_demo_30fps.mp4'
    subprocess.run(['ffmpeg','-n','-v','error','-f','concat','-safe','0','-i',str(listing),
        '-map','0:v:0','-an','-c:v','copy','-movflags','+faststart',str(destination)],check=True)
    info=probe(destination,True)
    assert int(info['nb_read_frames'])==150*len(selected) and info['avg_frame_rate']==info['r_frame_rate']=='30/1'
    save_json(metadata/'summary.json',dict(clips=selected,frames=150*len(selected),fps=30,
        duration_seconds=5*len(selected),extra_training=False,labels=['Gaussian-conditioned input','Restoration'],
        curation='Retained complete, visible two-hand scenes; excluded P0012 candidate with hand near frame edge.',
        outside_both_hand_masks_unchanged=True,video=str(destination)))
    print('SHOWCASE',destination,info,flush=True)


if __name__=='__main__':main()
