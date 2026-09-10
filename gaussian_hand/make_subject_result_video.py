"""CPU-only original/full-frame Gaussian comparison from existing crop renders."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def restore_render(rgb, alpha, shape, box):
    """Invert the recorded crop; RGB is already premultiplied, not an overlay."""
    canvas = np.zeros(shape, np.uint8)
    x0,y0,x1,y1 = box
    if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
        raise ValueError("Crop lies outside the source image")
    clean = rgb.copy()
    clean[alpha == 0] = 0  # suppress lossy RGB ringing outside lossless alpha
    resized = cv2.resize(clean,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
    support = cv2.resize(alpha,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
    resized[support == 0] = 0
    canvas[y0:y1,x0:x1] = resized
    return canvas


def fit_panel(image, size):
    scale = min(size/image.shape[1],size/image.shape[0])
    w,h = round(image.shape[1]*scale),round(image.shape[0]*scale)
    panel = np.zeros((size,size,3),np.uint8)
    x,y = (size-w)//2,(size-h)//2
    panel[y:y+h,x:x+w] = cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
    return panel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root",type=Path,default=ROOT/"outputs/gaussian_hand/train_mano_surface_v2_30k")
    parser.add_argument("--hand",action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--panel-size",type=int,default=1024)
    args=parser.parse_args();cv2.setNumThreads(1)
    if args.output.exists():raise FileExistsError(args.output)
    if len({k.split('_')[0] for k in args.hand}) != len(args.hand):
        raise ValueError("Use one sequence/hand per subject")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    size,header,fps=args.panel_size,64,30
    encoder=subprocess.Popen(["ffmpeg","-n","-v","error","-f","rawvideo","-pix_fmt","bgr24",
        "-s",f"{size*2}x{size+header}","-r",str(fps),"-i","-","-an","-c:v","libx264",
        "-threads","4","-preset","fast","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",
        str(args.output)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    segments=[];gallery=[];total=0
    try:
        for key in args.hand:
            directory=args.model_root/key/"preview_render"
            summary=json.loads((directory/"summary.json").read_text())
            records=[json.loads(x) for x in (directory/"frames.jsonl").read_text().splitlines()]
            source=ROOT/"data/HUGG_ARIA_PINHOLE"/summary["sequence"]
            with (source/"frame_timestamps_214_1.csv").open() as f:
                timestamps=[int(r["timestamp_ns"]) for r in csv.DictReader(f)]
            captures=[cv2.VideoCapture(str(p)) for p in
                      (source/"rgb_214_1_pinhole.mp4",directory/"render.mp4",directory/"alpha.mkv")]
            if not all(c.isOpened() for c in captures):raise RuntimeError("Cannot open input video")
            captures[0].set(cv2.CAP_PROP_POS_FRAMES,records[0]["frame_index"])
            begin=total
            try:
                for index,record in enumerate(records):
                    if record["frame_index"] != records[0]["frame_index"]+index:
                        raise ValueError("Noncontiguous render frames")
                    if timestamps[record["frame_index"]] != record["timestamp_ns"]:
                        raise ValueError("Timestamp alignment mismatch")
                    decoded=[c.read() for c in captures]
                    if not all(ok for ok,_ in decoded):raise RuntimeError("Video ended before manifest")
                    original,rgb,alpha=[im for _,im in decoded]
                    gaussian=restore_render(rgb,alpha[...,0],original.shape,record["crop_xyxy"])
                    frame=np.zeros((size+header,size*2,3),np.uint8)
                    frame[header:,:size]=fit_panel(original,size)
                    frame[header:,size:]=fit_panel(gaussian,size)
                    title=f"{key.split('_')[0]} | {summary['sequence']} | {key.split('_')[1]} hand"
                    cv2.putText(frame,"Original video",(16,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(frame,"Gaussian rendering",(size+16,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(frame,title,(16,51),cv2.FONT_HERSHEY_SIMPLEX,.52,(180,180,180),1,cv2.LINE_AA)
                    encoder.stdin.write(frame.tobytes())
                    if index==len(records)//2:gallery.append(cv2.resize(frame,(1024,(size+header)//2)))
                    total+=1
            finally:
                for capture in captures:capture.release()
            segments.append({"subject":key.split('_')[0],"hand":key.split('_')[1],"sequence":summary["sequence"],
                "output_start_frame":begin,"frames":len(records),"source_start_frame":records[0]["frame_index"],
                "source_end_frame":records[-1]["frame_index"],"checkpoint":summary["checkpoint"],
                "render_directory":str(directory)})
    finally:
        encoder.stdin.close();error=encoder.stderr.read().decode();code=encoder.wait()
        if code:raise RuntimeError(error)
    info=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
        "-show_entries","stream=width,height,nb_read_frames,duration","-of","json",str(args.output)],text=True))["streams"][0]
    if int(info["nb_read_frames"])!=total:raise RuntimeError("Output frame count mismatch")
    if not cv2.imwrite(str(args.output.with_suffix('.jpg')),np.concatenate(gallery)):
        raise RuntimeError("Contact sheet write failed")
    result={"video":str(args.output),"frames":total,"fps":fps,"duration_seconds":total/fps,
            "layout":"left: original full frame; right: Gaussian on black in the same source coordinates",
            "scope":"Existing short single-hand excerpts, not full sequences; no new GPU rendering",
            "segments":segments,"probe":info,"sha256":hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
