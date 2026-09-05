"""Create a frame-aligned source-crop / Gaussian-render comparison video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--render-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=180)
    args = p.parse_args()
    summary = json.loads((args.render_directory / "summary.json").read_text())
    records = [json.loads(x) for x in (args.render_directory / "frames.jsonl").read_text().splitlines()]
    selected = [(i, r) for i, r in enumerate(records) if r["frame_index"] >= args.start_frame][:args.max_frames]
    if not selected:
        raise ValueError("No frames in requested preview interval")
    source = ROOT / "data/HUGG_ARIA_PINHOLE" / summary["sequence"] / "rgb_214_1_pinhole.mp4"
    cap = cv2.VideoCapture(str(source))
    pred = cv2.VideoCapture(str(args.render_directory / "render.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, selected[0][1]["frame_index"])
    pred.set(cv2.CAP_PROP_POS_FRAMES, selected[0][0])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "1024x512", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-threads", "2", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(args.output)], stdin=subprocess.PIPE)
    gallery = []
    chosen = set(np.linspace(0, len(selected)-1, min(6, len(selected))).astype(int).tolist())
    try:
        for offset, (_, record) in enumerate(selected):
            ok, image = cap.read()
            ok_pred, render = pred.read()
            if not ok or not ok_pred:
                raise RuntimeError("Source/render video ended before manifest")
            if record.get("crop_xyxy"):
                x0, y0, x1, y1 = record["crop_xyxy"]
                target = cv2.resize(image[y0:y1, x0:x1], (512, 512))
            else:
                target = cv2.resize(image, (512, 512))
            render = cv2.resize(render, (512, 512))
            panel = np.concatenate((target, render), axis=1)
            cv2.putText(panel, "Source crop", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        .7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(panel, "Gaussian / raw MANO", (524, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        .7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(panel, f"{summary['sequence']} frame {record['frame_index']}",
                        (10, 497), cv2.FONT_HERSHEY_SIMPLEX, .6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            encoder.stdin.write(np.ascontiguousarray(panel).tobytes())
            if offset in chosen:
                gallery.append(cv2.resize(panel, (768, 384)))
    finally:
        cap.release()
        pred.release()
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed with exit code {code}")
    if len(gallery) % 2:
        gallery.append(np.zeros_like(gallery[0]))
    sheet = np.concatenate([np.concatenate(gallery[i:i+2], axis=1)
                            for i in range(0, len(gallery), 2)], axis=0)
    cv2.imwrite(str(args.output.with_suffix(".jpg")), sheet)
    print(json.dumps({"video": str(args.output), "frames": len(selected),
                      "source_start_frame": selected[0][1]["frame_index"]}))


if __name__ == "__main__":
    main()
