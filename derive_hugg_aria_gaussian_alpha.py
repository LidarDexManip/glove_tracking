#!/usr/bin/env python3
"""Derive lossless binary alpha videos from aligned black-background Gaussian RGB.

This is the inexpensive fallback for legacy Gaussian renders that did not export
alpha.  RGB frames are decoded by FFmpeg, thresholded a frame at a time with
NumPy, and encoded as lossless gray8 FFV1.  Source RGB videos are never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FORMAT_VERSION = 1
BUFFER_FRAMES = 1


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "data/HUGG_ARIA_GAUSSIANS_ALIGNED",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to input-root, adding alpha.mkv without changing RGB.",
    )
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-sequences", type=int, default=136)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_metadata(path: Path, *, count_packets: bool = True) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_packets:
        command.append("-count_packets")
    command += [
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_read_packets",
        "-of",
        "json",
        str(path),
    ]
    streams = json.loads(subprocess.check_output(command, text=True)).get(
        "streams", []
    )
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}")
    stream = streams[0]
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", maxsplit=1)
    result = {
        "codec": str(stream["codec_name"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
        "fps_fraction": str(stream["avg_frame_rate"]),
    }
    if count_packets:
        result["frames"] = int(stream["nb_read_packets"])
    return result


def metadata_path(output: Path) -> Path:
    return output.with_name("_BLACK_ALPHA.json")


def is_complete(source: Path, output: Path, threshold: int) -> bool:
    record_path = metadata_path(output)
    if not output.is_file() or not record_path.is_file():
        return False
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        source_stat = source.stat()
    except (OSError, ValueError):
        return False
    expected = {
        "format_version": FORMAT_VERSION,
        "threshold": threshold,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
    }
    return all(record.get(key) == value for key, value in expected.items())


def derive_sequence(
    source: Path,
    output: Path,
    threshold: int,
    overwrite: bool,
    max_frames: int | None,
) -> dict:
    sequence = source.parent.name
    if max_frames is None and not overwrite and is_complete(source, output, threshold):
        return {"sequence": sequence, "status": "already_complete"}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial.{os.getpid()}.mkv")
    temporary.unlink(missing_ok=True)
    started = time.perf_counter()
    source_meta = video_metadata(source)
    width = source_meta["width"]
    height = source_meta["height"]
    frame_bytes = width * height * 3

    decode_command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(source),
    ]
    if max_frames is not None:
        decode_command += ["-frames:v", str(max_frames)]
    decode_command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    encode_command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        source_meta["fps_fraction"],
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "gray",
        str(temporary),
    ]

    decoder = subprocess.Popen(decode_command, stdout=subprocess.PIPE)
    encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE)
    frames = 0
    foreground_pixels = 0
    try:
        assert decoder.stdout is not None
        assert encoder.stdin is not None
        while True:
            raw = decoder.stdout.read(frame_bytes * BUFFER_FRAMES)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError(
                    f"Truncated raw frame for {sequence}: {len(raw)} != {frame_bytes}"
                )
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            alpha = np.where(rgb.max(axis=2) > threshold, 255, 0).astype(np.uint8)
            foreground_pixels += int(np.count_nonzero(alpha))
            encoder.stdin.write(alpha.tobytes())
            frames += 1
        encoder.stdin.close()
        decoder_code = decoder.wait()
        encoder_code = encoder.wait()
        if decoder_code != 0:
            raise RuntimeError(f"FFmpeg decoder failed with exit code {decoder_code}")
        if encoder_code != 0:
            raise RuntimeError(f"FFmpeg encoder failed with exit code {encoder_code}")
        expected_frames = (
            min(source_meta["frames"], max_frames)
            if max_frames is not None
            else source_meta["frames"]
        )
        alpha_meta = video_metadata(temporary)
        if frames != expected_frames or alpha_meta["frames"] != expected_frames:
            raise RuntimeError(
                f"Frame mismatch for {sequence}: decoded={frames}, "
                f"alpha={alpha_meta['frames']}, expected={expected_frames}"
            )
        if (alpha_meta["width"], alpha_meta["height"]) != (width, height):
            raise RuntimeError(f"Dimension mismatch for {sequence}")
        source_stat = source.stat()
        record = {
            "format_version": FORMAT_VERSION,
            "sequence": sequence,
            "method": "legacy_black_background_threshold",
            "alpha_semantics": "binary_foreground_support",
            "threshold": threshold,
            "threshold_rule": "foreground_if_max_rgb_gt_threshold",
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "frames": frames,
            "width": width,
            "height": height,
            "fps": source_meta["fps_fraction"],
            "foreground_pixels": foreground_pixels,
            "elapsed_seconds": time.perf_counter() - started,
            "partial_max_frames": max_frames,
        }
        temporary.replace(output)
        metadata_path(output).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "sequence": sequence,
            "status": "converted",
            "frames": frames,
            "elapsed_seconds": record["elapsed_seconds"],
            "bytes": output.stat().st_size,
        }
    except BaseException:
        decoder.kill()
        encoder.kill()
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    args = arguments()
    if not 0 <= args.threshold <= 254:
        raise ValueError("--threshold must be between 0 and 254")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    output_root = args.output_root or args.input_root
    sources = sorted(args.input_root.glob("*/reconstruction.mp4"))
    if args.sequence:
        selected = set(args.sequence)
        sources = [source for source in sources if source.parent.name in selected]
        missing = selected - {source.parent.name for source in sources}
        if missing:
            raise FileNotFoundError(f"Missing sequences: {sorted(missing)}")
    elif len(sources) != args.expected_sequences:
        raise RuntimeError(
            f"Expected {args.expected_sequences} sequences, found {len(sources)}"
        )
    print(
        f"Converting {len(sources)} sequence(s) with threshold={args.threshold}, "
        f"workers={args.workers}, output_root={output_root}",
        flush=True,
    )
    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                derive_sequence,
                source,
                output_root / source.parent.name / "alpha.mkv",
                args.threshold,
                args.overwrite,
                args.max_frames,
            ): source.parent.name
            for source in sources
        }
        for index, future in enumerate(as_completed(futures), start=1):
            sequence = futures[future]
            try:
                result = future.result()
                print(f"[{index}/{len(sources)}] {json.dumps(result)}", flush=True)
            except Exception as error:
                failures.append((sequence, repr(error)))
                print(f"[{index}/{len(sources)}] FAILED {sequence}: {error!r}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} conversion(s) failed: {failures}")


if __name__ == "__main__":
    main()
