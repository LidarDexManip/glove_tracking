#!/usr/bin/env python3
"""Render existing HUGG Aria Gaussian models to aligned RGB and alpha videos.

This is deliberately render-only: it reuses each sequence's trained
``hand_gaussians.npz`` and reconstructs only the annotated MANO motion and
camera track needed to pose it.  It never repeats appearance baking.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEQUENCE_RE = re.compile(r"^P[0-9]+_[0-9a-f]+$")
FORMAT_VERSION = 1


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--renderer-root",
        type=Path,
        default=ROOT / "external/bare-hand_gaussian_reconstruction",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "data/HUGG_ARIA_GAUSSIANS",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data/raw/hot3d_aria",
        help="Use a local raw sequence when present; otherwise download it.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mano-pkl",
        type=Path,
        default=ROOT / "mano_v1_2/models/MANO_RIGHT.pkl",
    )
    parser.add_argument("--hf-repo", default="LIDAR-GT/HUGG_ARIA")
    parser.add_argument("--expected-sequences", type=int, default=136)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Benchmark only; requires --allow-partial.",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def renderer_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def video_metadata(path: Path) -> dict:
    payload = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_read_packets",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    streams = json.loads(payload).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}")
    stream = streams[0]
    return {
        "codec": str(stream["codec_name"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_packets"]),
    }


def is_complete(
    output: Path,
    *,
    model_digest: str,
    commit: str,
    supersample: int,
    partial_frames: int | None,
) -> bool:
    success = output / "_SUCCESS.json"
    if not success.is_file():
        return False
    try:
        metadata = json.loads(success.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected = {
        "format_version": FORMAT_VERSION,
        "model_sha256": model_digest,
        "renderer_commit": commit,
        "supersample": supersample,
        "partial_frames": partial_frames,
    }
    return (
        all(metadata.get(key) == value for key, value in expected.items())
        and (output / "reconstruction.mp4").is_file()
        and (output / "alpha.mkv").is_file()
    )


def free_memory(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def render_sequence(
    sequence: str,
    args: argparse.Namespace,
    *,
    commit: str,
    api: dict,
) -> dict:
    model_path = args.model_root / sequence / "hand_gaussians.npz"
    model_digest = sha256(model_path)
    final = args.output_root / sequence
    if is_complete(
        final,
        model_digest=model_digest,
        commit=commit,
        supersample=args.supersample,
        partial_frames=args.max_frames,
    ) and not args.overwrite:
        return {"sequence": sequence, "status": "already_complete"}
    if final.exists() and not args.overwrite:
        raise FileExistsError(f"Incomplete or stale output exists: {final}")

    local_source = args.raw_root / sequence
    source = str(local_source if local_source.is_dir() else sequence)
    source_kind = "local_raw" if local_source.is_dir() else "huggingface"
    started = time.perf_counter()
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{sequence}.partial.", dir=args.output_root)
    )
    torch = api["torch"]
    cuda_device = torch.device(args.device)
    try:
        loaded = api["load_hot3d_sequence"](
            source,
            hand="both",
            max_frames=args.max_frames,
            stride=1,
            mano_pkl=str(args.mano_pkl),
            hf_repo=args.hf_repo,
        )
        fits = [api["build_hot3d_fit"](loaded, hand) for hand in loaded.manos()]
        fit = api["merge_fits"](fits)
        model = api["load_npz"](str(model_path))
        if int(model.bind_faces.max()) >= int(fit.verts.shape[1]):
            raise RuntimeError(
                f"Gaussian binding exceeds posed mesh: "
                f"{int(model.bind_faces.max())} >= {fit.verts.shape[1]}"
            )
        total_frames = len(fit.verts)
        valid_frames = int(fit.valid.sum())
        rgb_path = temporary / "reconstruction.mp4"
        alpha_path = temporary / "alpha.mkv"
        frame_iterator = api["reconstruction_frames"](
            model,
            fit,
            loaded.frames,
            loaded.cameras,
            max_frames=None,
            supersample=args.supersample,
            device=args.device,
        )
        api["write_rgb_alpha_video_stream"](
            str(rgb_path), str(alpha_path), frame_iterator, fps=loaded.fps
        )
        rgb_metadata = video_metadata(rgb_path)
        alpha_metadata = video_metadata(alpha_path)
        if rgb_metadata["frames"] != total_frames:
            raise RuntimeError(
                f"RGB frame mismatch: {rgb_metadata['frames']} != {total_frames}"
            )
        if alpha_metadata["frames"] != total_frames:
            raise RuntimeError(
                f"Alpha frame mismatch: {alpha_metadata['frames']} != {total_frames}"
            )
        if (
            rgb_metadata["width"],
            rgb_metadata["height"],
        ) != (
            alpha_metadata["width"],
            alpha_metadata["height"],
        ):
            raise RuntimeError("RGB and alpha dimensions differ")
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(cuda_device))
            if torch.cuda.is_available()
            else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved(cuda_device))
            if torch.cuda.is_available()
            else 0
        )
        metadata = {
            "format_version": FORMAT_VERSION,
            "sequence": sequence,
            "render_kind": "gaussian",
            "frame_alignment": "output_frame_index_equals_source_frame_index",
            "rgb_semantics": "straight_rgb; composite_with_alpha",
            "alpha_semantics": "continuous_gaussian_coverage",
            "alpha_codec": "ffv1_lossless_gray8",
            "source_kind": source_kind,
            "source": source,
            "model_path": str(model_path.resolve()),
            "model_sha256": model_digest,
            "renderer_root": str(args.renderer_root.resolve()),
            "renderer_commit": commit,
            "total_frames": total_frames,
            "valid_mano_frames": valid_frames,
            "blank_frames": total_frames - valid_frames,
            "partial_frames": args.max_frames,
            "supersample": args.supersample,
            "device": args.device,
            "fps": float(loaded.fps),
            "rgb": rgb_metadata,
            "alpha": alpha_metadata,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (temporary / "_SUCCESS.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        if final.exists():
            shutil.rmtree(final)
        temporary.replace(final)
        return {"status": "rendered", **metadata}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        for name in ("frame_iterator", "model", "fit", "fits", "loaded"):
            if name in locals():
                del locals()[name]
        free_memory(torch)


def main() -> None:
    args = arguments()
    args.renderer_root = args.renderer_root.resolve()
    args.model_root = args.model_root.resolve()
    args.raw_root = args.raw_root.resolve()
    args.output_root = args.output_root.resolve()
    args.mano_pkl = args.mano_pkl.resolve()
    if args.max_frames is not None and not args.allow_partial:
        raise ValueError("--max-frames requires --allow-partial")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if not args.mano_pkl.is_file():
        raise FileNotFoundError(args.mano_pkl)
    if not (args.renderer_root / "glove2hand").is_dir():
        raise FileNotFoundError(args.renderer_root / "glove2hand")

    all_sequences = sorted(
        path.name
        for path in args.model_root.iterdir()
        if path.is_dir()
        and SEQUENCE_RE.fullmatch(path.name)
        and (path / "hand_gaussians.npz").is_file()
    )
    if len(all_sequences) != args.expected_sequences:
        raise RuntimeError(
            f"Expected {args.expected_sequences} Gaussian models, "
            f"found {len(all_sequences)}"
        )
    if args.sequence:
        requested = set(args.sequence)
        missing = sorted(requested - set(all_sequences))
        if missing:
            raise KeyError(f"Unknown Gaussian sequences: {missing}")
        sequences = [value for value in all_sequences if value in requested]
    else:
        sequences = all_sequences
    sequences = [
        value
        for index, value in enumerate(sequences)
        if index % args.num_shards == args.shard_index
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.renderer_root))
    from glove2hand.gaussians.io import load_npz
    from glove2hand.io.hot3d import build_hot3d_fit, load_hot3d_sequence
    from glove2hand.io.video import write_rgb_alpha_video_stream
    from glove2hand.pipeline import _merge_fits
    from glove2hand.utils.deps import import_torch
    from glove2hand.viewer.render_views import reconstruction_frames

    torch = import_torch()
    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    api = {
        "torch": torch,
        "load_npz": load_npz,
        "build_hot3d_fit": build_hot3d_fit,
        "load_hot3d_sequence": load_hot3d_sequence,
        "merge_fits": _merge_fits,
        "write_rgb_alpha_video_stream": write_rgb_alpha_video_stream,
        "reconstruction_frames": reconstruction_frames,
    }
    commit = renderer_commit(args.renderer_root)
    print(
        json.dumps(
            {
                "event": "start",
                "renderer_commit": commit,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "sequence_count": len(sequences),
                "device": args.device,
                "torch_threads": args.torch_threads,
                "output_root": str(args.output_root),
            }
        ),
        flush=True,
    )
    failures = []
    for index, sequence in enumerate(sequences, 1):
        try:
            result = render_sequence(sequence, args, commit=commit, api=api)
            result["shard_position"] = f"{index}/{len(sequences)}"
            print(json.dumps(result), flush=True)
        except Exception as exc:
            failure = {
                "sequence": sequence,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "shard_position": f"{index}/{len(sequences)}",
            }
            failures.append(failure)
            print(json.dumps(failure), flush=True)
    summary = {
        "event": "complete",
        "shard_index": args.shard_index,
        "sequence_count": len(sequences),
        "failures": failures,
    }
    print(json.dumps(summary), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
