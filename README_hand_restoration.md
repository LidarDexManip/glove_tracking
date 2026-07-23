# HOT3D Single-Frame Hand Restoration MVP

This module validates the first part of the Glove2Hand idea: an aligned MANO
mesh is rendered into the same canonical C1 camera as a HOT3D frame, used to
corrupt the hand region, and supplied as a three-channel ControlNet condition
for realistic bare-hand restoration. It is deliberately **not** an exact
Glove2Hand reproduction: it uses MANO instead of Gaussian Hand and has no
AnimateDiff or video training.

## Existing code reused

- `hot3d/hot3d/clips/clip_util.py`: clip tar images, annotations, and cameras.
- `export_hot3d_clip_undistorted.py`: established C1 calibration and the
  required `-90` degree upright target-camera roll.
- `hot3d_glove_torch_utils.py`: SMPL-X MANO loading and the HOT3D left-hand
  shape-direction fix.
- `hand_tracking_toolkit.rasterizer`: aligned MANO mesh mask/rendering.

`Hot3DSingleFrameDataset` returns tensors with `RGB = float32 [3,H,W]` in
`[-1,1]`, masks as `float32 [1,H,W]` in `[0,1]`, plus metadata. Images are
warped to C1 first, then resized directly to square output resolution. There
is no crop, so all target/render/mask pixels remain exactly co-registered.
The default `grayscale: true` converts both target and MANO render to matching
three-channel grayscale, retaining SD-compatible channel count.

## Install

Create the Miniforge environment and install the pinned dependencies:

```powershell
conda env create -f environment.glove-hot3d.yml
conda activate glove-hot3d
python -m pip install -r requirements.glove-hot3d-gpu.txt
python -m pip install -r requirements.glove-hot3d.txt
```

The NVIDIA driver supplies the runtime; a separate system CUDA Toolkit is not
required. On PACE, choose the PyTorch wheel whose CUDA runtime is compatible
with the node driver, then run the same `torch.cuda.is_available()` check.

This installs Diffusers/Accelerate but does not download model weights. The
first train/inference invocation downloads the configurable public base model
`runwayml/stable-diffusion-v1-5` into the Hugging Face cache.

## Commands

Generate input/condition debug grids without Diffusers:

```powershell
C:\Users\Shaoyu\miniforge3\envs\glove-hot3d\python.exe debug_hand_restoration_samples.py --config configs/hand_restoration/debug_samples.json
```

Run the corresponding deterministic preprocessing smoke test:

```powershell
C:\Users\Shaoyu\miniforge3\envs\glove-hot3d\python.exe smoke_test_hand_restoration.py
```

After installing dependencies and allowing the public SD weights to download,
also check one real diffusion forward pass and ControlNet checkpoint reload:

```powershell
C:\Users\Shaoyu\miniforge3\envs\glove-hot3d\python.exe smoke_test_hand_restoration.py --diffusion
```

Tiny local-window overfit run (frames 10-20 of clip 000000, GPU strongly recommended):

```powershell
accelerate launch train_hand_restorer.py --config configs/hand_restoration/tiny_overfit.json
```

Run an inference using a saved ControlNet checkpoint:

```powershell
C:\Users\Shaoyu\miniforge3\envs\glove-hot3d\python.exe infer_hand_restorer.py --config configs/hand_restoration/inference.json --checkpoint outputs/hand_restoration/tiny_overfit/controlnet_final.pt
```

Compare one exact frame across every checkpoint in a training output directory:

```powershell
C:\Users\Shaoyu\miniforge3\envs\glove-hot3d\python.exe compare_hand_restoration_checkpoints.py
```

The GUI uses the same seed and sampling settings for every checkpoint. It saves
the target and condition once, then each checkpoint's raw `generated.png`,
hard-mask `restored.png`, PSNR metrics, and vertical generated/restored
comparison images in a timestamped experiment directory.

Preview the public SD 1.5 prior before any HOT3D training (useful as a
baseline, but it is not expected to follow the MANO condition reliably):

```powershell
C:\Users\Shaoyu\miniforge3\envs\glove-hot3d\python.exe infer_hand_restorer.py --config configs/hand_restoration/inference.json --pretrained
```

## Condition modes

- `masked_replace` (default): C1 target background is retained outside the
  edit mask; the hand, dilated boundary, and wrist transition are replaced by
  a coarse MANO render plus a neutral/noise fill.
- `overlay`: coarse MANO is alpha-composited onto the source RGB for alignment
  debugging.
- `mano_only`: MANO render on a plain/noise background for pose-following
  debugging.

The condition builder saves target, shaded MANO RGB, MANO mask, edit mask, and
condition in each debug grid. Inference uses a hard edit mask: generated pixels
are copied without feathering inside the editable region, while condition
pixels are retained outside it to prevent unrelated background edits.

## Provisional choices

The Glove2Hand paper does not publicly specify its exact Stable Diffusion
checkpoint, ControlNet initialization, frozen layers, learning rate, mask and
wrist construction, prompt strategy, scheduler, dataset mixture, training
scale, or image-to-video procedure. This MVP defaults to SD 1.5, freezes VAE,
text encoder, and U-Net, and trains ControlNet only. These are practical
starting assumptions, not claims of paper-faithful reproduction.

Preprocessing can run on CPU, but SD 1.5 training and multi-checkpoint
comparison should use a CUDA-enabled PyTorch installation. The GUI reports
progress checkpoint by checkpoint and keeps the same random seed for fair
visual comparison.

## Formal 20-clip training

The formal dataset uses `clip-000000` through `clip-000019` from HOT3D-Clips `train_quest3`. The committed seed-7 manifest keeps clips 000000--000015 in training and clips 000016--000019 in holdout; a clip name appearing in both sets is rejected before any model is loaded. The legacy `data.clip_tars` setting remains supported for tiny-overfit and inference.

Place the original, unextracted archives at `data/train_quest3/clip-NNNNNN.tar`. Download and validate all missing archives with the official HOT3D-Clips Hugging Face repository only after the required access asset is available:

```bash
conda activate glove-hot3d
python prepare_hot3d_clips.py --download
```

The validator opens every tar, checks the right-camera image and camera metadata, counts total frames, right-MANO frames, and final samples, and writes those statistics into `configs/hand_restoration/splits/clips_000000_000019_seed7.json`. Missing or annotation-poor clips are reported explicitly. Data, download manifests, MANO assets, caches, outputs, and checkpoints remain ignored by Git.

Validate both sides of the split before training:

```bash
python smoke_test_hand_restoration.py --config configs/hand_restoration/train_clips000000_000019.json --split train
python smoke_test_hand_restoration.py --config configs/hand_restoration/train_clips000000_000019.json --split validation
accelerate launch --num_processes 1 train_hand_restorer.py --config configs/hand_restoration/train_clips000000_000019_smoke.json
```

After the isolated smoke run passes, start the formal run with:

```bash
accelerate launch --num_processes 1 train_hand_restorer.py --config configs/hand_restoration/train_clips000000_000019.json
```

The formal configuration trains at 512x512 for ten dataset epochs with a micro-batch of 4 and two-way gradient accumulation (effective batch 8). At the documented 150 frames per clip and complete right-hand availability, that is 2,259 raster-visible train samples, 283 optimizer steps per epoch, and 2,830 optimizer steps total. Frames whose right MANO mesh is outside the canonical C1 view are explicitly filtered during indexing. Every remaining indexed sample is visited once per epoch and therefore ten times over the run, while diffusion noise and timestep are resampled on every visit. The trainer derives the actual total from the indexed sample count, runs fixed-order, fixed-seed validation exactly once per indexed epoch, and records configuration, split, Git commit, GPU, dependency versions, losses, steps, samples seen, and timing in the experiment directory. Resume remains available with `--resume PATH_TO_CHECKPOINT`.
