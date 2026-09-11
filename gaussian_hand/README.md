# Gaussian Hand

Independent, subject-bound implementation of the Gaussian Hand stage in
[Glove2Hand](https://arxiv.org/abs/2603.20850). It does not import the old
`external/bare-hand_gaussian_reconstruction` code or checkpoints.

## Current: MANO-only Gaussian models and controlled diffusion preparation (v2)

Prepared **9 GT Aria train subjects, both hands: 18 datasets / 54,468 crops**.
Each model is bound to one subject and one hand; all accepted frames train.
No Aria test sequences or new SAM generation are involved.

The v2 contract fixes GT MANO geometry/pose, keeps Gaussian centers on its
triangles, bounds surfel scales, and clips rendering to the MANO silhouette.
SAM only selects visible color-reference pixels inside that silhouette; it is
not an alpha target or rendering boundary. This deliberately disables the
learnable geometry/pose corrections described in the legacy baseline below.

All 18 formal v2 models have completed 30,000 steps, using four GPUs per model.
Checkpoints are under `outputs/gaussian_hand/train_mano_surface_v2_30k/`.
The five-subject full-sequence demo is complete (10 min 33 s). A separate
Gaussian-input diffusion dataset is being prepared; **diffusion training remains
unauthorized and has not started**. See the
[strict comparison runbook](reports/diffusion_strict_v2.md) for preparation
outputs, runtime readiness checks and the frozen baseline contract.

See [the earlier pretraining report](reports/pretrain_v2.md) for per-hand
counts, visual examples, limitations and commands. P0011/left has only 349
accepted frames, and most hands have strongly imbalanced viewing directions.

## Historical P0001 baseline (v1 / training run v3)

Identity: **HOT3D / P0001 / right** (`hot3d-P0001-right-gaussian-hand-v1`).
The diffusion baseline is already merged into local `main` and tagged
`baseline-adaptive-splitcrop-v1`; this work is on `agent/gaussian-hand-hot3d`.

## Completed experimental baseline

The four-GPU run completed 30,000 optimizer steps in 764 seconds, producing
19,763 Gaussians. It used all 1,251 selected P0001/right frames. The checkpoint is
`outputs/gaussian_hand/P0001_right/train_4gpu_30k_v3/step_030000.pt` (about 5 MB).

Start visual review with
[the 18-second comparison video](../outputs/gaussian_hand/P0001_right/previews_v3/comparison.mp4),
[fixed training-frame comparisons](../outputs/gaussian_hand/P0001_right/train_4gpu_30k_v3/visual/step_030000.jpg),
and [low/medium/high error examples](../outputs/gaussian_hand/P0001_right/appearance_v3/quality_range.jpg).
All 16 sequences are rendered under `outputs/gaussian_hand/P0001_right/renders_v3`:
58,820 aligned frames, of which 50,733 contain a visible rendered right hand.
RGB/alpha video counts and source-frame indices were verified.

All-training raw-MANO appearance metrics: black-background crop PSNR 21.17 dB,
SAM foreground PSNR 16.71 dB, mean alpha IoU 0.853. These are overfit diagnostics.
The result has recognizable hand shape and skin color, but fine detail is
smoothed and large rotations/close-ups can produce stretched or planar wrist
artifacts. It is an experimental baseline, not a high-fidelity reproduction.
Possible contributors include forearm pixels in SAM targets, large learned wrist
surfels, and limited surface/view coverage; these causes have not been isolated.

See `reports/P0001_right_result.json` for provenance, counts and limitations.
Repeat the full training-set appearance audit with:

```bash
CUDA_VISIBLE_DEVICES=4 python -m gaussian_hand.evaluate_appearance \
  --checkpoint outputs/gaussian_hand/P0001_right/train_4gpu_30k_v3/step_030000.pt \
  --output outputs/gaussian_hand/P0001_right/appearance_v3
```

## Method and reproduction choices

- Learnable canonical MANO mesh and per-training-frame PCA/wrist pose corrections.
  Unmodified MANO vertices match the official HOT3D provider within 7.1e-8 m.
- Triangle-bound 2D Gaussians with barycentric locations, bounded normal offsets,
  albedo, opacity, anisotropic scales and in-plane rotations.
- Triangle deformation-gradient transport of covariance; gradient-based Gaussian
  splitting/pruning and closest-triangle rebinding.
- Pose-conditioned palm/back SH illumination, multiplied by learned albedo.
- Differentiable gsplat 2DGS and calibrated pinhole crop rendering.

The paper does not specify all implementation hyperparameters. Four surfels per
face (6,152 initially), degree-2 SH, a 2x128 lighting MLP, loss weights, learning
rates, bounds and adaptive schedules are our reproduction choices, recorded in
`configs/train/P0001_right_512.json`. Lighting uses MANO PCA plus wrist translation
as its MLP input, and normalized softplus keeps illumination positive. These
choices should not be described as the authors' released implementation.

The objective combines RGB L1, local SSIM, alpha L1 and weak depth distortion.
Distortion uses an actual depth channel (`RGB+D`), not the blue RGB channel.
A CUDA finite-difference check caught incorrect `avg_pool2d` gradients on strided
NCHW images in this environment; SSIM now explicitly makes images contiguous.
The regression test and `reports/gradient_validation.json` record the check.

## Data

All 16 P0001 sequences were audited: 35,580 eligible candidate frames yielded
1,251 selected crops. **Every selected crop is training data; no validation or
test split is held out.** Crops are 512x512 with pixel-centre-correct intrinsics.

Selection uses existing SAM database exposure/QA eligibility, at least 98% of
MANO vertices in the source image, at least 99.5% in the 1.35x crop, at least 90%
SAM coverage of projected MANO, projected area at least 1%, and projected object
bbox overlap at most 5% of the MANO bbox. Sharpness and luma are exported
measurements, not additional rejection thresholds.

An independent per-triangle-union silhouette recheck passed all 1,251 frames
(minimum coverage 0.907); see `reports/silhouette_union_recheck.json`. The audit
rasterizer now fills triangles individually to avoid even/odd contour holes.
The original frozen audit counts describe the earlier v3 rasterizer; rerunning
the improved audit can change the full candidate selection.

Geometric coverage and object boxes are proxies: selected images can still
contain nearby props, tracking markers, motion blur and SAM forearm regions.
Cropping magnifies existing pixels; no diffusion-generated textures supervise
this model. Unseen surfaces and fine skin detail remain limited by the data.

Generated data is on the `/data`-backed `outputs` symlink:

- `outputs/gaussian_hand/P0001_right/audit_v3/frames.jsonl`
- `outputs/gaussian_hand/P0001_right/dataset_all_v4/frames.jsonl`
- `outputs/gaussian_hand/P0001_right/dataset_v3/contact_sheet.jpg`

## Environment and training

Use `/data/szeng87/venvs/gaussian-hand` (Python 3.10, PyTorch 2.11.0+cu128,
gsplat 1.5.3; see `requirements.txt`). Activation puts ninja on PATH for CUDA
extension compilation. Keep checkpoints and environments on `/data`.

```bash
source /data/szeng87/venvs/gaussian-hand/bin/activate
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run \
  --standalone --nproc-per-node=4 -m gaussian_hand.train \
  --config gaussian_hand/configs/train/P0001_right_512.json \
  --output outputs/gaussian_hand/P0001_right/train_4gpu_30k_v3
```

Four synchronous workers average gradients using manual all-reduce, allowing
Gaussian parameter sizes to change. **30,000 optimizer steps x batch four =
120,000 sampled frames**, not the same optimization as 120,000 batch-one steps.
Warm speed is approximately 26 ms/global step at the start of this run.
Checkpointing, fixed training-frame metrics and galleries occur every 2,500
steps. The metric called `masked_psnr` measures the entire crop against an RGB
image with black masked background; it is not foreground-only PSNR.

Resume with `--resume PATH/step_015000.pt` using the same dataset manifest,
frame ordering, model code and four-worker configuration. Fixed galleries and
sequence rendering use the supplied raw MANO poses, without training-frame pose
corrections, so they also expose the inference-time alignment error.

## Frame-aligned rendering

```bash
CUDA_VISIBLE_DEVICES=4 python -m gaussian_hand.render_sequence \
  --checkpoint outputs/gaussian_hand/P0001_right/train_4gpu_30k_v3/step_030000.pt \
  --sequence P0001_a9d6c83d \
  --output outputs/gaussian_hand/P0001_right/renders_v3/P0001_a9d6c83d
```

Each sequence produces `render.mp4` (RGB), `alpha.mkv` (lossless alpha),
`frames.jsonl` (source frame index, timestamp, crop box and adjusted camera
intrinsics), and `summary.json`. Missing poses/cameras or out-of-view hands yield
blank frames with a reason; frame alignment is preserved. Subject mismatches are
rejected. `--start-frame` and `--max-frames` support bounded intervals.

To render all configured sequences on four GPUs with a wall-time bound and
verify RGB/alpha frame counts:

```bash
python -m gaussian_hand.render_subject \
  --checkpoint outputs/gaussian_hand/P0001_right/train_4gpu_30k_v3/step_030000.pt \
  --subject-config gaussian_hand/configs/subjects/P0001_right.json \
  --gpus 4,5,6,7 --time-limit-seconds 1800 \
  --output outputs/gaussian_hand/P0001_right/renders_v3
```

To make a source-crop/render comparison video and contact sheet:

```bash
python -m gaussian_hand.make_preview \
  --render-directory outputs/gaussian_hand/P0001_right/renders_v3/P0001_a9d6c83d \
  --start-frame 2150 --max-frames 180 \
  --output outputs/gaussian_hand/P0001_right/previews_v3/a9d6c83d.mp4
```

These are appearance-overfit checks, not evidence of held-out generalization.
