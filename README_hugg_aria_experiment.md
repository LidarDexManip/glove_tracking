# HUGG Aria aligned hand-restoration experiment

This document freezes the data policy and commands used for the final
hand-weight-10 ControlNet experiment. Generated data, SQLite masks, videos,
logs, and checkpoints are intentionally excluded from Git.

## Required local assets

The experiment expects these paths relative to the repository root:

```text
data/HUGG_ARIA_PINHOLE/<sequence>/rgb_214_1_pinhole.mp4
data/HUGG_ARIA_GAUSSIANS_ALIGNED/<sequence>/reconstruction.mp4
outputs/sam2_hugg_aria_masks_v3_pilot/<sequence>/masks.sqlite
outputs/sam2_hugg_aria_masks_v3_pilot/<sequence>/_SUCCESS.json
model/sam2/sam2.1_hiera_large.pt
```

There are exactly 136 useful sequences. The aligned Gaussian render must use
commit `18d74ba1bade042335a563640d3d38407e582c1e` of
`LidarDexManip/bare-hand_gaussian_reconstruction`. That renderer preserves
empty frames, so `gaussian_frame_index == frame_index`.

The renderer repository is a separate dependency. Clone it under `external/`
if desired; that directory is ignored rather than committed as a nested Git
repository.

## SAM v3 policy

The v3 pipeline is implemented by:

- `hugg_aria_sam_policy.py`
- `segment_hugg_aria_pinhole_v3.py`
- `run_hugg_aria_sam2_v3_batch.py`
- `visualize_sam_mask_sqlite.py`

SAM propagation is restarted only when per-hand presence changes. The first
frame of an episode uses a MANO-derived box when pose QA allows it, otherwise
the original annotated box reprojected into the pinhole camera. Training
eligibility is applied after propagation. A frame is rejected when Gaussian
hand validity, general QA, MANO-pose QA, hand visibility, exposure, or required
SAM output fails. There is no large-mask rejection threshold.

The current v3 batch command is:

```bash
python run_hugg_aria_sam2_v3_batch.py --gpus 0,1,2,3,4,5,6
```

The original compact Gaussian render and
`outputs/gaussian_frame_mapping/*.csv` are inputs to this historical SAM v3
generation step because they provide per-hand validity. They are not used by
the aligned training loader.

## Rebuild the exact aligned manifests

The committed seen-eval specification contains one training-eligible frame
from every sequence. All 136 sequences and all eligible frames are used for
training; this is an in-domain evaluation, not a held-out test split.

```bash
python build_hugg_aria_aligned_manifest.py
```

The command reads the v3 SQLite databases directly, groups each sequence into
64-record blocks, shuffles the blocks with seed 7, and writes:

```text
data/derived/hugg_aria_diffusion_aligned/train_manifest.jsonl
data/derived/hugg_aria_diffusion_aligned/seen_eval_manifest.jsonl
data/derived/hugg_aria_diffusion_aligned/manifest_summary.json
```

Expected invariants:

```text
sequences: 136
training frames: 399164
seen-eval frames: 136
train SHA-256: 02345e733f5876a484260a14f9edff53ae407b82d0ef5764de65750de1f8678f
seen-eval SHA-256: b1f29e75f2c2dd83623cc514c7579d3f852b45509d720f8fe880cd50fe079c69
```

Validate loader construction before training:

```bash
python smoke_test_hugg_aria_dataset.py \
  --config configs/hand_restoration/hugg_aria_aligned_sam_weight10_chunk5s_512_8gpu_batch256_resume10epochs.json
```

## Two-stage eight-GPU training

Both stages use 512 px inputs, BF16, 32 samples per GPU, eight GPUs, global
batch 256, and temporal 150-frame chunk shuffling.

Stage 1 trains ten epochs with hand-region weight 5:

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  train_hand_restorer.py \
  --config configs/hand_restoration/hugg_aria_aligned_sam_weighted_chunk5s_512_8gpu_batch256_epochs10.json
```

Stage 2 warm-starts from stage 1 and trains ten more epochs with hand-region
weight 10:

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  train_hand_restorer.py \
  --config configs/hand_restoration/hugg_aria_aligned_sam_weight10_chunk5s_512_8gpu_batch256_resume10epochs.json \
  --resume outputs/hand_restoration/hugg_aria_aligned_sam_weighted_chunk5s_512_8gpu_batch256_epochs10/controlnet_final.pt
```

The completed model has total step 31200. The historical six-epoch
continuation config documents the earlier weights-only recovery from
`controlnet_step021840.pt`; that run was necessarily a warm start.

New runs also save `trainer_state_stepNNNNNN/` directories containing the
ControlNet, AdamW moments, exact cosine scheduler/scaler state, per-rank RNG,
loss EMA, epoch and next batch. Two recent full states are retained. Continue
an interrupted run with the original config and world size:

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  train_hand_restorer.py \
  --config configs/hand_restoration/hugg_aria_aligned_sam_weight10_chunk5s_512_8gpu_batch256_resume10epochs.json \
  --resume outputs/hand_restoration/hugg_aria_aligned_sam_weight10_chunk5s_512_8gpu_batch256_resume10epochs/trainer_state_stepNNNNNN
```

Directory resume continues toward the original target rather than adding more
epochs. A `.pt` resume remains available when a deliberate new optimizer and
learning-rate schedule are desired.

## Curves and browser inference

Generate the connected 20-epoch path (weight 5 followed by weight 10) while
replacing the abandoned, overlapping interruption branch with the later
checkpoint-backed continuation rows:

```bash
python plot_hugg_weight10_training.py \
  outputs/hand_restoration/hugg_aria_aligned_sam_weight10_chunk5s_512_8gpu_batch256_resume10epochs/training_log.csv \
  --base-log outputs/hand_restoration/hugg_aria_aligned_sam_weighted_chunk5s_512_8gpu_batch256_epochs10/training_log.csv \
  --output outputs/hand_restoration/hugg_aria_aligned_sam_weight10_chunk5s_512_8gpu_batch256_resume10epochs/full_training_curves_weight5_to10.png \
  --epoch-csv outputs/hand_restoration/hugg_aria_aligned_sam_weight10_chunk5s_512_8gpu_batch256_resume10epochs/full_epoch_metrics_weight5_to10.csv
```

Start the local Gradio page on one GPU:

```bash
CUDA_VISIBLE_DEVICES=7 python hugg_aria_hand_restoration_web.py \
  --host 127.0.0.1 --port 7860
```

Use SSH or VS Code port forwarding for port 7860. The page only exposes the
frozen 136 in-domain frames and shows condition input, raw model output, and
ground truth. It defaults to the newest complete checkpoint, including
`controlnet_final.pt`.

## Gaussian versus MANO render ablation

The render-source ablation must start both branches from the same unconditioned
initialization. Do not initialize the MANO branch from a checkpoint that has
already learned Gaussian conditions. Both committed configs use seed 7 and
create ControlNet from the same frozen SD 1.5 UNet. They differ only in render
root, render kind, and output directory.

Render the MANO prior once. The output includes every source frame; filtered
or MANO-out-of-view frames are black so frame indices never shift. Each sequence
also records `mano_availability.csv`. Rendering uses the same per-frame LINEAR
camera calibration, headset trajectory, MANO poses, and licensed MANO model
files as SAM prompting:

```bash
python render_hugg_aria_mano_aligned.py \
  --workers 24 --torch-threads 1 --codec libx264

python build_hugg_aria_render_ablation_manifest.py
```

The second command intersects the original filtered manifest with MANO raster
availability. Both branches therefore train on the exact same common frames;
an empty MANO render is never allowed to turn into an unchanged condition that
already equals the target. It also keeps one deterministic seen-eval frame per
sequence, replacing the old choice with the nearest common frame when needed.
The condition overlay remains `render foreground AND SAM`. The weight-10 loss
region is instead the foreground rasterized from the aligned MANO video for
both branches (`loss_mask_source=mano`). The Gaussian branch therefore reads
MANO only to construct its loss mask; MANO pixels are not placed in its
condition. This keeps the supervised hand geometry identical without allowing
SAM errors or the two condition silhouettes to change the objective.

The paired five-epoch run uses eight GPUs, 32 images/GPU (global batch 256),
BF16, hand weight 10, 150-frame chunk shuffle, AdamW with weight decay 0.01, LR
`5e-6`, 100 warmup steps, and then a constant LR. Equal 500-step continuation
pilots at `5e-6`, `1e-5`, and `2e-5` selected `5e-6`: it had the lowest fixed
validation total, hand-region, and background-region losses. Unlike cosine, the
constant-with-warmup schedule does not decay the LR to zero during this short
comparison.

```bash
# Select LR from equal 500-step Gaussian pilots at 5e-6, 1e-5 and 2e-5.
./run_hugg_lr_sweep.sh
```

Run the paired experiment after selecting the LR from fixed validation metrics
and images:

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  train_hand_restorer.py \
  --config configs/hand_restoration/hugg_aria_ablation_gaussian_manoloss_weight10_lr5e6_constant_5epochs.json

accelerate launch --num_processes 8 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  train_hand_restorer.py \
  --config configs/hand_restoration/hugg_aria_ablation_mano_manoloss_weight10_lr5e6_constant_5epochs.json
```

Run both commands without `--resume`. The common-manifest summary records the
final sample count; both runs use those exact records, 136 in-domain sequences,
chunk order, random seed, optimizer, batch geometry, MANO loss mask, and loss.
Compare validation hand-region loss, background-region loss, total loss, and
raw inference output on the same seeds. A lower training loss alone is not
evidence that one render prior is better.

The older configs without `manoloss` are retained only to reproduce the first
completed pair, whose weight-10 region was based on SAM rather than MANO.
