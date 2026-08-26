# HUGG Aria hand-restoration data contract

This document describes the data pipeline that must be satisfied before a new
baseline is frozen. It intentionally does not freeze optimizer hyperparameters
or launch a training run.

## 1. Canonical RGB source

Do not rebuild pinhole RGB from raw VRS as part of an experiment. Download the
published, frame-aligned dataset and use it as the only RGB/calibration source:

```bash
huggingface-cli download LIDAR-GT/HUGG_ARIA_PINHOLE \
  --repo-type dataset \
  --local-dir data/HUGG_ARIA_PINHOLE
```

Every downstream asset is keyed only by `(sequence_id, frame_index)`.
The historical compact-render index and `gaussian_frame_index` are not part of
the training schema.

## 2. Replaceable render-provider interface

A Gaussian or naive-MANO provider must expose the same files for every sequence:

```text
<render_root>/<sequence>/reconstruction.mp4
<render_root>/<sequence>/alpha.mkv
```

Requirements:

- both videos contain exactly one frame per pinhole source frame;
- `reconstruction.mp4` stores straight RGB hand color;
- `alpha.mkv` stores aligned 8-bit alpha using lossless FFV1/gray;
- an invalid or absent render is represented by zero alpha, never by dropping a
  frame;
- black RGB values do not define foreground.

The Gaussian exporter implementing this contract lives in the separate
`LidarDexManip/bare-hand_gaussian_reconstruction` repository. Renderer commits
and output hashes belong in each generated asset's metadata; the restoration
trainer itself accepts any provider satisfying the interface.

The fixed condition construction is:

```text
effective_alpha = render_alpha * SAM_binary * render_opacity
condition = target * (1 - effective_alpha) + render_rgb * effective_alpha
```

## 3. SAM2

The current SAM2 v3 propagation policy is unchanged while its masks are reviewed:

- split episodes only when left/right hand presence changes;
- prompt only the first episode frame;
- prefer a MANO-projected bbox;
- fall back to the original bbox reprojected into the pinhole camera;
- propagate through the episode;
- apply QA/visibility/exposure filtering after propagation;
- apply aligned render-alpha validity only in the final manifest;
- do not use a large-mask rejection threshold.

Existing SQLite outputs remain the current SAM source, so this refactor does not
silently regenerate masks before review. Future SAM generation uses the same
source `frame_index` directly and has no compact-Gaussian mapping input. The
episode prompts and propagation policy are unchanged.

## 4. Naive MANO RGB and alpha

Generate aligned MANO assets directly from SAM-eligible frame indices:

```bash
python render_hugg_aria_mano_aligned.py \
  --workers 16 --torch-threads 1 --codec libx264
```

The renderer writes every source frame and records, for every SAM-eligible
frame, whether rasterization produced nonempty alpha. Metadata QA can be valid
while a mesh raster is empty (for example, the mesh projects completely outside
the output camera), so nonempty alpha is a necessary final observed condition.

## 5. Single filtered manifest product

After MANO alpha generation:

```bash
python build_hugg_aria_aligned_manifest.py
```

This creates one product directory:

```text
data/derived/hugg_aria_diffusion/
  train_manifest.jsonl
  seen_eval_manifest.jsonl
  manifest_summary.json
```

A training row is retained iff it is SAM `training_eligible`, its aligned
MANO alpha is nonempty, and `MANO alpha ∩ SAM` is nonempty. The last condition
prevents an unchanged condition image from entering training when both masks
exist but do not overlap. Rows contain only:

```text
sequence_id, frame_index, timestamp_ns
```

The seen-eval manifest contains one deterministic in-domain training frame per
sequence. There is no held-out split.

## 6. Training input and loss

The render provider is selected by `data.render_root`; Gaussian and naive MANO
experiments use the same manifest and differ only in render source/output path.

The current regional objective uses the aligned naive-MANO alpha as its hand
region for both providers. It intentionally differs from the condition support:
condition support is render alpha intersected with SAM, while the weighted loss
support is the full MANO alpha.

The text condition is inherited from the original SD1.5 ControlNet
implementation in this repository. It is fixed for every sample and is not an
ablation variable.

Spatial augmentation and hand-centric crops are not part of the current data
contract; they may be tested later.

## 7. Exact resume and LR horizon

Full trainer states contain model, AdamW, LR scheduler, mixed-precision scaler,
RNG, sampler position, optimizer step, epoch, and reporting state.

For `constant` and `constant_with_warmup`, a new config may increase
`num_train_epochs` or `max_train_steps` and resume the complete state without
resetting optimizer/scheduler/RNG. Every other config and data field, batch
geometry, dataset fingerprint, and world size must remain unchanged. Shrinking
the target is rejected.

Target extension is deliberately rejected for cosine/linear schedules because
their behavior depends on the originally declared horizon. This prevents an
already-decayed schedule from being mislabeled as an exact, useful extension.
