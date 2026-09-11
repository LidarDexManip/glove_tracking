# Gaussian vs. frozen MANO diffusion: strict input preparation

This supersedes the paused `diffusion_bridge_v2.md` investigation. No diffusion
training is authorized or started by any preparation script. The long five-subject
demo is complete: `outputs/gaussian_hand/result_videos/five_subjects_full_sequences.mp4`
(18,992 frames, 10 min 33 s).

## Final acceptance: ready, training not started

Completed on 2026-09-11. All 136 aligned render caches and 136 image archives
are complete. Coverage is exactly 393,684 unique source frames and 524,917
adaptive crop records. The 1,049,834 exported images occupy 162,415,953,920
archive bytes (151.26 GiB). Every archive header and every index record was
checked. All preparation processes exited; rendering GPUs 0–2 were released.

Both sparse and encoded-video QA passed all 183 records. Frozen baseline
conditioning drift, outside-edit pixel changes and protected-object pixel
changes are all zero. Nine encoded examples spanning all nine subjects were
also visually reviewed (`aligned_visual_review.json`), in addition to the
earlier 18 sparse examples. Three actual CPU training-dataloader batches
spanning nine subjects passed; one lossless archived condition per subject
matched live conditioning exactly after the documented uint8 conversion.

The independent dataset auditor called `validate_ready(CONFIG)` successfully
after binding `dataset_audit.json` and `ASSET_LOCK.json` into `READY.json`.
Audit SHA256: `c78f349bd180eaef270fdf6cc956669874d52496fd40c93977dc1e96f67393c4`.
Encoded QA SHA256: `529840a573b42401adb0f41e74d04edddd25dcf3905fb83c4b2110a58c38947e`.
Tests: 65 passed, one CUDA-only test skipped with CUDA deliberately disabled.
Accelerate command arguments parsed successfully without launching processes.
An extra invocation of the launcher's default validation mode was rejected by
the execution safety reviewer because that module also contains a training
entrypoint; it was not executed. This does not invalidate the independently
passed readiness gate. No diffusion model or optimizer was initialized, and the
new diffusion training output directory does not exist.

## Controlled comparison

The baseline is `baselines/adaptive_splitcrop_v1/BASELINE_FREEZE.json`, training
commit `3b299332fded18066753ca039d5bf17ff3f14427`, checkpoint SHA256
`76a9a0243700388b20cffed5d0b8cf3f4bf5e32ebaa3a7a8a257149fd509e1e3`.

| Item | Both arms |
| --- | --- |
| Population | Same 9 subjects, 136 GT Aria training sequences |
| Training data | 393,684 unique source frames; 524,917 adaptive crop records |
| Seen evaluation | Same 183 records; same seeded 136-record numeric subset |
| Geometry and crop | Original GT MANO, same adaptive single/joint crops |
| Masks | Original MANO loss/edit support, SAM visibility, dilation and wrist rules |
| Initialization | Fresh ControlNet from local SD1.5 UNet; no baseline ControlNet warm start |
| Schedule | 8 ranks × batch 32 × accumulation 1; 15,380 optimizer steps |
| Other settings | Same seed 7, optimizer, LR, losses, sampler, BF16, evaluation inference |

The changed input is the **subject-bound Gaussian RGB and native opacity**, with
both hands rendered jointly. Gaussian premultiplied RGB is handled explicitly.
This tests the Gaussian rendering treatment, not color alone with alpha held
fixed. It is not a held-out-subject/generalization experiment: evaluation is
seen-data reconstruction, as requested.

The original rasterizer uses strict barycentric interior pixels, whereas the
Gaussian support helper uses conservative rounded triangle filling. The latter
covers at least 99.9438% of the old silhouette on the 136 diagnostic frames;
boundary overfill explains the previously lower IoU. For this controlled cache,
Gaussian RGB/alpha are additionally clipped to **frozen baseline MANO alpha**, not
SAM. Full-FOV 512 rendering, old H.264 CRF 20 encoding and the existing crop path
are retained. Gaussian training/checkpoints are unchanged.

The original wrist rule can blank some SAM-protected pixels. That behavior is
preserved in both arms; it is not silently fixed as part of the Gaussian change.
Some difficult crops remain blurred, occluded or dominated by wrist blanking.

All 54,468 Gaussian fitting crops were verified to fall inside the original
diffusion training frame pool. Package versions and core model/sampler sources
match the historical baseline. The existing SD asset cache is pinned and hashed
in `ASSET_LOCK.json`; the old run did not record initial asset hashes, so its
historical initialization bytes cannot be proven retrospectively.
All 1,360 historical source-video, MANO, SAM and wrist assets match their frozen
SHA256 hashes, including SQLite WAL content. The readiness gate rechecks asset
sizes and hashes any whose modification time has changed since verification.

## Outputs and readiness

All new data live under `outputs/gaussian_hand/diffusion_strict_v2/`:

- `aligned/<sequence>/`: source-index-aligned Gaussian `reconstruction.mp4`,
  lossless `alpha.mkv`, timestamp/metadata and completion marker. Nonrequested
  source frames remain blank; manifest frame indices are not renumbered.
- `image_sequences/<sequence>/images.tar`: per-record 512-pixel adaptive crops,
  lossless `NNNNNNN.condition.png` and JPEG95 `NNNNNNN.overlay.jpg`.
  `index.jsonl` maps each image to subject, sequence, source frame and crop.
  These are crop overlay sequences, not additional full-resolution videos.
- Training still computes float conditions online from aligned video caches;
  it **does not train on quantized PNG/JPEG archives**.
- `seen_eval_qa/`: all 183 sparse-render comparisons. Numeric tests passed with
  identical target/crop/masks and zero frozen-baseline conditioning drift.
  Eighteen examples spanning all nine subjects were manually inspected.
- `aligned_qa/`: the same checks repeated after full video encoding.
- `dataset_audit.json`: exhaustive archive-header/index checks, CPU training
  dataloader batches and PNG-vs-live comparisons spanning all nine subjects.
- `evaluation_plan.json`: the historical same-ten-frame visual selection and
  all 183 seen-eval crops frozen before training. Use matching per-sample noise
  seeds and final 15,380-step checkpoints; assess raw outputs and identical-mask
  paste-backs separately. The generic old `evaluate_hand_restorer.py` only
  supports derived WebDataset; use the HUGG Aria dataset factory for this run.
- `READY.json`: created only after complete render/image export and encoded QA.
  `preparation_status.json` and `image_prefill_status.json` show live progress.

Full-image counts are 524,917 conditions + 524,917 overlays, archived by sequence
to avoid more than a million loose filesystem entries. Files not yet complete
must not be treated as ready simply because the config exists.

## Commands

Use the existing `/data/szeng87/venvs/gaussian-hand` environment, from repo root.
The read-only final checks are:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m gaussian_hand.audit_prepared_diffusion_dataset
python -m gaussian_hand.launch_diffusion_strict
```

The launcher defaults to validation only. Its explicit `--execute` switch is
reserved for a later user training instruction and a fresh GPU availability
check. It requires all eight ranks for the original layout and offline frozen
pretrained assets. Do not substitute four ranks with accumulation and call it
an exact replay. The JSON retains `num_train_epochs: 10`, but the actual frozen
baseline budget is the explicit 15,380 steps (approximately 7.5 effective epochs).

Training writes only to the separate
`outputs/hand_restoration/hugg_aria_gaussian_surface_strict_v2/` run directory.
The old checkpoint, old MANO data and old manifests are never overwritten.
