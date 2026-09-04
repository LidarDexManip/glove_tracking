# Gaussian Hand

Independent, subject-bound reproduction of the Gaussian Hand stage described by
Glove2Hand. This package does not import code or checkpoints from the old
external/bare-hand_gaussian_reconstruction directory.

Status: **training-ready; the 120k-step run has intentionally not started**.
The selected identity is HOT3D / P0001 / right, model id
hot3d-P0001-right-gaussian-hand-v1.

## What is implemented

- A learnable canonical MANO mesh, verified against the official HOT3D provider
  to 0.000000071 m maximum vertex error before canonical correction.
- Four 2D Gaussian surfels per MANO triangle (6,152 total), with learned
  barycentric coordinates, in-plane scale/rotation, opacity, albedo, and bounded
  normal offset.
- Triangle deformation-gradient transport of the local ellipse.
- Exact point-to-triangle adaptive re-assignment after canonical mesh updates.
- Pose-conditioned degree-2 SH lighting with separate palm/back branches.
- Learnable per-training-frame MANO PCA, wrist rotation, and translation deltas.
- Differentiable gsplat 2DGS rasterization, image/alpha reconstruction,
  distortion regularization, validation PSNR, resumable checkpoints, and
  frame-aligned RGB/alpha sequence rendering.

The paper does not publish Gaussian count, SH order/MLP shape, learning rates,
loss weights, or adaptive schedule, and no official Gaussian training code is
linked as of 2026-09-04. Those choices are listed as reproduction assumptions in
configs/train/P0001_right_512.json. Dynamic split/clone densification is not
claimed: this version starts with dense fixed surface coverage and performs the
paper's triangle re-assignment. Treat dynamic densification as a later ablation.

## Frozen P0001/right data audit

All 16 available P0001 pinhole sequences were audited. A frame must pass HOT3D
QA/SAM eligibility and all of:

- at least 98% of MANO vertices inside the source image;
- visible-MANO completeness at least 0.90;
- projected MANO area at least 1% of the image;
- at least 99.5% of MANO vertices inside a 1.35x square crop;
- no reprojected HOT3D object bbox overlapping more than 5% of the MANO bbox.

Of 35,580 candidates, 1,251 passed. The sequence-disjoint split is 904 train,
230 validation, and 117 test. Crops are 512x512, and the pinhole intrinsics are
updated with pixel-centre-correct crop/resize math. The final contact sheet was
reviewed after the object gate; obvious held-object examples present in v1 are
absent in v3.

Versioned generated data (on the /data-backed outputs symlink):

- outputs/gaussian_hand/P0001_right/audit_v3/frames.jsonl
- outputs/gaussian_hand/P0001_right/dataset_v3/frames.jsonl
- outputs/gaussian_hand/P0001_right/dataset_v3/contact_sheet.jpg

## Environment

The tested environment is /data/szeng87/venvs/gaussian-hand, Python 3.10,
PyTorch 2.11.0+cu128, gsplat 1.5.3, and the versions in requirements.txt.
Activate it before any gsplat command:

    source /data/szeng87/venvs/gaussian-hand/bin/activate

The first CUDA call compiles gsplat for the installed GPU. Keep environments,
datasets, and checkpoints on /data; the system filesystem is 98% full.

## Reproduce the preparation

    python -m gaussian_hand.data.audit_hot3d \
      --config gaussian_hand/configs/subjects/P0001_right.json \
      --output outputs/gaussian_hand/P0001_right/audit_v3

    python -m gaussian_hand.data.export_crops \
      --config gaussian_hand/configs/subjects/P0001_right.json \
      --manifest outputs/gaussian_hand/P0001_right/audit_v3/frames.jsonl \
      --output outputs/gaussian_hand/P0001_right/dataset_v3

## Train (only after explicit approval)

    CUDA_VISIBLE_DEVICES=0 python -m gaussian_hand.train \
      --config gaussian_hand/configs/train/P0001_right_512.json \
      --output outputs/gaussian_hand/P0001_right/train_120k

Resume without resetting optimizer or per-frame pose deltas:

    CUDA_VISIBLE_DEVICES=0 python -m gaussian_hand.train \
      --config gaussian_hand/configs/train/P0001_right_512.json \
      --output outputs/gaussian_hand/P0001_right/train_120k \
      --resume outputs/gaussian_hand/P0001_right/train_120k/step_060000.pt

A 1,200-step 512px benchmark crossed the canonical-mesh activation point and
two adaptive cycles without non-finite values. Warm speed was 20.42 ms/step.
The current fixed-6,152-Gaussian configuration therefore estimates 41 minutes
of raw optimization and 50-70 minutes including 32-frame validation,
checkpoints, and variance. Measured device allocation including CUDA setup has
stayed below 2 GiB. Dataset plus 25 checkpoints should remain below 0.25 GiB.

## Render every MANO frame

This emits an RGB MP4, lossless alpha MKV, and JSONL with original frame indices.
Frames without a right-hand MANO pose, camera pose, or any in-image hand are
blank but remain aligned.

    CUDA_VISIBLE_DEVICES=0 python -m gaussian_hand.render_sequence \
      --checkpoint outputs/gaussian_hand/P0001_right/train_120k/step_120000.pt \
      --sequence P0001_a9d6c83d \
      --output outputs/gaussian_hand/P0001_right/renders/P0001_a9d6c83d

A short checkpoint is only a pipeline smoke test, not a usable appearance model.
