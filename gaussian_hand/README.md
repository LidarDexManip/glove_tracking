# Gaussian Hand

This directory is a clean implementation of the subject-specific Gaussian Hand
stage used by Glove2Hand. It must not import code or checkpoints from
`external/bare-hand_gaussian_reconstruction`.

The first target is a high-quality, right-hand model for HOT3D subject `P0001`.
HOT3D cannot reproduce the paper's dedicated hand-only capture, so deviations
belong in the data adapter and must not silently simplify the Gaussian model.

## Method fidelity

The implementation should preserve the original method's defining pieces:

1. Surface-grounded **2D** Gaussians on a learnable canonical hand mesh.
2. Learnable triangle assignment/barycentric coordinates and normal offsets.
3. Learnable canonical geometry plus per-frame hand-pose refinement.
4. Deformation-gradient transport of each Gaussian's local ellipse under hand
   articulation, rather than rigidly attaching a fixed 3D covariance.
5. Adaptive Gaussian densification/reassignment during optimization.
6. Pose-conditioned appearance/lighting and separate palm/back appearance where
   required by the paper.

Every simplification must be exposed as an ablation. A static MANO mesh with
fixed bindings and baked vertex colour is useful as a smoke test, but is not the
faithful model.

## Subject identity contract

One trained Gaussian model belongs to exactly one `(dataset, subject_id,
handedness)` tuple. All source sequences must have that subject prefix. Training
must write the binding, sequence list, selected-frame manifest hash, config hash,
code commit, and MANO shape hash into the model metadata. Loading a model for a
different subject must fail loudly.

The initial binding lives in `configs/subjects/P0001_right.json`. The six listed
sequences are an audit-derived starting set chosen for complementary views, not
a final accepted-frame manifest.

## HOT3D data adapter

The adapter will produce a subject-specific manifest keyed by
`(sequence_id, frame_index, timestamp_ns, handedness)` and apply these stages:

1. Join pinhole RGB, per-frame calibration, headset trajectory, MANO pose/shape,
   HOT3D QA masks, visible-hand segmentation, and object annotations.
2. Project the selected hand's full MANO surface into the pinhole image.
3. Estimate completeness with both geometry and pixels:
   `mano_inside_image`, `mano_inside_crop`, and
   `visible_ratio = area(visible_hand_mask ∩ projected_mano) /
   area(projected_mano)`.
4. Reject frames with excessive object/depth occlusion, truncation, blur,
   exposure failure, tiny hand area, invalid pose, or bad calibration.
5. Form a square hand crop and zoom it to the training resolution. Cropping is a
   camera transform: for crop origin `(x0, y0)` and resize `(sx, sy)`, use
   `fx'=sx*fx`, `fy'=sy*fy`, `cx'=sx*(cx-x0)`, and `cy'=sy*(cy-y0)`.
6. Split train/validation by sequence or contiguous temporal windows, never by
   independent neighboring frames, then sample for pose and wrist-local view
   coverage instead of retaining every redundant frame.

Raw object-box overlap is only an early rejection proxy. Final acceptance must
use the visible hand mask against the full projected MANO silhouette and, when
available, object segmentation/depth ordering.

## Planned layout

```text
gaussian_hand/
  configs/subjects/       subject-bound data and model configs
  data/                   HOT3D joins, visibility audit, crop/export manifest
  model/                  canonical mesh and surface 2D Gaussians
  rendering/              differentiable rasterization and camera transforms
  training/               losses, pose refinement, densification, checkpoints
  identity.py             enforced subject/model binding
tests/gaussian_hand/       geometry, camera, identity, and training smoke tests
```

Generated crops, manifests, checkpoints, and renders belong under ignored
`data/derived/gaussian_hand/` and `outputs/gaussian_hand/`. Only configs, compact
reports, hashes, and model cards should be committed.

## Milestones

- G0: reproduce and visualize the P0001 frame audit inside this directory.
- G1: export calibrated zoomed crops and masks; manually review a diverse sheet.
- G2: validate the differentiable 2D-Gaussian renderer on a static canonical hand.
- G3: add articulated deformation gradients and per-frame pose refinement.
- G4: add adaptive assignment/densification and the complete appearance model.
- G5: train/freeze the P0001 model and expose its aligned RGB/alpha render provider
  to the already-frozen diffusion baseline.

Do not install a guessed dependency stack before G2. First pin the chosen
upstream 2D-Gaussian rasterizer commit and its CUDA/PyTorch compatibility, then
record the environment in this directory.
