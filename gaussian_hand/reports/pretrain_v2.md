# MANO-only Gaussian Hand: pre-training readiness

Prepared on 2026-09-10, branch `agent/gaussian-hand-hot3d`.
Formal 30,000-step training has **not** started. Existing diffusion and P0001
Gaussian baselines are unchanged. The following is preparation plus short
pipeline validation, not a claim of final appearance quality.

## Geometry and supervision contract

- Separate model for each `(HOT3D, subject_id, handedness)`; no mixed identities.
- Fixed subject MANO shape and GT frame pose. Canonical geometry and per-frame
  pose corrections are frozen at zero.
- Gaussian centers are barycentric points on MANO triangles, with zero normal
  offset. Canonical scales are bounded by local triangle altitude and 4 mm.
- Gaussians have noncompact mathematical tails, so center binding alone does
  not guarantee zero spill. RGB and alpha are explicitly clipped to the
  projected MANO triangle union. No SAM input is used by the renderer.
- RGB reference is `SAM hand mask ∩ projected MANO`. L1 is normalized by this
  reference area; SSIM uses only windows completely inside it. Alpha coverage
  is supervised by MANO, not SAM. SAM forearm pixels cannot grow the geometry.
- Renderer RGB is premultiplied: overlay is `RGB + (1-alpha) * source`.
  The hard outside-MANO guarantee applies before lossy video encoding.

Freezing geometry/pose and applying strict silhouette support are intentional
changes from the earlier learnable-mesh reproduction to meet the overlay
requirement. The implementation is independent of the old external bare-hand
repository. The existing pose-conditioned appearance network remains; no
manual relighting interface is added.

## Prepared data

Official split membership, GT metadata and nonempty MANO files select 136 Aria
train sequences from 9 subjects. All 136 already have completed SAM outputs;
none were regenerated. The 62 available test/no-GT sequences were excluded.
Some train sequences have no frames surviving selection for a given hand.

| Subject | Left frames | Right frames | Left/right contributing sequences |
|---|---:|---:|---:|
| P0001 | 1,497 | 1,493 | 11 / 11 |
| P0002 | 2,469 | 1,516 | 5 / 5 |
| P0003 | 1,321 | 863 | 8 / 8 |
| P0009 | 6,119 | 5,754 | 19 / 19 |
| P0010 | 3,078 | 2,466 | 13 / 13 |
| P0011 | 349 | 1,934 | 15 / 19 |
| P0012 | 6,270 | 6,896 | 21 / 21 |
| P0014 | 1,781 | 2,570 | 11 / 11 |
| P0015 | 4,102 | 3,990 | 17 / 17 |

Total: **54,468 crops**, all training, no held-out split. Per-subject MANO betas
are identical across each prepared hand dataset (maximum variation zero).

Selection requires at least 98% MANO vertices inside the image, 99.5% inside
the crop, 90% SAM coverage of projected MANO and 1% projected hand area.
The audit uses a downsampled triangle-union coverage proxy. Crops are 1.35x
projected MANO bounds, resized to 512x512 with transformed intrinsics.
Laplacian variance on reference pixels must be at least 8; grayscale values
below 8 or above 247 may occupy at most 5% of reference pixels.

The earlier raw-object-bbox overlap gate is **not used**: raw box metadata is
not uniformly available. Existing SAM coverage is the occlusion proxy, not
proof of zero occlusion. No diffusion synthesis or invented texture is used.

Each dataset stores RGB, full SAM, MANO support, intersected reference masks,
pose/camera metadata, and a manifest. Inventory, preparation status, per-hand
warnings and manifest SHA256 values are under
`outputs/gaussian_hand/pretrain_v2/{inventory,preparation_report,readiness}.json`.
The 36 audit/train configs are in `gaussian_hand/configs/pretrain_v2/`.

## Validation completed

- Official HOT3D MANO parity, 20 poses per side: maximum vertex error
  1.35e-7 m left / 4.47e-8 m right, including the left-hand shapedirs fix.
  See [MANO validation](pretrain_v2_mano_validation.json).
- P0002 left and right: each 600 optimizer steps on GPUs 4–7, including adaptive
  splitting, completed with finite losses. Measured training-loop durations
  were 12.21 and 11.95 seconds; these exclude startup and are not a full-run
  duration estimate. Left checkpoint resume from step 600 to 620 also passed;
  all four ranks had identical complete model/pose state SHA256 values.
- On 12 checked frames per side, Gaussian center deviation from its triangle
  surface was exactly zero, geometry/pose corrections remained zero, and
  outside-MANO overlay pixel change was exactly zero. Mean reconstruction loss
  decreased from 0.3072 to 0.1511 left and 0.3119 to 0.1459 right.
  See [overlay validation](pretrain_v2_overlay_validation.json).
- Left/right source-render-overlay crop videos each have exactly 60 frames,
  2 seconds, 1536x512. These are short-test previews, not full-resolution
  original-video deliverables:
  [left preview](../../outputs/gaussian_hand/pretrain_v2/smoke/left_overlay.mp4),
  [right preview](../../outputs/gaussian_hand/pretrain_v2/smoke/right_overlay.mp4).
- Loss regression tests include CPU/CUDA SSIM finite differences, invariance
  to SAM forearm pixels and premultiplied-alpha composition; all four passed.
- Full repository CPU test suite: 44 passed, one CUDA test skipped there;
  that CUDA regression passed in the separate GPU check above.

Smoke artifacts and logs are under `outputs/gaussian_hand/pretrain_v2/smoke/`.
They were produced from this turn's working-tree changes; embedded historical
HEAD hashes alone do not identify those uncommitted changes.

## Visual review and limitations

Inspected 24 diverse samples per hand, **432 samples total**, not every frame.
Each hand's `dataset/diverse_review.jpg` dims pixels outside MANO and marks
MANO pixels missing SAM reference in red. Sampling includes contributing
sequences and diverse pose/view features. For example:
[P0011 left review](../../outputs/gaussian_hand/pretrain_v2/P0011_left/dataset/diverse_review.jpg).

- P0011 left has only 349 accepted frames. It is prepared but low-data; filters
  were not automatically relaxed.
- 16 of 18 hands have under 5% of frames on the minority hand-coordinate
  Z viewing hemisphere. This is a view-direction proxy, not measured mesh
  surface coverage. Visual review confirms predominantly hand-back views.
- Some blur, lighting variation, fingertip/wrist tracking mismatch and nearby
  objects remain. Thresholds are heuristics. Unobserved surfaces cannot be
  expected to acquire accurate personalized texture through overfitting.
- Fixed MANO cannot correct GT-to-image contour error. Strict silhouette
  clipping can expose a hard wrist boundary; 600-step previews still have
  incomplete fine-scale coverage. Neither is evidence of final quality.
- MANO support does not encode external object occlusion. Final original-video
  compositing will need an independent foreground-occluder/depth policy to keep
  objects in front of the hand. SAM remains a color-reference mask during
  training, not a replacement for this scene-compositing policy.

The old P0001/right checkpoint remains a historical baseline but does not meet
this strict contract. Future formal v2 training therefore covers all **18**
models, including P0001/right, rather than reusing that old checkpoint.

## Reproduction entry points

Preparation only (reruns audit/export into the chosen output directory):

```bash
source /data/szeng87/venvs/gaussian-hand/bin/activate
python -m gaussian_hand.prepare_subjects --workers 6
python -m gaussian_hand.review_preparation --root outputs/gaussian_hand/pretrain_v2
```

All preparation workers hide CUDA. Four-GPU smoke checks have finished and
GPUs 4–7 were released. No formal training process was left running.
