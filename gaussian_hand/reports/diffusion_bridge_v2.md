# Gaussian-to-diffusion bridge: paused before GPU validation

2026-09-11. User authorized conditioning QA and a separate full rendering cache,
not diffusion training. GPUs 4–7 were occupied by other users. The user then
requested not to use GPUs when none are free. **No GPU task or automatic
waiting queue was started.**

## Implemented, CPU-tested preparation

- Explicit `render_rgb_semantics="premultiplied"` support in faithful
  conditioning. The default remains `straight`, preserving the MANO baseline.
- Sparse aligned-frame reads for the seen-eval diagnostic cache; existing
  aligned-video reads remain the full-cache path. New cache metadata checks
  enforce RGB semantics, storage and subject identity.
- `render_scene.py`: both hands enter one depth-sorted Gaussian rasterization;
  support is the union of fixed MANO projections, never SAM. This new CUDA path
  is **not yet GPU-tested**.
- `export_diffusion_cache.py`: independent subject-bound sparse/video export,
  source timestamp checks, checkpoint hashes, renderer fingerprint, atomic
  per-sequence completion and encoded frame-count checks. No old data is
  overwritten. This exporter is **not yet end-to-end GPU-validated**.
- CPU tests: 51 passed, 1 CUDA test skipped. New tests cover premultiplied
  equivalence, no target-hand leakage through transparent holes, instance RGB
  suppression and pixel-center-correct camera scaling.

## CPU geometry audit: needs review

All 136 unique seen-eval frames (183 adaptive crop records) were checked for
timestamp agreement and projected fixed-MANO support against the original
naive MANO alpha. Mean silhouette IoU: **0.93699**; minimum: **0.85591**.
This did **not** meet the provisional per-frame 0.90 acceptance threshold.
The differences must be inspected; do not silently lower the threshold or
describe this as a passed Gaussian appearance audit.

Lowest examples:

- P0011_cee8fe4f:635 — IoU 0.85591; old/new area 2,085 / 2,436 pixels.
- P0001_f6cc0cc8:484 — IoU 0.86167; old/new area 1,713 / 1,988 pixels.
- P0015_3a9bb2ae:163 — IoU 0.86397; old/new area 10,448 / 12,093 pixels.

Full diagnostics: `outputs/gaussian_hand/diffusion_bridge_v2/geometry_audit.json`.
This is geometry-only, **not** Gaussian RGB or diffusion condition validation.

## Still pending

1. Inspect raster support differences and resolve or explicitly characterize
   them while preserving the baseline's crop, masks and sample population.
2. When GPUs are available and work resumes, test single/joint-hand CUDA
   rendering, then render the 136 seen-eval source frames. Compare all 183
   conditions to the naive baseline, including alpha, occluders and wrists.
   The comparison/acceptance driver still needs to be completed.
3. Only after that review, export the full 136-sequence cache covering 393,684
   unique training frames, with blanks at nonrequested source frames to retain
   exact video alignment. This feeds the existing 524,917 adaptive crop records.
4. Create a separate diffusion config pointing to that verified cache. No new
   diffusion training config has been activated, and no training has started.

All new outputs belong under `outputs/gaussian_hand/diffusion_bridge_v2/`.
The old MANO render root, manifests, Gaussian checkpoints and diffusion baseline
remain unchanged.
