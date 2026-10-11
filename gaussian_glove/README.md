# Gaussian Glove

Train one appearance model per tracking glove from the synchronized Hugging Face
`LIDAR-GT/ego-clinical-task` delivery. All accepted frames train; there is no
validation split. Object-interaction and retargeting sessions are not selected.

## Prepare

```bash
conda activate glove-hot3d
python -m gaussian_glove.sync_dataset
CUDA_VISIBLE_DEVICES='' python -m gaussian_glove.prepare_dataset
```

Synchronization pins an HF commit, verifies every file against HF hashes, and
points `data/ego-clinical-task` at that immutable snapshot. `SOURCE.json` records
the revision and payload hashes. Preparation consumes only this source, including
SAM 3.1 masks and `quality_exceptions.csv`; it never runs segmentation or falls
back to locally produced masks. Session 040 is withdrawn and explicitly rejected.

Outputs are `gaussian_glove/data/processed_sam31_offset/{left,right}`. Each side
contains 512-pixel RGB crops, SAM masks, supervision regions, `frames.jsonl`,
`summary.json`, `rejected_frames.jsonl`, and a contact sheet. Rejected frames do
not appear in the training manifest. The loader requires a complete preparation
record and an unchanged quality-passing manifest. Use a fresh output directory
for rebuilding; completed inputs are not silently overwritten.

Quality gates reject reviewed invalid/uncertain annotations, empty masks,
out-of-view hands, severe MANO/mask disagreement, fragmentation, excessive holes,
large isolated mask-area jumps and cross-hand mask overlaps. Thresholds live in
`quality.py` and are copied into each summary. Automated gates are reproducible
proxies, not a guarantee of perfect annotations; review the contact sheets.

## Geometry and training

```bash
CUDA_VISIBLE_DEVICES=<allocated_gpu> python -m gaussian_hand.train \
  --config gaussian_glove/configs/black_glove_left_sam31_offset_1gpu.json \
  --output gaussian_glove/outputs/black_glove_left_sam31_offset
```

Use the right-hand config for the other glove. Coordinate GPU/time allocation
before formal training; start with a short resource measurement on one GPU.
The defaults retain 120,000 steps, no hard Gaussian count cap and 8% growth per
adaptive event. Training diagnostics use training frames, not a held-out split.

Exact 45D MANO axis-angle poses and a fixed mean shape per hand are retained.
Canonical mesh corrections and per-frame pose corrections are frozen. Each
Gaussian learns a nonnegative displacement along its deformed triangle normal,
bounded to 8 mm (an initial engineering bound, not measured physical thickness).
A small offset penalty discourages unnecessary inflation. Local surfel sizes
remain bounded, but rendering is not clipped to the MANO silhouette. The real
SAM glove boundary can therefore supervise finger thickness. Glove pixels far
outside the bounded surface neighborhood, primarily the cuff, are ignored by
the loss; this setup does not reconstruct the cuff. No delivery mask is changed.

Existing checkpoints and their saved configs remain historical artifacts.
Their strict-surface behavior remains supported. Old SAM2 presets have been
removed from the active config directory. A changed source or geometry contract
requires a new run; old checkpoints cannot be resumed against these new inputs.
