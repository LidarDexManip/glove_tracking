# Gaussian visualization archive retention

The existing four-GPU Gaussian final model and its training inputs are retained.
The original full image export was a visualization/audit artifact, not the
floating-point training input. Source videos, Gaussian models, aligned RGB/alpha
videos, masks, wrists, complete manifests/indices and historical audits remain.

After compaction, each sequence under
`outputs/gaussian_hand/diffusion_strict_v2/image_sequences/` keeps:

- `samples.tar`: original encoded condition PNGs and overlay JPEGs, copied without
  re-encoding and verified against the original member bytes;
- `samples.jsonl`: selected record identities;
- original `index.jsonl` and `_SUCCESS.json`: complete historical export metadata.

Selection includes the five existing 150-frame demo clips, matching frozen
seen-evaluation records, and the first/middle/last record of every sequence.
The complete historical `READY.json`, `dataset_audit.json`, `ASSET_LOCK.json`
and preparation reports are immutable historical evidence. They describe the
original export, not the current number of archived images.

`ARCHIVE_RETENTION.json` binds the retained files by SHA256 to those historical
records; `ARCHIVE_CLEANUP.json` records deletion of original full archives.
Training readiness accepts intentionally compacted archives only with a valid
retention manifest; source assets, configuration, environment and model checks
remain enforced. Missing or modified samples still fail validation.

Run the new sample audit without changing historical evidence:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  conda run -n glove-hot3d python \
  -m gaussian_hand.audit_prepared_diffusion_dataset --retained
```

This checks every retained header/index and compares live conditions on all nine
subjects plus start/middle/end samples of the demo clips. Results are written
to `retained_sample_audit.json`. It is not a full re-audit of deleted images.
The old full audit fails explicitly when full archives are absent. The training
data demo automatically uses retained samples for its five fixed clips.

For future detail tuning or AnimateDiff experiments, create separate experiment
configs and any new temporal crop/clip manifests. The old strict comparison gate
intentionally still rejects uncontrolled changes. The preserved source videos,
frame indices and timestamps support new preparation; old crops do not guarantee
stable hand identity/cropping across time.

To recover a full visualization archive, run the exporter for the original
sequence manifest and original strict config with a NEW output directory:

```bash
python -m gaussian_hand.export_condition_sequences \
  --config configs/hand_restoration/hugg_aria_gaussian_surface_strict_v2.json \
  --manifest outputs/gaussian_hand/diffusion_strict_v2/manifests/SEQUENCE.jsonl \
  --output /data/szeng87/glove_tracking_data/reexports/SEQUENCE
```

Replace `SEQUENCE` with the desired sequence ID. Pixel reproduction depends on
the retained source assets and frozen decoding/conditioning environment; tar
container bytes need not match if library versions change. Do not replace the
historical completion markers or silently regenerate all archives in place.
There is no full archive backup after deletion.
