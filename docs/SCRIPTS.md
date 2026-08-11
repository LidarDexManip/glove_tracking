# Utility scripts

The repository root contains only the primary training/inference entry points
and helper modules imported by other code. Secondary command-line tools are
grouped here by workflow.

Run Python tools from the repository root with module syntax:

```bash
python -m scripts.data.prepare_hot3d_clips --help
python -m scripts.evaluation.evaluate_hand_restorer --help
python -m scripts.geometry.export_mano_mean_pose --help
```

Module syntax is required because tools import packages and helper modules from
the repository root.

## `scripts/data/`

- `build_hot3d_sequence_split.py`: deterministic sequence-disjoint split.
- `download_hot3d_clip.py`: download one HOT3D clip.
- `prepare_hot3d_clips.py`: download and validate a fixed clip set.
- `preprocess_hot3d_c1_shards.py`: build canonical C1 training shards.
- `verify_hot3d_derived.py`: validate shards, manifests, and split isolation.
- `export_hot3d_clip_preview.py`: export clip previews.
- `export_hot3d_clip_undistorted.py`: export canonical-camera video.
- `export_hot3d_object_masks.py`: export canonical-camera object masks.
- `download_and_prepare_train_quest3.sh`: headless Quest 3 setup launcher.

## `scripts/evaluation/`

- `evaluate_hand_restorer.py`: deterministic holdout evaluation.
- `compare_hand_restoration_checkpoints.py`: interactive checkpoint comparison.
- `debug_hand_restoration_samples.py`: save condition-building debug samples.
- `infer_hand_restorer_animatediff.py`: clip experiment with AnimateDiff.
- `plot_training_curves.py`: plot `training_log.csv` metrics.
- `select_best_hand_checkpoint.py`: select the lowest validation-loss checkpoint.

## `scripts/geometry/`

- `build_glove_hot3d_rig.py`, `transfer_glove_weights.py`: glove rig creation.
- `fit_mano_betas_to_glove.py`: fit MANO shape parameters.
- `export_mano_canonical.py`, `export_mano_mean_pose.py`: MANO exports.
- `export_hot3d_mano_sequence.py`, `export_hot3d_glove_sequence_torch.py`:
  sequence geometry exports.
- `render_glove_c1_overlay.py`, `composite_glove_rgba_sequence.py`: rendering
  and compositing helpers.
- `export_hot3d_clip_preview.py`, `export_hot3d_clip_undistorted.py`, and
  `export_hot3d_object_masks.py` live under `scripts/data/` because they operate
  on dataset camera streams.
- `view_glove_mano_correspondence_tk.py`: correspondence inspector.
- `blender_import_hot3d.py`: Blender import helper.
