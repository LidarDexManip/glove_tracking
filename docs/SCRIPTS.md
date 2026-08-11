# Script index

Top-level scripts remain at the repository root because several import local
modules by filename. Moving them into folders would change Python import
behavior. This index groups them by purpose instead.

## Training, evaluation, and inference

- `train_hand_restorer.py`: train the ControlNet hand-restoration model.
- `evaluate_hand_restorer.py`: deterministic holdout evaluation.
- `infer_hand_restorer.py`: single-frame checkpoint inference.
- `infer_hand_restorer_animatediff.py`: clip comparison with AnimateDiff.
- `hand_restoration_web.py`: browser inference UI.
- `compare_hand_restoration_checkpoints.py`: interactive checkpoint comparison.
- `select_best_hand_checkpoint.py`: choose the lowest validation-loss checkpoint.
- `plot_training_curves.py`: plot metrics from `training_log.csv`.

## Dataset preparation

- `prepare_hot3d_clips.py`: download and validate fixed HOT3D clips.
- `download_hot3d_clip.py`: download one HOT3D clip.
- `build_hot3d_sequence_split.py`: deterministic sequence-disjoint split.
- `preprocess_hot3d_c1_shards.py`: build canonical C1 training shards.
- `verify_hot3d_derived.py`: validate shards, manifests, and split isolation.
- `aria_mano_dataset_utils.py`: shared Aria dataset helpers.
- `pilot_aria_sam2_video_tracking.py`: one-sequence SAM2 tracking pilot.
- `preprocess_aria_video_sam_dataset.py`: full Aria/SAM2 preprocessing pipeline.
- `run_aria_video_sam_preprocess.sh`: multi-GPU preprocessing launcher.

## Diagnostics and visualization

- `check_training_setup.py`: dependency, path, submodule, and CUDA preflight.
- `smoke_test_hand_restoration.py`: end-to-end preprocessing smoke test.
- `debug_hand_restoration_samples.py`: save condition-building debug samples.
- `render_glove_c1_overlay.py`: render glove overlays in the canonical camera.
- `composite_glove_rgba_sequence.py`: composite rendered sequences.
- `view_glove_mano_correspondence_tk.py`: inspect glove/MANO correspondences.

## Geometry and asset conversion

- `build_glove_hot3d_rig.py`, `transfer_glove_weights.py`: glove rig creation.
- `fit_mano_betas_to_glove.py`: fit MANO shape parameters.
- `export_mano_canonical.py`, `export_mano_mean_pose.py`: MANO exports.
- `export_hot3d_mano_sequence.py`, `export_hot3d_glove_sequence_torch.py`:
  sequence geometry exports.
- `export_hot3d_clip_preview.py`, `export_hot3d_clip_undistorted.py`,
  `export_hot3d_object_masks.py`: camera-space exports and previews.
- `blender_import_hot3d.py`: Blender import helper.

Run scripts from the repository root unless their help text says otherwise.
