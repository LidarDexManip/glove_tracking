# Fourteen-session Gaussian overlay cache

The rendering-only run uses the locked inventory in
`outputs/hand_restoration/overfit14_preparation_20261008/input_inventory.json`.
It contains all 6,300 frames from 14 task sessions and the SHA256 hashes of the
latest left/right `stabilize_4gpu_20261008T201709Z/*_formal/step_070000.pt` models.
No source frame is quality-filtered and no held-out split is created.

The output root is
`outputs/hand_restoration/overfit14_gaussian_overlay_sam31_v1` on the server.
The supervisor launches only render workers on GPUs 0–3. It does not launch
diffusion training. Completion requires `READY.json`, followed by `AUDIT.json`
from `python -m gaussian_glove.audit_task_overlays --output OUTPUT_ROOT`.

Each task/session folder contains:

- `render.mkv`: native 1920×1080 jointly depth-sorted Gaussian premultiplied RGB,
  quantized to uint8, FFV1 lossless encoding, on black.
- `alpha.mkv`: matching uint8 opacity, FFV1 lossless.
- `overlay.mp4`: Gaussian render composited over source RGB only in the supplied
  SAM3.1 union mask. H.264 visualization; not used as a training source.
- `timestamps_s.npy`: original per-frame timestamps. Video playback uses nominal
  FPS, so frame indices plus this array are authoritative for temporal alignment.
- `crops.tar`: lossless 512×512 PNG pairs, nine members per crop: target, render,
  alpha, condition, overlay, mano_mask, visible_mask, edit_mask, sam_mask.
- `frames.jsonl`: source frame, timestamp, crop rectangle, mode, archive key.
- `contact_sheet.jpg`: fixed evenly spaced source-frame previews.
- `_SUCCESS.json`: full frame/crop counts, hashes, and numerical invariants.

The native render is not clipped to MANO. Both hands are rasterized in the same
2DGS call. SAM masks supply object-occlusion protection for overlays; this is not
object-depth rendering. The native overlay uses all SAM-visible Gaussian pixels.
The crop condition follows the bare-hand code: MANO/SAM intersection for visible
conditioning, MANO loss support, 3-pixel dilation, palm-axis wrist sleeve, 1.5×
single-hand / 1.2× close-hand adaptive crops. These crop overlays consequently
have narrower support than native SAM-only overlays. Palm anchors are estimated
from the posed MANO proximal joints. Empty-MANO frames fall back to a full-frame
crop instead of being discarded. Conditions/targets are cached as uint8 PNGs.

`train_manifest.jsonl` includes all crops; `seen_eval_manifest.jsonl` uses eight
evenly spaced source frames per session, including all crops of those frames.
It measures training reconstruction, never generalization.

`task_overlay_dataset.py` reads the prepared conditions directly.
`train_task_restorer.py` adapts the existing trainer without editing it and
preserves empty negative strings in its text embedding method. A configuration
is prepared under `gaussian_glove/configs/overfit14_gaussian_sam31_v1.json`, but
the user's current instruction is to stop after rendering: do not run training.
