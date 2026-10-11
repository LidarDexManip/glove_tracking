# Gaussian-overlay diffusion + AnimateDiff (inference-only)

This experiment is isolated from the frozen baseline/Gaussian comparisons and
does not modify existing checkpoints, manifests, archive-retention work, or web UI.

## Current delivery: showcase videos only

The requested deliverable is now a visually curated demo, not a benchmark or a
web inference application. Do not start a gallery server for this workflow.
The retained two-hand delivery is described below. The earlier single-hand
`outputs/hand_restoration/demo_delivery_30fps` contains two clean 20-second MP4s:
`hand_demo_full_view_30fps.mp4` (1408px original-view composites) and
`hand_demo_30fps.mp4` (512px generated hand close-ups), plus individual clips.
Four full five-second clips were selected (P0014, P0003, P0010, P0015), retaining
native 30 FPS and clockwise display orientation. There are no experiment labels,
metrics, side-by-side panels or interpolated frames. Export uses packet-copy
concatenation, with no additional GPU inference or re-encoding.

The demo gallery process on port 7863 was stopped. Forty audited 10-FPS MP4s
were moved out of the active demo directories into a recoverable `.trash`
directory; all 74 existing 30-FPS MP4s were verified unchanged by SHA-256.
See `demo_delivery_30fps/metadata/cleanup.json` for the exact recovery directory,
and its `manifest.json` for original paths and checksums. No training data,
checkpoints, manifests, or unrelated videos were removed. The historical
gallery URLs below are not running endpoints anymore.

## Method

### Two-hand presentation update

The old showcase processed one selected hand per clip; the other hand remained
original. The new delivery is
`outputs/hand_restoration/two_hand_demo_30fps/delivery/two_hand_demo_30fps.mp4`:
two curated five-second clips, 1920x1024, native 30 FPS. Its only panel titles
are **Gaussian-conditioned input** and **Restoration** (not "inpainted video").
Both hands have actual Gaussian conditions on every frame. Left and right are
restored independently using fixed, disjoint temporal crops, then both outputs
are pasted into the same original frame. Pixels outside the union of their edit
masks are asserted unchanged. Dense mask/Gaussian-coverage checks and per-hand
changed-pixel checks prevent silently displaying one untouched source hand.

The visually approved clips are P0014_f7ba43e0_both_001052 and
P0003_f4730606_both_001847. A P0012 candidate was excluded after visual review
because a hand approached the frame edge. The owned GPU 6 worker was stopped
after the two approved clips completed; the unused candidate was not inferred.
The final montage was assembled on CPU. No web service or extra training.

To reproduce in a new output directory, use `temporal_demos.render_two_hands`
with the saved `screening_all/selection.json`, `--clip` for each approved ID,
and an explicitly authorized GPU. Shared mode requires `--allow-shared-gpu`,
caps the allocator at 16 GiB, and does not modify other running jobs. The same
0.35 / context-24 / overlap-16 motion settings and nine-frame brightness
stabilization are retained.

Example (choose an idle, authorized GPU; the output directory must be new):

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  python -m temporal_demos.render_two_hands --gpu 0 \
  --selection outputs/hand_restoration/two_hand_demo_30fps/screening_all/selection.json \
  --clip P0014_f7ba43e0_both_001052 --clip P0003_f4730606_both_001847 \
  --output outputs/hand_restoration/NEW_TWO_HAND_DEMO
```

The sections below retain the earlier single-hand experiments for provenance;
their gallery helpers are optional offline tools, not an active web service.

- Reuses the Gaussian-overlay `controlnet_final.pt` (step 15380), frozen SD1.5
  components and the existing custom 5-channel **VAE latent + visible-hand mask**
  condition embedding. This is not a stock RGB ControlNet pipeline.
- Adds the cached `guoyww/animatediff-motion-adapter-v1-5-3` to the UNet with
  `UNetMotionModel.from_unet2d`; no optimizer or additional training.
- Motion strength 0 skips the temporal modules; 0.25 scales their residual
  contribution. It is **not** a change to the Gaussian ControlNet strength.
- Initial experiment at strength 1.0, under this sampler/conditioning/noise
  combination, produced severe hand artifacts. 0.25 was selected on one pilot;
  all six full clips then use that same setting, without per-clip seed search.
- Frames remain in native orientation for inference. The 90° clockwise rotation
  only affects video presentation.
- Uses one fixed crop per clip, retaining each frame's MANO component seeds and
  the original mask/wrist rules. New records store both original and fixed boxes.
- 512px, BF16, guidance=1, original DDPM schedule, 20 steps, shared initial noise,
  seed 7, 16-frame temporal windows with 8-frame overlap. Overlapping noise
  predictions are blended **at every denoising step** into one global trajectory;
  separately generated clips are not joined at hard boundaries.
- The off/on comparison shares all these settings. It is not a bitwise replay of
  the original single-frame FP32, adaptive-crop benchmark.
- Optional `--refine-strength .35 --scheduler ddim` initializes AnimateDiff from
  the Gaussian diffusion outputs (video-to-video); a pilot gave less temporal
  improvement than the direct low-motion setting, so it is not the demo default.

## Selection

CPU screening begins with complete 150-source-frame windows of a consistent
left/right training crop. Rejects excessive crop motion/expansion and small hands;
uses previous audit/appearance statistics for coarse ordering. Ten evenly spaced
frames per candidate are inspected with the actual condition builder, measuring
visible MANO/SAM overlap, crop-border contact, Gaussian color error excluding the
blanked wrist, mask stability and sharpness. Subject diversity is prioritized.

The first run found 62 coarse sequence/hand candidates, pixel-audited 29 windows,
and selected six subjects. Before inference all 50 sampled frames per clip are
checked again for visible-mask coverage. These are **quality proxies**, not
manually verified masks or proof of complete, unoccluded anatomy. This is a
curated demonstration batch, not an unbiased evaluation or a new training split.

## Historical single-hand run

From the repository root, activate the shared Conda environment:

```bash
conda activate glove-hot3d
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m temporal_demos.select_clips \
  --output outputs/hand_restoration/NEW_DEMO/screening
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  python -m temporal_demos.render_clips --gpu 0 --strength .25 \
  --selection outputs/hand_restoration/NEW_DEMO/screening/selection.json \
  --output outputs/hand_restoration/NEW_DEMO/videos
```

Output directories must not exist. GPU must be idle; the worker shares the
paired-video UI's per-GPU lock. No network downloads during inference. The static
gallery has no GPU access: choosing or playing videos cannot launch inference.
For remote access, forward port 7863 over SSH. Favorites are kept in the browser
and exported as JSON; there are no external services or uploads.

Videos: `review_all.mp4` concatenates comparisons; each clip has `comparison.mp4`,
`animatediff_crop.mp4`, `gaussian_only_crop.mp4`, and `animatediff_full.mp4`.
The full video changes only the selected hand's edit mask, not both hands.
The native source runs at 30 FPS; these previews sample every third frame and
play at 10 FPS without slowing time. No optical-flow interpolation is applied.

The completed first batch is in `outputs/hand_restoration/animatediff_demo_v1/videos`:
six clips × five seconds. Including loading and encoding, it took 228.9 seconds
on GPU 0 (B200), peaking at 5.87 GiB PyTorch allocations. The process exited and
released GPU memory. Three preceding short pilot jobs took approximately
23–25 seconds each. All 25 MP4 files are frame-count/duration checked.

At strength 0.25, the flow-compensated temporal residual proxy decreased by
13.7–23.6% across these six selected clips; five clips also increased hand PSNR.
The P0009 grasping clip lost 0.82 dB despite lower temporal error, so it is kept
with a warning rather than hidden from the comparison. Try P0014_9b7a0725 and
P0003_f4730606 first when selecting demos. These measurements do not establish
an overall model improvement: the sample is curated, and temporal error can
also benefit from oversmoothing.

## Training / default policy

### Native-FPS, brightness-stabilized demo (v2)

The v1 preview used 10 FPS; source videos are 30 FPS. `prepare_native_fps`
verifies each source with ffprobe and expands the exact same selected windows
to all 150 consecutive training-manifest frames. No repeated frames or optical
flow interpolation is used. Original checkpoints and the v1 outputs stay intact.

Pilot tests with deterministic DDIM and temporally shared DDPM step noise
reduced brightness metrics but produced unacceptable speckled hand textures,
especially with motion strength above 0.25. These remain experimental options,
**not recommended presets**. The selected v2 recipe keeps original DDPM,
motion strength 0.25, and 16-frame / overlap-8 context.

It adds explicit offline **brightness-only postprocessing**: measure mean
generated-minus-Gaussian-condition luminance inside the eroded visible hand,
smooth that scalar bias over a centered nine-frame window (0.3 s at 30 FPS),
and apply 90% of the difference, capped at 24/255. Apply the same offset to
each RGB channel with a five-pixel feather strictly inside the edit mask.
No spatial frame averaging, warping, target RGB, or new training is used.
This suppresses exposure pulsing, not all texture flicker or geometric errors;
it is not a learned temporal-consistency improvement. Real lighting changes
represented by the Gaussian condition are retained; fast changes missing from
that condition can still be attenuated. The centered filter needs future frames,
so this is an offline demo, not a zero-latency streaming method.

```bash
python -m temporal_demos.prepare_native_fps \
  --selection outputs/hand_restoration/animatediff_demo_v1/screening/selection.json \
  --output outputs/hand_restoration/NEW_30FPS/selection.json
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  python -m temporal_demos.render_clips --gpu 3 \
  --selection outputs/hand_restoration/NEW_30FPS/selection.json \
  --output outputs/hand_restoration/NEW_30FPS/videos \
  --strength .25 --reference-strength .25 --brightness-window 9 --brightness-strength .9
python -m temporal_demos.make_gallery outputs/hand_restoration/NEW_30FPS/videos
```

The reference column is **the old sampling recipe rerun at 30 FPS**, not the
old 10-FPS file. New and reference outputs share the exact same raw generation,
isolating the photometric postprocess. Raw reference PNGs and per-frame scalar
corrections are retained. `brightness_flicker_mae_255` measures absolute mean
luminance of the GT-flow-compensated temporal residual inside the common hand
mask. Lower is better, but must be checked alongside fidelity and visual artifacts.

Completed v2 outputs: `outputs/hand_restoration/animatediff_demo_v2_30fps/videos`.
Six 5-second clips, 900 unique source frames; all 25 MP4 files are encoded at
30 FPS. The batch took 429.2 seconds including loading/CPU encoding, with
5.87 GiB peak PyTorch allocations on GPU 3. Against the same raw 30-FPS outputs,
the brightness-flicker proxy decreased 80.4–84.6%; per-clip hand PSNR changes
ranged from -0.09 to +0.47 dB. These are selected-clip, postprocessing results,
not evidence that the pretrained motion adapter has learned hand dynamics.
The former gallery path was `/native30/`; the server is now stopped and the
10-FPS videos have been moved to the recoverable trash described above.

### Future training

Stronger inference preset (v3): motion strength 0.35,
context 24, overlap 16; keep original DDPM, 20 steps, native 30 FPS, nine-frame
brightness stabilization at 0.9. `bash temporal_demos/run_stronger_demo.sh GPU_ID`
runs a two-clip, 45-frame pilot. Review hand textures before explicitly launching
`bash temporal_demos/run_stronger_demo.sh GPU_ID full` for all six five-second clips.
By default the renderer refuses busy GPUs and never overwrites output directories.
Only after explicit authorization, append `shared` to the preset command to allow
sharing (for example, `bash temporal_demos/run_stronger_demo.sh 6 pilot shared`).
Shared mode caps the PyTorch allocator at 16 GiB and requires at least that much
free VRAM plus 4 GiB headroom before loading. CUDA context overhead is outside
that allocator cap; this does not reserve GPU compute or prevent other jobs
from expanding. Existing jobs are never stopped or reconfigured.
New results go to `animatediff_demo_v3_stronger`; the raw comparison column uses
the new stronger settings before brightness postprocessing, not the v2 preset.
Use the preserved v2 gallery for cross-preset comparison. Launching is explicit;
the preset does not queue or automatically retry jobs on occupied GPUs.

After both full batches finish, `python -m temporal_demos.compare_presets --previous
OLD_VIDEOS_DIR --current NEW_VIDEOS_DIR` creates synchronized old/new crop videos
on CPU. It verifies identical frame manifests, checkpoints and photometric
settings before comparison. Rebuild the gallery to expose the comparison button.

Completed v3: `outputs/hand_restoration/animatediff_demo_v3_stronger/videos`.
Its former gallery path was `/stronger30/`; no gallery server is running now.
The authorized shared GPU 6 pilot took 60.1 seconds and the six-clip full run
took 498.3 seconds, including CPU work. Peak PyTorch allocations were 5.87 GiB;
sampled process VRAM was about 8 GiB (16 GiB allocator cap). Both inference
processes exited; the pre-existing GPU 6 jobs were not stopped or reconfigured.
Six synchronized `preset_comparison.mp4` videos compare the preserved v2 outputs
against v3, both with identical brightness postprocessing. Average per-clip
temporal-proxy reduction was 12.0%, and mean PSNR change was +0.06 dB; individual
PSNR changes ranged from -0.70 to +0.85 dB. P0009 lost 0.70 dB and is flagged in
the gallery. These selected-clip proxy results do not prove perceptual improvement.

Pretrained AnimateDiff can be tried without training. For this custom restorer,
compatibility and quality must be validated; a plug-in is not guaranteed to
improve hand fidelity. Keep the original image-only path as the default for
controlled comparisons. The new video-demo command offers the tested 0.25 preset;
do not silently enable the adapter on all existing inference calls.

If future hand-domain adaptation is requested, a first **unbenchmarked budget**
for motion-attention LoRA is one B200, 16-frame 512px clips, BF16 + gradient
checkpointing, 2k–5k optimizer steps: roughly 1–4 GPU-hours and 24–60 GiB peak
memory. A 100-step forward/backward benchmark is needed to refine this estimate.
This excludes data preparation and engineering, is not a measured training run,
and no such training has been started. Retraining generic AnimateDiff from scratch
is a different, substantially larger project and is not required for this demo.

Primary references:

- [AnimateDiff official repository](https://github.com/guoyww/AnimateDiff)
- [Diffusers AnimateDiff / ControlNet documentation](https://huggingface.co/docs/diffusers/api/pipelines/animatediff)
- [Motion adapter v3](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3)
