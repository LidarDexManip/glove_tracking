# Gaussian-overlay diffusion + AnimateDiff (inference-only)

This experiment is isolated from the frozen baseline/Gaussian comparisons and
does not modify existing checkpoints, manifests, archive-retention work, or web UI.

## Method

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

## Run

From the repository root, activate the existing gaussian-hand environment:

```bash
source /data/szeng87/venvs/gaussian-hand/bin/activate
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m temporal_demos.select_clips \
  --output outputs/hand_restoration/NEW_DEMO/screening
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  python -m temporal_demos.render_clips --gpu 0 --strength .25 \
  --selection outputs/hand_restoration/NEW_DEMO/screening/selection.json \
  --output outputs/hand_restoration/NEW_DEMO/videos
python -m temporal_demos.make_gallery outputs/hand_restoration/NEW_DEMO/videos
python -m http.server 7863 --bind 127.0.0.1 \
  --directory outputs/hand_restoration/NEW_DEMO/videos
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
