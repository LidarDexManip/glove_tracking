# Four-GPU Gaussian restorer: same-data controlled run

The user explicitly authorized GPU 0–3 on 2026-09-11, superseding the earlier
eight-GPU-only execution plan. The original eight-GPU config, preparation
READY marker, caches, baseline checkpoint and manifests remain unchanged.

New config: `configs/hand_restoration/hugg_aria_gaussian_surface_v2_4gpu.json`.
New output: `outputs/hand_restoration/hugg_aria_gaussian_surface_v2_4gpu/`.

## Data confirmation

Actual dataset objects for baseline and Gaussian contain identical ordered
lists of 524,917 crop records, covering 393,684 source frames in 136 sequences.
Training, seen-eval and summary manifest paths and SHA256 hashes are identical.
There is no additional Gaussian-fit visibility/occlusion filtering in the
diffusion dataset. Gaussian fitting uses 54,468 hand-specific crops; that
separate filtering never changes the diffusion record population. Existing
baseline SAM visibility, wrist, crop and loss-mask rules are retained.

Only the RGB/native-alpha rendering input is replaced by the new subject-bound
Gaussian renderer. This does not claim a pure color-only ablation with Gaussian
opacity held fixed. The prepared 183-sample condition QA and all-image archive
audit continue to apply to the unchanged data config.

## Execution change

| Setting | Frozen baseline | Requested run |
| --- | --- | --- |
| GPUs | 8 | 4, physical 0–3 |
| Per-GPU training batch | 32 | 64 |
| Accumulation | 1 | 1 |
| Global training batch | 256 | 256 |
| Per-GPU validation batch | 8 | 16 |
| Global validation batch | 64 | 64 |
| Optimizer steps | 15,380 | 15,380 |
| LR / seed | 5e-6 / 7 | 5e-6 / 7 |

Actual Accelerate BatchSamplerShard index grouping was checked for the complete
training and numeric-validation populations: global index groups match exactly,
including last-batch padding (2,051 training updates per epoch; three validation
global batches). Sampler source and seed are unchanged. Different rank-local
RNG assignment, CUDA kernels and gradient reduction order mean this is **not
an identical eight-rank numerical replay**. Paired visual evaluation uses the
same preselected images and inference seeds; validation-noise realization need
not be bitwise identical across rank layouts.

The initial estimate is 10–12 hours on four B200 GPUs, based on the baseline's
5.40 hours on eight GPUs. Baseline reserved memory was approximately 61.7 GiB
per rank at batch32; doubling batch is expected to fit each 179 GiB B200.
The estimate and memory fit must be checked against real training startup.

Launch (explicit user training authorization required):

```bash
source /data/szeng87/venvs/gaussian-hand/bin/activate
python -m gaussian_hand.launch_diffusion_strict --four-gpu --execute
```

The four-GPU gate allows only this named execution layout and binds it to the
original audited preparation config SHA256. It does not relax data, model,
optimizer or budget checks, and does not overwrite the original run.
