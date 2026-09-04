#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/shaoyu/miniconda3/envs/glove-hot3d/bin/accelerate}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$ROOT/outputs/hand_restoration/hugg_aria_aligned_sam_weight10_chunk5s_512_8gpu_batch256_resume10epochs/controlnet_final.pt}"
test -f "$RESUME_CHECKPOINT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

configs=(
  hugg_aria_lr_sweep_5e6_500steps.json
  hugg_aria_lr_sweep_1e5_500steps.json
  hugg_aria_lr_sweep_2e5_500steps.json
)
for name in "${configs[@]}"; do
  "$ACCELERATE_BIN" launch --num_processes 8 --num_machines 1 \
    --mixed_precision bf16 --dynamo_backend no \
    "$ROOT/train_hand_restorer.py" \
    --config "$ROOT/configs/hand_restoration/$name" \
    --resume "$RESUME_CHECKPOINT"
done
