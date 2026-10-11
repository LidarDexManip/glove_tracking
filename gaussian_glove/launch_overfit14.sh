#!/usr/bin/env bash
set -euo pipefail
cd /home/szeng87/glove_tracking
OUT=outputs/hand_restoration/overfit14_gaussian_sam31_v1
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_NVLS_ENABLE=0
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PATH=/home/szeng87/miniconda3/envs/glove-hot3d/bin:$PATH
printf '%s\n' 'User authorized GPUs 0-3 for training to overfit on 2026-10-08; supersedes prior render-only stop for this training run.' > "$OUT/TRAINING_AUTHORIZATION.txt"
set +e
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 --module gaussian_glove.train_task_restorer --config gaussian_glove/configs/overfit14_gaussian_sam31_v1.json > "$OUT/train.log" 2>&1
result=$?
printf '%s\n' "$result" > "$OUT/exit_code.txt"
exit "$result"


