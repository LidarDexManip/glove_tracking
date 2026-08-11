#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${CONDA_BASE:-$HOME/miniconda3}/envs/glove-hot3d/bin/python"

cd "$SCRIPT_DIR" || exit 1
mkdir -p outputs/aria_video_sam_preprocess

pids=()
status=0
for gpu in 1 2 3 4; do
  worker=$((gpu - 1))
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    preprocess_aria_video_sam_dataset.py \
    --worker-index "$worker" --num-workers 4 --device cuda:0 \
    >"outputs/aria_video_sam_preprocess/worker_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "One or more preprocessing workers failed; not finalizing." >&2
  exit "$status"
fi

"$PYTHON" \
  preprocess_aria_video_sam_dataset.py --finalize \
  >outputs/aria_video_sam_preprocess/finalize.log 2>&1
