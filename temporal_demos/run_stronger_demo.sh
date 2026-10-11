#!/usr/bin/env bash
# Explicit launch only; sharing requires the optional third argument. Pilot must be reviewed
# visually before invoking full. Both modes preserve all prior demo outputs.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 3 || ! "$1" =~ ^[0-7]$ ]]; then
    echo "Usage: bash temporal_demos/run_stronger_demo.sh GPU_ID [pilot|full] [shared]" >&2
    exit 2
fi
demo_gpu="$1"
demo_mode="${2:-pilot}"
case "$demo_mode" in
    pilot) demo_args=(--count 2 --pilot-frames 45); demo_dir=pilot ;;
    full) demo_args=(--count 6); demo_dir=videos ;;
    *) echo "Mode must be pilot or full" >&2; exit 2 ;;
esac
demo_shared=()
if [[ $# -eq 3 ]]; then
    if [[ "$3" != shared ]]; then echo 'Third argument must be shared' >&2; exit 2; fi
    demo_shared=(--allow-shared-gpu --memory-limit-gib 16)
fi
cd /home/szeng87/glove_tracking
source /home/shaoyu/miniconda3/etc/profile.d/conda.sh
conda activate glove-hot3d
export CUDA_VISIBLE_DEVICES="$demo_gpu"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python -u -m temporal_demos.render_clips \
    --gpu "$demo_gpu" \
    --selection outputs/hand_restoration/animatediff_demo_v2_30fps/selection.json \
    --output "outputs/hand_restoration/animatediff_demo_v3_stronger/$demo_dir" \
    --strength .35 --reference-strength .35 \
    --context 24 --overlap 16 --reference-overlap 16 \
    --scheduler original --steps 20 \
    --brightness-window 9 --brightness-strength .9 \
    "${demo_args[@]}" "${demo_shared[@]}"
