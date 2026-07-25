#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

command -v hf >/dev/null 2>&1 || {
  echo "ERROR: 'hf' is unavailable. Activate glove-hot3d and run:"
  echo "  python -m pip install -U huggingface_hub hf_xet"
  exit 1
}

mkdir -p data
available_kib="$(df -Pk data | awk 'NR == 2 {print $4}')"
recommended_kib=$((150 * 1024 * 1024))
if (( available_kib < recommended_kib )); then
  echo "WARNING: less than 150 GiB is currently available."
  echo "A fresh download may run out of space; a partial download may need less."
  df -h data
fi

echo "[1/3] Downloading the complete HOT3D-Clips train_quest3 folder and metadata."
hf download bop-benchmark/hot3d \
  --repo-type dataset \
  --include "train_quest3/*" \
  --include "clip_definitions.json" \
  --include "clip_splits.json" \
  --local-dir data

echo "[2/3] Checking downloaded paths."
test -s data/clip_definitions.json
clip_count="$(find data/train_quest3 -maxdepth 1 -type f -name 'clip-*.tar' | wc -l)"
if (( clip_count == 0 )); then
  echo "ERROR: no clip tar files found directly under data/train_quest3/."
  echo "Check for an accidentally nested data/train_quest3/train_quest3/ directory."
  exit 1
fi
echo "Found $clip_count Quest3 clip tar files."

echo "[3/3] Building the deterministic sequence-disjoint split."
python build_hot3d_sequence_split.py \
  --clip-definitions data/clip_definitions.json \
  --clips-dir data/train_quest3 \
  --output configs/hand_restoration/splits/train_quest3_sequence_seed7.json \
  --device Quest3 \
  --holdout-fraction 0.2 \
  --seed 7 \
  --require-existing

echo "Download and split preparation completed."
echo "Next, follow HEADLESS_TRAIN_QUEST3.md to preprocess and verify the derived dataset."
