# Headless server: complete HOT3D-Clips `train_quest3`

This workflow downloads the complete Hugging Face `train_quest3` clip folder,
downloads its official metadata, and creates the sequence-disjoint split used
by this repository.

The download is currently about 131 GB. Keep at least 180-200 GB free for the
raw download, and additional space for the derived training dataset.

## One-command download and split preparation

Clone or update the repository, activate the environment, and run:

```bash
git pull
chmod +x download_and_prepare_train_quest3.sh
tmux new-session -d -s hot3d-download \
  "bash -lc 'cd /home/shaoyu/glove_tracking && conda run --no-capture-output -n glove-hot3d ./download_and_prepare_train_quest3.sh 2>&1 | tee hot3d-download.log'"
```

Change `/home/shaoyu/glove_tracking` if the repository is elsewhere. Follow progress:

```bash
tail -f /home/shaoyu/glove_tracking/hot3d-download.log
```

The Hugging Face downloader is resumable. If the server or process stops,
launch the same command again; completed files are skipped.

If `hf` is unavailable, install the downloader once:

```bash
python -m pip install -U huggingface_hub hf_xet
hf auth login
```

## Why the old split command failed

The official `clip_definitions.json` currently contains `sequence_id` and
`device`, but does not contain a separate `participant_id`. Older versions of
`build_hot3d_sequence_split.py` required both fields and therefore extracted
zero records before reporting `No matching clip definitions found`.

The current script derives `participant_id` from a sequence ID such as
`P0002_210fc0da` and supports the official metadata directly.

Expected paths:

```text
data/clip_definitions.json
data/clip_splits.json
data/train_quest3/clip-000000.tar
data/train_quest3/clip-......tar
```

Do not use `data/train_quest3/train_quest3/`; that indicates the wrong
`--local-dir` was used.

## Preprocess the downloaded clips

```bash
tmux new-session -d -s hot3d-preprocess \
  "bash -lc 'cd /home/shaoyu/glove_tracking && conda run --no-capture-output -n glove-hot3d python preprocess_hot3d_c1_shards.py --split-json configs/hand_restoration/splits/train_quest3_sequence_seed7.json --output-dir data/derived/train_quest3_c1 --clips-per-shard 8 --min-mask-pixels 64 --jpeg-quality 95 2>&1 | tee hot3d-preprocess.log'"
```

Follow progress:

```bash
tail -f /home/shaoyu/glove_tracking/hot3d-preprocess.log
```

Preprocessing is resumable. A completed shard and its sidecars are validated
and skipped when the command is restarted.

## Verify the derived dataset

```bash
python verify_hot3d_derived.py \
  --dataset-dir data/derived/train_quest3_c1 \
  --split-json configs/hand_restoration/splits/train_quest3_sequence_seed7.json \
  --decode-samples 500
```

Do not delete raw tar files until verification passes, the derived dataset is
backed up, and smoke training has read both train and holdout successfully.
