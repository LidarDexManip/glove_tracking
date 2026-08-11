# HOT3D Hand Restoration Workspace

This repository contains the reproducible training and inference workspace for
HOT3D hand restoration. It covers canonical-camera dataset preparation, MANO
and Gaussian hand conditions, ControlNet training, evaluation, and browser
inference.

Large datasets, licensed MANO files, downloaded model weights, and experiment
outputs are intentionally excluded from Git.

## Synchronization boundary

Use Git for source code, JSON/YAML experiment configuration, documentation,
tests, and pinned submodule revisions. Keep these machine-local:

- `data/` (currently about 234 GB): raw and derived datasets.
- `outputs/` (currently about 46 GB): checkpoints, previews, and run logs.
- `mano_v1_2/` and `model/`: licensed or locally generated assets.
- Hugging Face, pip, and Conda caches.

This keeps a normal clone small and avoids publishing datasets, credentials,
or licensed models. Read [`docs/SYNCING.md`](docs/SYNCING.md) before moving work
to another machine.

## Repository map

- `hand_restoration/`: datasets, conditions, diffusion, inference, and
  visualization modules.
- `configs/hand_restoration/`: tracked experiment definitions and split files.
- `tests/`: this repository's unit tests.
- `hot3d/`: pinned HOT3D Git submodule.
- `docs/SCRIPTS.md`: categorized index of top-level command-line tools.
- `README_hand_restoration.md`: implementation and experiment details.
- `data/README.md`, `outputs/README.md`: local storage conventions.

Secondary command-line tools are grouped under `scripts/` by workflow. Run
Python tools with `python -m scripts.<category>.<module>` from the repository root.

## Clone

```bash
git clone --recurse-submodules <YOUR_REPOSITORY_URL> glove_tracking
cd glove_tracking
```

For an existing clone:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Always run `git status --short` before changing machines. Untracked source
files do not exist elsewhere until they are committed and pushed.

## Environments

The main Python 3.10 environment contains training, inference, web UI, and test
dependencies:

```bash
conda env create -f environment.glove-hot3d.yml
conda activate glove-hot3d
python -m pip install -r requirements.glove-hot3d-gpu.txt
python -m pip install -r requirements.glove-hot3d.txt
python -m pip install -r requirements.web-ui.txt
```

`requirements.glove-hot3d-gpu.txt` pins the CUDA 12.8 PyTorch wheels. The
server driver must support CUDA 12.8; no separate CUDA Toolkit is required.

The first diffusion run downloads `runwayml/stable-diffusion-v1-5`. Point
`HF_HOME` at persistent storage when home directories are ephemeral:

```bash
export HF_HOME=/path/to/persistent/huggingface-cache
```

## External assets

The small 3000-step example expects:

```text
data/
  train_quest3/
    clip-000000.tar
mano_v1_2/
  models/
    MANO_LEFT.pkl
    MANO_RIGHT.pkl
```

Copy the original HOT3D clip tar without extracting it. Download MANO v1.2
from the official MANO site after accepting its license; these files must not
be committed or redistributed.

## Verify

`pytest.ini` limits collection to this repository and excludes tests belonging
to the upstream submodules.

```bash
conda activate glove-hot3d
pytest
python check_training_setup.py \
  --config configs/hand_restoration/tiny_overfit_shaded_3000.json \
  --require-cuda
python smoke_test_hand_restoration.py \
  --config configs/hand_restoration/tiny_overfit_shaded_3000.json
```

## Train and inspect

```bash
accelerate launch --num_processes 1 train_hand_restorer.py \
  --config configs/hand_restoration/tiny_overfit_shaded_3000.json
```

The example writes checkpoints and `training_log.csv` under
`outputs/hand_restoration/tiny_overfit_shaded_frames10_20_3000/`. Resume with:

```bash
accelerate launch --num_processes 1 train_hand_restorer.py \
  --config configs/hand_restoration/tiny_overfit_shaded_3000.json \
  --resume outputs/hand_restoration/tiny_overfit_shaded_frames10_20_3000/controlnet_step1500.pt
```

Useful inspection entry points:

```bash
python -m scripts.evaluation.compare_hand_restoration_checkpoints
python hand_restoration_web.py --help
python -m scripts.evaluation.evaluate_hand_restorer --help
```

Relative paths are resolved from the repository root. Run commands there.
