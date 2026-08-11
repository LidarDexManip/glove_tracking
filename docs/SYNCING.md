# Synchronizing this workspace

The repository has two different synchronization channels. Do not mix them.

## 1. Code and experiment definitions: Git

Git tracks source code, tests, docs, small JSON/YAML files, and the pinned HOT3D submodule revision. A new machine starts with:

```bash
git clone --recurse-submodules <repository-url> glove_tracking
cd glove_tracking
conda env create -f environment.glove-hot3d.yml
conda activate glove-hot3d
python -m pip install -r requirements.glove-hot3d-gpu.txt
python -m pip install -r requirements.glove-hot3d.txt
python -m pip install -r requirements.web-ui.txt
pytest
```

If the repository already exists:

```bash
git status --short
git pull --rebase
git submodule sync --recursive
git submodule update --init --recursive
```

Commit and push before changing machines. On concurrent work, use a branch per
task instead of making unrelated changes directly on `main`.

## 2. Datasets and generated artifacts: rsync or shared storage

The following paths are deliberately ignored by Git:

| Path | Purpose | Recommended handling |
| --- | --- | --- |
| `data/` | Raw and derived datasets | rsync from one authoritative machine |
| `outputs/` | Checkpoints and run artifacts | Copy selected run directories only |
| `mano_v1_2/` | Licensed MANO assets | Restore from the licensed source |
| `model/` | Local glove/model assets | Private storage or rsync |

Example: copy data into a fresh clone without deleting destination files:

```bash
rsync -a --partial --info=progress2 \
  USER@SOURCE_HOST:/path/to/glove_tracking/data/ ./data/
```

Example: copy one training run:

```bash
rsync -a --partial --info=progress2 \
  USER@SOURCE_HOST:/path/to/glove_tracking/outputs/hand_restoration/RUN_NAME/ \
  ./outputs/hand_restoration/RUN_NAME/
```

Do not add `--delete` until both paths have been checked manually. Do not put
signed HOT3D download manifests, MANO model files, credentials, or `.env` files
in Git.

## Before leaving a machine

```bash
git status --short
git diff --check
git submodule status
```

The working tree should contain no untracked source/config files. Large ignored
data may remain local and will not appear in ordinary `git status` output.
