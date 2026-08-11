# External HOT3D data

Dataset files are intentionally excluded from Git. They must be copied between
machines separately; Git will only transfer this README.

For the current 3000-step tiny-overfit experiment, place the original archive
at:

```text
data/train_quest3/clip-000000.tar
```

Do not extract the tar. The dataset loader reads it directly. Other raw clips
belong in the same `data/train_quest3/` directory, while processed assets
belong under `data/train_quest3_processed/`.

The current workspace also uses these local-only areas:

```text
data/raw/       downloaded source datasets
data/derived/   generated training shards and manifests
data/.cache/    disposable dataset caches
```

For a non-destructive machine-to-machine copy, run from the destination repo:

```bash
rsync -a --partial --info=progress2 \
  USER@SOURCE_HOST:/path/to/glove_tracking/data/ ./data/
```

This intentionally omits `--delete`: an incomplete source cannot erase data on
the destination. See `docs/SYNCING.md` for the full workflow.

Never commit `Hot3DAria_download_urls.json`; its signed download URLs are
account-specific and may be sensitive.
