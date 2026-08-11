#!/usr/bin/env python3
"""Print the available checkpoint with the lowest logged validation loss."""
import csv
import sys
from pathlib import Path

run = Path(sys.argv[1])
points = []
with (run / "training_log.csv").open(newline="") as stream:
    for row in csv.DictReader(stream):
        if row.get("validation_loss"):
            step = int(row["total_step"])
            checkpoint = run / f"controlnet_step{step:06d}.pt"
            if checkpoint.is_file():
                points.append((float(row["validation_loss"]), step, checkpoint))
if points:
    loss, step, checkpoint = min(points)
    print(checkpoint)
elif (run / "controlnet_final.pt").is_file():
    print(run / "controlnet_final.pt")
else:
    checkpoints = sorted(run.glob("controlnet_step*.pt"))
    if not checkpoints:
        raise SystemExit(f"No checkpoint in {run}")
    print(checkpoints[-1])
