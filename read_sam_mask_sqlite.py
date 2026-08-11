#!/usr/bin/env python3
"""Read or export an exact uint8 SAM hand label mask from masks.sqlite."""
from __future__ import annotations

import argparse
import json
import sqlite3
import zlib
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("frame", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database)
    metadata = {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key,value_json FROM metadata")
    }
    row = connection.execute(
        "SELECT timestamp_ns,labels_zlib,area_left,area_right FROM frames WHERE frame_index=?",
        (args.frame,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Frame {args.frame} is absent")
    timestamp, blob, area_left, area_right = row
    labels = np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(
        int(metadata["height"]), int(metadata["width"])
    )
    output = args.output or Path(f"mask_{args.frame:06d}.png")
    if not cv2.imwrite(str(output), labels):
        raise RuntimeError(f"Could not write {output}")
    print(
        json.dumps(
            {
                "frame": args.frame,
                "timestamp_ns": timestamp,
                "area_left": area_left,
                "area_right": area_right,
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
