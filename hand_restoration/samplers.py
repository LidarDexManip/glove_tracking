from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class TemporalChunkShuffleSampler(Sampler[int]):
    """Shuffle short temporal chunks while preserving order inside each chunk."""

    def __init__(
        self,
        records: Sequence[dict],
        chunk_frames: int,
        seed: int = 0,
    ) -> None:
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
        for dataset_index, record in enumerate(records):
            sequence = str(record["sequence_id"])
            frame = int(record["frame_index"])
            grouped[(sequence, frame // int(chunk_frames))].append(
                (frame, dataset_index)
            )
        self.chunks = [
            [dataset_index for _, dataset_index in sorted(items)]
            for _, items in sorted(grouped.items())
        ]
        if sum(map(len, self.chunks)) != len(records):
            raise AssertionError("Temporal chunks do not cover the dataset exactly once")

    def __len__(self) -> int:
        return sum(map(len, self.chunks))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        order = list(range(len(self.chunks)))
        random.Random(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        for chunk_index in order:
            yield from self.chunks[chunk_index]
