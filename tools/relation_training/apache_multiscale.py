from __future__ import annotations

from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


def scale_ladder(
    base: int,
    lo: float,
    hi: float,
    n: int,
    *,
    patch: int = 16,
) -> list[int]:
    """Apache-reference square multi-scale ladder."""
    if base <= 0 or patch <= 0:
        raise ValueError("base and patch must be positive")
    if n < 1:
        raise ValueError("multi-scale rung count must be >= 1")
    if not (0.0 < lo <= hi):
        raise ValueError("multi-scale range must satisfy 0 < lo <= hi")
    values = (
        [(lo + hi) * 0.5]
        if n == 1
        else [
            lo + (hi - lo) * index / (n - 1)
            for index in range(n)
        ]
    )
    return sorted(
        {
            max(
                patch,
                int(round(base * scale / patch)) * patch,
            )
            for scale in values
        }
    )


class MultiScaleBatchSampler(
    Sampler[list[tuple[int, int]]]
):
    """Attach one shared resolution to every item in one batch."""

    def __init__(
        self,
        sampler,
        batch_size: int,
        resolutions: Sequence[int],
        *,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.resolutions = [
            int(value)
            for value in resolutions
        ]
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.resolutions:
            raise ValueError("resolution ladder must not be empty")
        if any(value <= 0 for value in self.resolutions):
            raise ValueError("multi-scale resolutions must be positive")
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def _draw(self, generator: torch.Generator) -> int:
        index = int(
            torch.randint(
                len(self.resolutions),
                (1,),
                generator=generator,
            ).item()
        )
        return self.resolutions[index]

    def __iter__(
        self,
    ) -> Iterator[list[tuple[int, int]]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        batch: list[int] = []
        for value in self.sampler:
            batch.append(int(value))
            if len(batch) == self.batch_size:
                resolution = self._draw(generator)
                yield [
                    (index, resolution)
                    for index in batch
                ]
                batch = []

        if batch and not self.drop_last:
            resolution = self._draw(generator)
            yield [
                (index, resolution)
                for index in batch
            ]

    def __len__(self) -> int:
        count = len(self.sampler)
        if self.drop_last:
            return count // self.batch_size
        return (
            count + self.batch_size - 1
        ) // self.batch_size
