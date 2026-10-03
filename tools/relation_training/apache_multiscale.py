from __future__ import annotations

from typing import Iterator

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
    if base <= 0 or patch <= 0:
        raise ValueError("base and patch must be positive")
    if n < 1:
        raise ValueError("multi-scale rung count must be >= 1")
    if not (0.0 < lo <= hi):
        raise ValueError("invalid multi-scale range")
    if n == 1:
        values = [(lo + hi) * 0.5]
    else:
        values = [
            lo + (hi - lo) * index / (n - 1)
            for index in range(n)
        ]
    return sorted(
        {
            max(
                patch,
                int(round(base * value / patch)) * patch,
            )
            for value in values
        }
    )


class EpochRandomSampler(Sampler[int]):
    def __init__(
        self,
        size: int,
        *,
        seed: int,
    ) -> None:
        if size <= 0:
            raise ValueError("sampler size must be positive")
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(
            self.size,
            generator=generator,
        ).tolist()

    def __len__(self) -> int:
        return self.size


class MultiScaleBatchSampler(Sampler[list[tuple[int, int]]]):
    def __init__(
        self,
        sampler: Sampler[int],
        batch_size: int,
        resolutions: list[int],
        *,
        drop_last: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not resolutions:
            raise ValueError("resolution ladder must not be empty")
        if any(value <= 0 for value in resolutions):
            raise ValueError("resolutions must be positive")
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.resolutions = [int(value) for value in resolutions]
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

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        batch: list[int] = []
        for index in self.sampler:
            batch.append(int(index))
            if len(batch) == self.batch_size:
                resolution = self._draw(generator)
                yield [
                    (value, resolution)
                    for value in batch
                ]
                batch = []
        if batch and not self.drop_last:
            resolution = self._draw(generator)
            yield [
                (value, resolution)
                for value in batch
            ]

    def __len__(self) -> int:
        count = len(self.sampler)
        if self.drop_last:
            return count // self.batch_size
        return (
            count + self.batch_size - 1
        ) // self.batch_size
