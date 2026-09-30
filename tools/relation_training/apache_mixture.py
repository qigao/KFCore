from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from benchmark import DatasetManifest, RelationVocabulary


MIXTURE_SCHEMA = "kfcore.relation-training-mixture/1"
RELEASED_SOURCE_NAMES = (
    "megasg_clean",
    "vg_raw",
    "hicodet",
)
RELEASED_MIX_FRACTIONS = (
    0.7274,
    0.0630,
    0.2096,
)


@dataclass(frozen=True)
class RelationMixtureSource:
    name: str
    annotations: Path
    image_root: Path
    fraction: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError(
                "mixture source name must not be empty"
            )
        if not math.isfinite(self.fraction) or self.fraction <= 0.0:
            raise ValueError(
                "mixture source fraction must be finite and positive"
            )


@dataclass(frozen=True)
class RelationMixtureConfig:
    sources: tuple[RelationMixtureSource, ...]
    samples_per_epoch: int = 0
    seed: int = 42
    config_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError(
                "relation training mixture requires at least one source"
            )
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(
                "mixture source names must be unique"
            )
        if (
            isinstance(self.samples_per_epoch, bool)
            or not isinstance(self.samples_per_epoch, int)
            or self.samples_per_epoch < 0
        ):
            raise ValueError(
                "samples_per_epoch must be a non-negative integer"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError(
                "mixture seed must be a non-negative integer"
            )

    @property
    def fractions(self) -> tuple[float, ...]:
        total = sum(source.fraction for source in self.sources)
        return tuple(
            source.fraction / total
            for source in self.sources
        )

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(
            source.name
            for source in self.sources
        )

    def matches_released_mixture(self) -> bool:
        if self.source_names != RELEASED_SOURCE_NAMES:
            return False
        return all(
            math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for actual, expected in zip(
                self.fractions,
                RELEASED_MIX_FRACTIONS,
            )
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "RelationMixtureConfig":
        config_path = Path(path)
        raw = config_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != MIXTURE_SCHEMA
        ):
            raise ValueError(
                "unsupported relation training mixture schema"
            )

        raw_sources = payload.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(
                "relation training mixture requires a sources array"
            )

        base = config_path.parent.resolve()
        sources: list[RelationMixtureSource] = []
        for source in raw_sources:
            if not isinstance(source, dict):
                raise ValueError(
                    "mixture source must be a JSON object"
                )
            name = source.get("name")
            annotations = source.get("annotations")
            image_root = source.get("image_root")
            fraction = source.get("fraction")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "mixture source name must be a non-empty string"
                )
            if (
                not isinstance(annotations, str)
                or not annotations
                or not isinstance(image_root, str)
                or not image_root
            ):
                raise ValueError(
                    "mixture source annotations/image_root must be paths"
                )
            if (
                isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
            ):
                raise ValueError(
                    "mixture source fraction must be numeric"
                )

            annotation_path = Path(annotations)
            if not annotation_path.is_absolute():
                annotation_path = base / annotation_path
            root_path = Path(image_root)
            if not root_path.is_absolute():
                root_path = base / root_path

            sources.append(
                RelationMixtureSource(
                    name=name,
                    annotations=annotation_path.resolve(),
                    image_root=root_path.resolve(),
                    fraction=float(fraction),
                )
            )

        samples_per_epoch = payload.get(
            "samples_per_epoch",
            0,
        )
        seed = payload.get("seed", 42)
        return cls(
            sources=tuple(sources),
            samples_per_epoch=samples_per_epoch,
            seed=seed,
            config_sha256=hashlib.sha256(raw).hexdigest(),
        )


@dataclass(frozen=True)
class LoadedRelationMixture:
    config: RelationMixtureConfig
    manifests: tuple[DatasetManifest, ...]
    combined_manifest: DatasetManifest
    source_of_index: np.ndarray
    source_annotation_sha256: tuple[str, ...]

    @property
    def draws_per_epoch(self) -> int:
        return (
            self.config.samples_per_epoch
            if self.config.samples_per_epoch > 0
            else len(self.combined_manifest.examples)
        )

    def report(self) -> dict[str, object]:
        counts = np.bincount(
            self.source_of_index,
            minlength=len(self.config.sources),
        )
        weights = sample_weights_from_fractions(
            self.source_of_index,
            self.config.fractions,
        )
        realized = tuple(
            float(
                weights[
                    self.source_of_index == index
                ].sum()
            )
            for index in range(len(self.config.sources))
        )
        return {
            "schema": MIXTURE_SCHEMA,
            "config_sha256": self.config.config_sha256,
            "source_names": list(
                self.config.source_names
            ),
            "source_counts": [
                int(value)
                for value in counts.tolist()
            ],
            "source_annotation_sha256": list(
                self.source_annotation_sha256
            ),
            "target_fractions": list(
                self.config.fractions
            ),
            "weight_realized_fractions": list(
                realized
            ),
            "draws_per_epoch": self.draws_per_epoch,
            "seed": self.config.seed,
            "matches_released_mixture": (
                self.config.matches_released_mixture()
            ),
        }


def load_relation_mixture(
    path: str | Path,
    vocabulary: RelationVocabulary,
) -> LoadedRelationMixture:
    config = RelationMixtureConfig.load(path)
    manifests: list[DatasetManifest] = []
    source_index_parts: list[np.ndarray] = []
    combined_examples = []
    annotation_hashes: list[str] = []

    for source_id, source in enumerate(config.sources):
        manifest = DatasetManifest.load(
            source.annotations,
            vocabulary,
        )
        manifests.append(manifest)
        annotation_hashes.append(
            manifest.annotations_sha256
        )
        combined_examples.extend(
            replace(
                example,
                source_id=source_id,
            )
            for example in manifest.examples
        )
        source_index_parts.append(
            np.full(
                len(manifest.examples),
                source_id,
                dtype=np.int64,
            )
        )

    source_of_index = np.concatenate(
        source_index_parts
    )
    digest = hashlib.sha256()
    digest.update(
        config.config_sha256.encode("ascii")
    )
    digest.update(b"\0")
    for source, manifest in zip(
        config.sources,
        manifests,
    ):
        digest.update(
            source.name.encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            manifest.annotations_sha256.encode("ascii")
        )
        digest.update(b"\0")

    combined_manifest = DatasetManifest(
        examples=tuple(combined_examples),
        annotations_sha256=digest.hexdigest(),
        vocabulary_sha256=vocabulary.sha256(),
    )
    return LoadedRelationMixture(
        config=config,
        manifests=tuple(manifests),
        combined_manifest=combined_manifest,
        source_of_index=source_of_index,
        source_annotation_sha256=tuple(
            annotation_hashes
        ),
    )


def sample_weights_from_fractions(
    source_of_index: np.ndarray,
    target_fractions: Sequence[float],
) -> np.ndarray:
    source = np.asarray(
        source_of_index,
        dtype=np.int64,
    )
    if source.ndim != 1 or source.size <= 0:
        raise ValueError(
            "source_of_index must be a non-empty 1-D array"
        )
    if np.any(source < 0):
        raise ValueError(
            "source_of_index must be non-negative"
        )
    if not target_fractions:
        raise ValueError(
            "target_fractions must not be empty"
        )
    if int(source.max()) >= len(target_fractions):
        raise ValueError(
            "source_of_index exceeds target fraction count"
        )

    fractions = np.asarray(
        target_fractions,
        dtype=np.float64,
    )
    if (
        fractions.ndim != 1
        or not np.isfinite(fractions).all()
        or np.any(fractions <= 0.0)
    ):
        raise ValueError(
            "target fractions must be finite and positive"
        )
    fractions = fractions / fractions.sum()

    counts = np.bincount(
        source,
        minlength=len(fractions),
    )
    if np.any(counts == 0):
        missing = int(
            np.nonzero(counts == 0)[0][0]
        )
        raise ValueError(
            f"target source {missing} has no samples"
        )

    per_source = (
        fractions
        / counts.astype(np.float64)
    )
    weights = per_source[source]
    weights = weights / weights.sum()
    return weights.astype(
        np.float64,
        copy=False,
    )


class DistributedWeightedSampler(Sampler[int]):
    """Reference weighted-with-replacement epoch draw with rank sharding."""

    def __init__(
        self,
        weights: np.ndarray,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        tensor = torch.as_tensor(
            weights,
            dtype=torch.double,
        )
        if (
            tensor.ndim != 1
            or tensor.numel() <= 0
            or not torch.isfinite(tensor).all()
            or (tensor < 0).any()
            or tensor.sum() <= 0
        ):
            raise ValueError(
                "sampler weights must be finite non-negative 1-D values"
            )
        if (
            isinstance(num_replicas, bool)
            or not isinstance(num_replicas, int)
            or num_replicas <= 0
        ):
            raise ValueError(
                "num_replicas must be a positive integer"
            )
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= num_replicas
        ):
            raise ValueError(
                "rank must be within [0,num_replicas)"
            )
        requested = (
            tensor.numel()
            if num_samples is None
            else num_samples
        )
        if (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or requested <= 0
        ):
            raise ValueError(
                "num_samples must be a positive integer"
            )
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise ValueError(
                "sampler seed must be a non-negative integer"
            )

        self.weights = tensor
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.total_size = int(
            math.ceil(
                requested
                / self.num_replicas
            )
        ) * self.num_replicas
        self.num_samples = (
            self.total_size
            // self.num_replicas
        )
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
        ):
            raise ValueError(
                "sampler epoch must be a non-negative integer"
            )
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(
            self.seed + self.epoch
        )
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        yield from indices[
            self.rank :
            self.total_size :
            self.num_replicas
        ].tolist()

    def __len__(self) -> int:
        return self.num_samples
