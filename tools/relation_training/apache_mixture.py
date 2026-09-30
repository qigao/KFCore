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
RELEASED_SAMPLES_PER_EPOCH = 503_754
RELEASED_WORLD_SIZE = 4
RELEASED_MICRO_BATCH_SIZE = 32
RELEASED_SEED = 42


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
    seed: int = RELEASED_SEED
    config_sha256: str = ""
    exclude_ids: Path | None = None
    exclude_ids_sha256: str = ""

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

    def matches_released_source_spec(self) -> bool:
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

    def matches_released_mixture(self) -> bool:
        """Config-level gate for the released source-mixture recipe.

        Dataset identity is checked after loading, because the released epoch
        size is a property of the post-exclusion source universe.
        """
        return (
            self.matches_released_source_spec()
            and self.samples_per_epoch == 0
            and self.seed == RELEASED_SEED
            and self.exclude_ids is not None
            and bool(self.exclude_ids_sha256)
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
        seed = payload.get("seed", RELEASED_SEED)

        raw_exclude_ids = payload.get("exclude_ids")
        exclude_ids = None
        exclude_ids_sha256 = ""
        if raw_exclude_ids is not None:
            if (
                not isinstance(raw_exclude_ids, str)
                or not raw_exclude_ids
            ):
                raise ValueError(
                    "exclude_ids must be a non-empty path string"
                )
            exclude_ids = Path(raw_exclude_ids)
            if not exclude_ids.is_absolute():
                exclude_ids = base / exclude_ids
            exclude_ids = exclude_ids.resolve()
            exclude_raw = exclude_ids.read_bytes()
            exclude_ids_sha256 = hashlib.sha256(
                exclude_raw
            ).hexdigest()

        return cls(
            sources=tuple(sources),
            samples_per_epoch=samples_per_epoch,
            seed=seed,
            config_sha256=hashlib.sha256(raw).hexdigest(),
            exclude_ids=exclude_ids,
            exclude_ids_sha256=exclude_ids_sha256,
        )


@dataclass(frozen=True)
class LoadedRelationMixture:
    config: RelationMixtureConfig
    manifests: tuple[DatasetManifest, ...]
    combined_manifest: DatasetManifest
    source_of_index: np.ndarray
    source_annotation_sha256: tuple[str, ...]
    source_excluded_counts: tuple[int, ...]

    @property
    def draws_per_epoch(self) -> int:
        return (
            self.config.samples_per_epoch
            if self.config.samples_per_epoch > 0
            else len(self.combined_manifest.examples)
        )

    def matches_released_sampling_contract(self) -> bool:
        return (
            self.config.matches_released_mixture()
            and self.draws_per_epoch == RELEASED_SAMPLES_PER_EPOCH
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
            "source_excluded_counts": list(
                self.source_excluded_counts
            ),
            "exclude_ids_sha256": self.config.exclude_ids_sha256,
            "target_fractions": list(
                self.config.fractions
            ),
            "weight_realized_fractions": list(
                realized
            ),
            "draws_per_epoch": self.draws_per_epoch,
            "released_draws_per_epoch": RELEASED_SAMPLES_PER_EPOCH,
            "seed": self.config.seed,
            "matches_released_source_spec": (
                self.config.matches_released_source_spec()
            ),
            "matches_released_mixture": (
                self.matches_released_sampling_contract()
            ),
        }


def _load_excluded_stems(
    path: Path | None,
) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = (
        payload.get("stems")
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(values, list):
        raise ValueError(
            "exclude_ids must be a list or an object with a stems list"
        )
    stems: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(
                "exclude_ids stems must be non-empty strings"
            )
        stems.add(value)
    return stems


def load_relation_mixture(
    path: str | Path,
    vocabulary: RelationVocabulary,
) -> LoadedRelationMixture:
    config = RelationMixtureConfig.load(path)
    excluded_stems = _load_excluded_stems(
        config.exclude_ids
    )
    manifests: list[DatasetManifest] = []
    source_index_parts: list[np.ndarray] = []
    combined_examples = []
    annotation_hashes: list[str] = []
    excluded_counts: list[int] = []

    for source_id, source in enumerate(config.sources):
        raw_manifest = DatasetManifest.load(
            source.annotations,
            vocabulary,
        )
        filtered_examples = tuple(
            example
            for example in raw_manifest.examples
            if Path(example.image).stem not in excluded_stems
        )
        excluded_counts.append(
            len(raw_manifest.examples)
            - len(filtered_examples)
        )
        manifest = DatasetManifest(
            examples=tuple(
                replace(
                    example,
                    source_id=source_id,
                )
                for example in filtered_examples
            ),
            annotations_sha256=raw_manifest.annotations_sha256,
            vocabulary_sha256=raw_manifest.vocabulary_sha256,
        )
        manifests.append(manifest)
        annotation_hashes.append(
            raw_manifest.annotations_sha256
        )
        combined_examples.extend(
            manifest.examples
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
    digest.update(
        config.exclude_ids_sha256.encode("ascii")
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
        source_excluded_counts=tuple(
            excluded_counts
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
        self.requested_num_samples = int(requested)
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
        self.last_global_indices: torch.Tensor | None = None

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

    def draw_global(self) -> torch.Tensor:
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
        self.last_global_indices = indices.clone()
        return indices

    def __iter__(self) -> Iterator[int]:
        indices = self.draw_global()
        yield from indices[
            self.rank :
            self.total_size :
            self.num_replicas
        ].tolist()

    def __len__(self) -> int:
        return self.num_samples


class ApacheReleasedBatchSampler(
    Sampler[list[int | tuple[int, int]]]
):
    """Single-process emulation of the released four-rank Apache stream.

    The upstream release draws one padded global multinomial, shards it by
    rank, batches each rank independently, and uses the same multi-scale
    resolution sequence on every rank. Yielding the four logical-rank
    micro-batches consecutively lets grad_accum=4 reproduce that grouping.
    """

    def __init__(
        self,
        weights: np.ndarray,
        *,
        resolutions: Sequence[int] = (),
        num_samples: int = RELEASED_SAMPLES_PER_EPOCH,
        batch_size: int = RELEASED_MICRO_BATCH_SIZE,
        world_size: int = RELEASED_WORLD_SIZE,
        seed: int = RELEASED_SEED,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size must be a positive integer"
            )
        if (
            isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size <= 0
        ):
            raise ValueError(
                "world_size must be a positive integer"
            )
        self.batch_size = int(batch_size)
        self.world_size = int(world_size)
        self.resolutions = tuple(
            int(value)
            for value in resolutions
        )
        if any(value <= 0 for value in self.resolutions):
            raise ValueError(
                "resolutions must be positive"
            )
        self.seed = int(seed)
        self.epoch = 0
        self.global_sampler = DistributedWeightedSampler(
            weights,
            num_replicas=self.world_size,
            rank=0,
            num_samples=num_samples,
            seed=self.seed,
        )

    @property
    def last_global_indices(self) -> torch.Tensor | None:
        return self.global_sampler.last_global_indices

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return (
            self.global_sampler.num_samples
            // self.batch_size
        )

    def matches_released_topology(self) -> bool:
        return (
            self.global_sampler.requested_num_samples
            == RELEASED_SAMPLES_PER_EPOCH
            and self.batch_size == RELEASED_MICRO_BATCH_SIZE
            and self.world_size == RELEASED_WORLD_SIZE
            and self.seed == RELEASED_SEED
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.global_sampler.set_epoch(epoch)

    def __iter__(
        self,
    ) -> Iterator[list[int | tuple[int, int]]]:
        global_indices = self.global_sampler.draw_global()
        shards = tuple(
            global_indices[
                rank :
                self.global_sampler.total_size :
                self.world_size
            ]
            for rank in range(self.world_size)
        )
        steps = self.optimizer_steps_per_epoch
        resolution_generator = torch.Generator()
        resolution_generator.manual_seed(
            self.seed + self.epoch
        )

        for step in range(steps):
            resolution = None
            if self.resolutions:
                resolution = self.resolutions[
                    int(
                        torch.randint(
                            len(self.resolutions),
                            (1,),
                            generator=resolution_generator,
                        ).item()
                    )
                ]
            start = step * self.batch_size
            stop = start + self.batch_size
            for rank in range(self.world_size):
                values = shards[rank][start:stop].tolist()
                if resolution is None:
                    yield [
                        int(value)
                        for value in values
                    ]
                else:
                    yield [
                        (int(value), resolution)
                        for value in values
                    ]

    def __len__(self) -> int:
        return (
            self.optimizer_steps_per_epoch
            * self.world_size
        )



def validate_mixture_disjoint_validation(
    mixture: LoadedRelationMixture,
    validation_manifest: DatasetManifest,
    *,
    validation_image_root: str | Path,
) -> None:
    validation_root = Path(validation_image_root).resolve()
    validation_paths = {
        (validation_root / example.image).resolve()
        for example in validation_manifest.examples
    }
    for source, manifest in zip(
        mixture.config.sources,
        mixture.manifests,
    ):
        source_root = source.image_root.resolve()
        for example in manifest.examples:
            path = (
                source_root / example.image
            ).resolve()
            if path in validation_paths:
                raise ValueError(
                    "train/validation image leakage: "
                    + str(path)
                )



def realized_source_draws(
    sampler: DistributedWeightedSampler,
    source_of_index: np.ndarray,
    source_count: int,
) -> dict[str, object]:
    if sampler.last_global_indices is None:
        raise ValueError(
            "sampler has not produced an epoch draw yet"
        )
    source = np.asarray(
        source_of_index,
        dtype=np.int64,
    )
    indices = (
        sampler.last_global_indices.detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    if (
        indices.ndim != 1
        or indices.size != sampler.total_size
        or np.any(indices < 0)
        or np.any(indices >= source.shape[0])
    ):
        raise RuntimeError(
            "sampler recorded invalid global indices"
        )
    counts = np.bincount(
        source[indices],
        minlength=source_count,
    )
    fractions = counts.astype(
        np.float64
    ) / float(indices.size)
    return {
        "epoch": sampler.epoch,
        "global_draw_count": int(indices.size),
        "source_counts": [
            int(value)
            for value in counts.tolist()
        ],
        "source_fractions": [
            float(value)
            for value in fractions.tolist()
        ],
    }
