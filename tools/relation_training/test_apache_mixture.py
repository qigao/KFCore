from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from apache_mixture import (
    MIXTURE_SCHEMA,
    RELEASED_MICRO_BATCH_SIZE,
    RELEASED_MIX_FRACTIONS,
    RELEASED_SAMPLES_PER_EPOCH,
    RELEASED_SEED,
    RELEASED_SOURCE_NAMES,
    RELEASED_WORLD_SIZE,
    ApacheReleasedBatchSampler,
    DistributedWeightedSampler,
    LoadedRelationMixture,
    RelationMixtureConfig,
    RelationMixtureSource,
    load_relation_mixture,
    realized_source_draws,
    sample_weights_from_fractions,
)
from benchmark import (
    DatasetManifest,
    RelationExample,
    RelationVocabulary,
)
from training import (
    FrozenBaselineConfig,
    RelationMixtureTrainingDataset,
    make_training_loader,
)


def _example_payload(
    *,
    image: str,
    predicate: int = 0,
    source_id: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "image": image,
        "width": 8,
        "height": 8,
        "boxes_xyxy": [
            [0.0, 0.0, 4.0, 4.0],
            [4.0, 4.0, 8.0, 8.0],
        ],
        "object_labels": ["person", "horse"],
        "relations": [[0, predicate, 1]],
    }
    if source_id is not None:
        payload["source_id"] = source_id
    return payload


def _write_examples(
    path: Path,
    payloads: list[dict[str, object]],
) -> None:
    path.write_text(
        "".join(
            json.dumps(payload) + "\n"
            for payload in payloads
        ),
        encoding="utf-8",
    )


def _write_example(
    path: Path,
    *,
    image: str,
    predicate: int = 0,
    source_id: int | None = None,
) -> None:
    _write_examples(
        path,
        [
            _example_payload(
                image=image,
                predicate=predicate,
                source_id=source_id,
            )
        ],
    )


class ApacheMixtureTest(unittest.TestCase):
    def test_reference_fractions_and_names_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exclude_path = root / "indoorvg_holdout.json"
            exclude_path.write_text(
                json.dumps({"stems": []}),
                encoding="utf-8",
            )
            payload = {
                "schema": MIXTURE_SCHEMA,
                "exclude_ids": exclude_path.name,
                "sources": [
                    {
                        "name": name,
                        "annotations": f"{name}.jsonl",
                        "image_root": name,
                        "fraction": fraction,
                    }
                    for name, fraction in zip(
                        RELEASED_SOURCE_NAMES,
                        RELEASED_MIX_FRACTIONS,
                    )
                ],
            }
            path = root / "mixture.json"
            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            config = RelationMixtureConfig.load(path)

        self.assertEqual(
            config.source_names,
            RELEASED_SOURCE_NAMES,
        )
        self.assertTrue(
            config.matches_released_mixture()
        )
        self.assertTrue(
            np.allclose(
                config.fractions,
                RELEASED_MIX_FRACTIONS,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        self.assertEqual(config.seed, RELEASED_SEED)
        self.assertEqual(config.samples_per_epoch, 0)
        self.assertTrue(config.exclude_ids_sha256)

    def test_released_config_gate_rejects_recipe_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exclude_path = root / "holdout.json"
            exclude_path.write_text(
                json.dumps({"stems": []}),
                encoding="utf-8",
            )
            sources = [
                {
                    "name": name,
                    "annotations": f"{name}.jsonl",
                    "image_root": name,
                    "fraction": fraction,
                }
                for name, fraction in zip(
                    RELEASED_SOURCE_NAMES,
                    RELEASED_MIX_FRACTIONS,
                )
            ]

            def load(**extra):
                payload = {
                    "schema": MIXTURE_SCHEMA,
                    "sources": sources,
                    **extra,
                }
                path = root / "mixture.json"
                path.write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                return RelationMixtureConfig.load(path)

            self.assertFalse(
                load().matches_released_mixture()
            )
            self.assertFalse(
                load(
                    exclude_ids=exclude_path.name,
                    seed=99,
                ).matches_released_mixture()
            )
            self.assertFalse(
                load(
                    exclude_ids=exclude_path.name,
                    samples_per_epoch=RELEASED_SAMPLES_PER_EPOCH,
                ).matches_released_mixture()
            )
            self.assertTrue(
                load(
                    exclude_ids=exclude_path.name,
                    seed=RELEASED_SEED,
                    samples_per_epoch=0,
                ).matches_released_mixture()
            )

    def test_exclude_ids_filters_before_source_weighting(self):
        vocabulary = RelationVocabulary(
            ("riding",),
            ("person", "horse"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for name in ("left", "right"):
                annotations = root / f"{name}.jsonl"
                _write_examples(
                    annotations,
                    [
                        _example_payload(
                            image=f"{name}_keep.jpg",
                        ),
                        _example_payload(
                            image=f"{name}_held.jpg",
                        ),
                    ],
                )
                sources.append(
                    {
                        "name": name,
                        "annotations": annotations.name,
                        "image_root": name,
                        "fraction": 0.5,
                    }
                )

            exclude_path = root / "holdout.json"
            exclude_path.write_text(
                json.dumps(
                    {
                        "stems": [
                            "left_held",
                            "right_held",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            mixture_path = root / "mixture.json"
            mixture_path.write_text(
                json.dumps(
                    {
                        "schema": MIXTURE_SCHEMA,
                        "sources": sources,
                        "exclude_ids": exclude_path.name,
                    }
                ),
                encoding="utf-8",
            )
            mixture = load_relation_mixture(
                mixture_path,
                vocabulary,
            )

        self.assertEqual(
            mixture.source_excluded_counts,
            (1, 1),
        )
        self.assertEqual(
            len(mixture.combined_manifest.examples),
            2,
        )
        self.assertEqual(
            mixture.source_of_index.tolist(),
            [0, 1],
        )
        self.assertTrue(
            mixture.config.exclude_ids_sha256
        )

    def test_sample_weights_realize_source_fractions_not_source_sizes(self):
        source = np.asarray(
            [0] * 1000 + [1] * 10 + [2] * 2,
            dtype=np.int64,
        )
        fractions = (0.7274, 0.0630, 0.2096)
        weights = sample_weights_from_fractions(
            source,
            fractions,
        )

        self.assertAlmostEqual(
            float(weights[source == 0].sum()),
            fractions[0],
            places=12,
        )
        self.assertAlmostEqual(
            float(weights[source == 1].sum()),
            fractions[1],
            places=12,
        )
        self.assertAlmostEqual(
            float(weights[source == 2].sum()),
            fractions[2],
            places=12,
        )
        self.assertAlmostEqual(
            float(weights.sum()),
            1.0,
            places=12,
        )
        self.assertGreater(
            float(weights[source == 2][0]),
            float(weights[source == 0][0]),
        )

    def test_distributed_sampler_shards_one_global_draw(self):
        weights = np.asarray(
            [0.1, 0.2, 0.3, 0.4],
            dtype=np.float64,
        )
        samplers = [
            DistributedWeightedSampler(
                weights,
                num_replicas=3,
                rank=rank,
                num_samples=10,
                seed=77,
            )
            for rank in range(3)
        ]
        for sampler in samplers:
            sampler.set_epoch(4)

        shards = [
            list(iter(sampler))
            for sampler in samplers
        ]
        global_draw = (
            samplers[0]
            .last_global_indices
            .tolist()
        )
        self.assertEqual(len(global_draw), 12)
        self.assertEqual(shards[0], global_draw[0::3])
        self.assertEqual(shards[1], global_draw[1::3])
        self.assertEqual(shards[2], global_draw[2::3])
        self.assertEqual(len(shards[0]), 4)

        for sampler in samplers[1:]:
            self.assertTrue(
                torch.equal(
                    samplers[0].last_global_indices,
                    sampler.last_global_indices,
                )
            )

        samplers[0].set_epoch(5)
        _ = list(iter(samplers[0]))
        self.assertFalse(
            torch.equal(
                torch.tensor(global_draw),
                samplers[0].last_global_indices,
            )
        )

    def test_mixture_loader_overrides_source_ids_and_routes_image_roots(self):
        vocabulary = RelationVocabulary(
            ("riding",),
            ("person", "horse"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for source_id, name in enumerate(("a", "b")):
                image_root = root / name
                image_root.mkdir()
                image_name = f"{name}.png"
                Image.new(
                    "RGB",
                    (8, 8),
                    color=(
                        10 + source_id,
                        20,
                        30,
                    ),
                ).save(image_root / image_name)
                annotations = root / f"{name}.jsonl"
                _write_example(
                    annotations,
                    image=image_name,
                    source_id=99,
                )
                sources.append(
                    {
                        "name": name,
                        "annotations": annotations.name,
                        "image_root": name,
                        "fraction": (
                            0.25
                            if source_id == 0
                            else 0.75
                        ),
                    }
                )

            mixture_path = root / "mixture.json"
            mixture_path.write_text(
                json.dumps(
                    {
                        "schema": MIXTURE_SCHEMA,
                        "sources": sources,
                        "samples_per_epoch": 8,
                        "seed": 123,
                    }
                ),
                encoding="utf-8",
            )
            mixture = load_relation_mixture(
                mixture_path,
                vocabulary,
            )
            dataset = RelationMixtureTrainingDataset(
                mixture,
                image_size=8,
                max_boxes=2,
                predicate_count=1,
                object_labels=vocabulary.object_labels,
            )

            first = dataset[0]
            second = dataset[(1, 8)]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(
            [example.source_id for example in mixture.combined_manifest.examples],
            [0, 1],
        )
        self.assertEqual(int(first["source_id"]), 0)
        self.assertEqual(int(second["source_id"]), 1)
        self.assertEqual(
            int(first["training_resolution"]),
            8,
        )
        self.assertEqual(
            int(second["training_resolution"]),
            8,
        )
        self.assertEqual(mixture.draws_per_epoch, 8)
        report = mixture.report()
        self.assertEqual(
            report["source_names"],
            ["a", "b"],
        )
        self.assertEqual(
            report["source_counts"],
            [1, 1],
        )
        self.assertTrue(
            np.allclose(
                report["weight_realized_fractions"],
                [0.25, 0.75],
            )
        )

    def test_weighted_sampler_composes_with_multiscale_loader(self):
        vocabulary = RelationVocabulary(
            ("riding",),
            ("person", "horse"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for source_id, name in enumerate(("left", "right")):
                image_root = root / name
                image_root.mkdir()
                image_name = f"{name}.png"
                Image.new(
                    "RGB",
                    (8, 8),
                    color=(20 + source_id, 30, 40),
                ).save(image_root / image_name)
                annotations = root / f"{name}.jsonl"
                _write_example(
                    annotations,
                    image=image_name,
                )
                sources.append(
                    {
                        "name": name,
                        "annotations": annotations.name,
                        "image_root": name,
                        "fraction": 0.5,
                    }
                )

            mixture_path = root / "mixture.json"
            mixture_path.write_text(
                json.dumps(
                    {
                        "schema": MIXTURE_SCHEMA,
                        "sources": sources,
                        "samples_per_epoch": 8,
                        "seed": 44,
                    }
                ),
                encoding="utf-8",
            )
            mixture = load_relation_mixture(
                mixture_path,
                vocabulary,
            )
            dataset = RelationMixtureTrainingDataset(
                mixture,
                image_size=8,
                max_boxes=2,
                predicate_count=1,
                object_labels=vocabulary.object_labels,
            )
            weights = sample_weights_from_fractions(
                mixture.source_of_index,
                mixture.config.fractions,
            )
            sampler = DistributedWeightedSampler(
                weights,
                num_samples=mixture.draws_per_epoch,
                seed=mixture.config.seed,
            )
            loader = make_training_loader(
                dataset,
                FrozenBaselineConfig(
                    epochs=1,
                    batch_size=2,
                    learning_rate=1.0e-3,
                    weight_decay=0.0,
                    seed=44,
                ),
                resolutions=[8, 16],
                drop_last=True,
                sampler=sampler,
            )
            loader.batch_sampler.set_epoch(0)
            first_epoch = [
                (
                    batch["source_id"].tolist(),
                    batch["training_resolution"].tolist(),
                )
                for batch in loader
            ]
            first_draw = sampler.last_global_indices.clone()

            loader.batch_sampler.set_epoch(1)
            second_epoch = [
                (
                    batch["source_id"].tolist(),
                    batch["training_resolution"].tolist(),
                )
                for batch in loader
            ]
            second_draw = sampler.last_global_indices.clone()

        self.assertEqual(len(first_epoch), 4)
        self.assertEqual(len(second_epoch), 4)
        for source_ids, resolutions in first_epoch + second_epoch:
            self.assertEqual(len(source_ids), 2)
            self.assertIn(source_ids[0], (0, 1))
            self.assertIn(source_ids[1], (0, 1))
            self.assertEqual(resolutions[0], resolutions[1])
            self.assertIn(resolutions[0], (8, 16))
        self.assertFalse(
            torch.equal(first_draw, second_draw)
        )

    def test_released_logical_rank_sampler_matches_apache_topology(self):
        weights = np.asarray(
            [0.1, 0.2, 0.3, 0.4],
            dtype=np.float64,
        )
        sampler = ApacheReleasedBatchSampler(
            weights,
            resolutions=(224, 304),
        )
        self.assertEqual(
            sampler.global_sampler.total_size,
            503_756,
        )
        self.assertEqual(
            sampler.global_sampler.num_samples,
            125_939,
        )
        self.assertEqual(
            sampler.optimizer_steps_per_epoch,
            3_935,
        )
        self.assertEqual(
            len(sampler),
            15_740,
        )
        self.assertTrue(
            sampler.matches_released_topology()
        )

        iterator = iter(sampler)
        logical_rank_batches = [
            next(iterator)
            for _ in range(RELEASED_WORLD_SIZE)
        ]
        global_draw = sampler.last_global_indices
        assert global_draw is not None
        for rank, batch in enumerate(logical_rank_batches):
            indices = [
                int(value[0])
                for value in batch
            ]
            resolutions = {
                int(value[1])
                for value in batch
            }
            self.assertEqual(
                indices,
                global_draw[
                    rank :
                    rank
                    + RELEASED_WORLD_SIZE
                    * RELEASED_MICRO_BATCH_SIZE :
                    RELEASED_WORLD_SIZE
                ].tolist(),
            )
            self.assertEqual(len(resolutions), 1)

        self.assertEqual(
            logical_rank_batches[0][0][1],
            logical_rank_batches[1][0][1],
        )
        self.assertEqual(
            logical_rank_batches[1][0][1],
            logical_rank_batches[2][0][1],
        )
        self.assertEqual(
            logical_rank_batches[2][0][1],
            logical_rank_batches[3][0][1],
        )

    def test_loaded_released_gate_requires_exact_post_exclusion_epoch(self):
        example = RelationExample(
            image="x.jpg",
            width=8,
            height=8,
            boxes_xyxy=(
                (0.0, 0.0, 4.0, 4.0),
                (4.0, 4.0, 8.0, 8.0),
            ),
            object_labels=("person", "horse"),
            relations=((0, 0, 1),),
            source_id=0,
        )
        examples = (
            (example,)
            * RELEASED_SAMPLES_PER_EPOCH
        )
        source = np.zeros(
            RELEASED_SAMPLES_PER_EPOCH,
            dtype=np.int64,
        )
        source[-2] = 1
        source[-1] = 2
        config = RelationMixtureConfig(
            sources=tuple(
                RelationMixtureSource(
                    name=name,
                    annotations=Path(f"{name}.jsonl"),
                    image_root=Path(name),
                    fraction=fraction,
                )
                for name, fraction in zip(
                    RELEASED_SOURCE_NAMES,
                    RELEASED_MIX_FRACTIONS,
                )
            ),
            samples_per_epoch=0,
            seed=RELEASED_SEED,
            exclude_ids=Path("holdout.json"),
            exclude_ids_sha256="golden",
        )
        manifest = DatasetManifest(
            examples=examples,
            annotations_sha256="manifest",
            vocabulary_sha256="vocab",
        )
        mixture = LoadedRelationMixture(
            config=config,
            manifests=(),
            combined_manifest=manifest,
            source_of_index=source,
            source_annotation_sha256=(),
            source_excluded_counts=(),
        )
        self.assertTrue(
            mixture.matches_released_sampling_contract()
        )

        short_manifest = DatasetManifest(
            examples=examples[:-1],
            annotations_sha256="manifest-short",
            vocabulary_sha256="vocab",
        )
        short = LoadedRelationMixture(
            config=config,
            manifests=(),
            combined_manifest=short_manifest,
            source_of_index=source[:-1],
            source_annotation_sha256=(),
            source_excluded_counts=(),
        )
        self.assertFalse(
            short.matches_released_sampling_contract()
        )

    def test_realized_draw_report_uses_global_padded_draw(self):
        source = np.asarray(
            [0, 0, 1, 1],
            dtype=np.int64,
        )
        weights = sample_weights_from_fractions(
            source,
            (0.5, 0.5),
        )
        sampler = DistributedWeightedSampler(
            weights,
            num_replicas=3,
            rank=1,
            num_samples=5,
            seed=9,
        )
        sampler.set_epoch(2)
        _ = list(iter(sampler))
        report = realized_source_draws(
            sampler,
            source,
            2,
        )
        self.assertEqual(
            report["global_draw_count"],
            6,
        )
        self.assertEqual(
            sum(report["source_counts"]),
            6,
        )
        self.assertAlmostEqual(
            sum(report["source_fractions"]),
            1.0,
            places=12,
        )

    def test_missing_target_source_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_weights_from_fractions(
                np.asarray([0, 0, 2], dtype=np.int64),
                (0.4, 0.3, 0.3),
            )


if __name__ == "__main__":
    unittest.main()
