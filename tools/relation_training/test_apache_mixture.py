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
    RELEASED_MIX_FRACTIONS,
    RELEASED_SOURCE_NAMES,
    DistributedWeightedSampler,
    RelationMixtureConfig,
    load_relation_mixture,
    realized_source_draws,
    sample_weights_from_fractions,
)
from benchmark import DatasetManifest, RelationVocabulary
from training import RelationMixtureTrainingDataset


def _write_example(
    path: Path,
    *,
    image: str,
    predicate: int = 0,
    source_id: int | None = None,
) -> None:
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
    path.write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )


class ApacheMixtureTest(unittest.TestCase):
    def test_reference_fractions_and_names_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "schema": MIXTURE_SCHEMA,
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
