from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from apache_multiscale import (
    EpochRandomSampler,
    MultiScaleBatchSampler,
    scale_ladder,
)
from apache_training_recipe import ApacheTrainingRecipeConfig
from benchmark import RelationExample
from model import KFRelationModel, RelationModelConfig
from test_model import ToyBackbone, boxes
from training import prepare_example


def apache_model_config(**overrides) -> RelationModelConfig:
    values = dict(
        image_size=8,
        max_boxes=4,
        pair_budget=6,
        hidden_dim=16,
        geometry_dim=8,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        tap_indices=(-3, -2, -1),
        pair_evidence_contract="apache",
        apache_cfa_prob=1.0,
        apache_cfa_alpha=1.0,
        allow_training_multiscale=True,
    )
    values.update(overrides)
    return RelationModelConfig(**values)


class ApacheTrainingRegularizationTest(unittest.TestCase):
    def test_reference_scale_ladder(self):
        self.assertEqual(
            scale_ladder(
                448,
                0.5,
                1.5,
                7,
                patch=16,
            ),
            [224, 304, 368, 448, 528, 592, 672],
        )

    def test_multiscale_sampler_uses_one_resolution_per_batch_and_epoch_seed(self):
        sampler = EpochRandomSampler(12, seed=42)
        batch_sampler = MultiScaleBatchSampler(
            sampler,
            4,
            [224, 448, 672],
            drop_last=True,
            seed=42,
        )

        batch_sampler.set_epoch(0)
        first = list(batch_sampler)
        batch_sampler.set_epoch(0)
        repeated = list(batch_sampler)
        batch_sampler.set_epoch(1)
        second = list(batch_sampler)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 3)
        for batch in first:
            self.assertEqual(len(batch), 4)
            resolutions = {
                resolution
                for _, resolution in batch
            }
            self.assertEqual(len(resolutions), 1)
            self.assertTrue(
                resolutions.issubset({224, 448, 672})
            )

    def test_dataset_preserves_last_predicate_for_cfa(self):
        example = RelationExample(
            image="image.png",
            width=8,
            height=8,
            boxes_xyxy=(
                (0.0, 0.0, 4.0, 4.0),
                (4.0, 0.0, 8.0, 4.0),
            ),
            object_labels=("person", "object"),
            relations=(
                (0, 0, 1),
                (0, 2, 1),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8), (32, 64, 96)).save(
                root / "image.png"
            )
            prepared = prepare_example(
                example,
                image_root=root,
                image_size=8,
                max_boxes=4,
                predicate_count=3,
            )

        self.assertEqual(
            int(prepared["cfa_predicate_labels"][0, 1]),
            2,
        )
        self.assertEqual(
            float(prepared["predicate_targets"][0, 1, 0]),
            1.0,
        )
        self.assertEqual(
            float(prepared["predicate_targets"][0, 1, 2]),
            1.0,
        )

    def test_cfa_never_crosses_predicate_groups(self):
        torch.manual_seed(92)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            apache_model_config(),
        )
        subject = torch.tensor(
            [[[0.0], [10.0], [100.0], [110.0]]]
        )
        object_ = torch.tensor(
            [[[5.0], [15.0], [105.0], [115.0]]]
        )
        labels = torch.tensor([[0, 0, 1, 1]])
        valid = torch.ones(1, 4, dtype=torch.bool)

        mixed_subject, mixed_object = model._apache_mix_entities(
            subject,
            object_,
            labels,
            valid,
        )

        for value in mixed_subject[0, :2, 0]:
            self.assertGreaterEqual(float(value), 0.0)
            self.assertLessEqual(float(value), 10.0)
        for value in mixed_subject[0, 2:, 0]:
            self.assertGreaterEqual(float(value), 100.0)
            self.assertLessEqual(float(value), 110.0)
        for value in mixed_object[0, :2, 0]:
            self.assertGreaterEqual(float(value), 5.0)
            self.assertLessEqual(float(value), 15.0)
        for value in mixed_object[0, 2:, 0]:
            self.assertGreaterEqual(float(value), 105.0)
            self.assertLessEqual(float(value), 115.0)

    def test_multiscale_shape_is_training_only(self):
        torch.manual_seed(93)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            apache_model_config(),
        )
        box_tensor, box_counts = boxes()
        image = torch.rand(1, 3, 6, 6)

        model.train()
        with torch.no_grad():
            output = model(
                image,
                box_tensor[:1],
                box_counts[:1],
            )
        self.assertEqual(output[0].shape[0], 1)

        model.eval()
        with self.assertRaises(ValueError):
            model(
                image,
                box_tensor[:1],
                box_counts[:1],
            )

    def test_reference_single_process_effective_batch_is_128(self):
        config = ApacheTrainingRecipeConfig()
        self.assertEqual(config.micro_batch_size, 32)
        self.assertEqual(config.grad_accum, 4)
        self.assertEqual(config.effective_batch_size, 128)
        self.assertEqual(config.multi_scale, "0.5,1.5")
        self.assertEqual(config.multi_scale_n, 7)
        self.assertEqual(config.cfa_prob, 0.5)
        self.assertEqual(config.cfa_alpha, 1.0)


if __name__ == "__main__":
    unittest.main()
