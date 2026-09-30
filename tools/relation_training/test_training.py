from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch
from torch import nn

from apache_training_recipe import ModelEMA, module_state_sha256
from benchmark import DatasetManifest, RelationExample, RelationVocabulary
from model import BackboneAdapter, KFRelationModel, RelationModelConfig
from training import (
    FrozenBaselineConfig,
    RelationTrainingDataset,
    build_predicate_weighting,
    build_zero_support_negative_weights,
    evaluate_gt_boxes,
    freeze_backbone,
    make_training_loader,
    photometric_jitter,
    prepare_example,
    seed_everything,
    train_epoch,
    trainable_parameters,
    validate_disjoint_splits,
)


class ToyBackbone(BackboneAdapter):
    hidden_size = 8
    patch_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, self.hidden_size, kernel_size=2, stride=2)

    def forward_taps(self, image: torch.Tensor, taps) -> list[torch.Tensor]:
        base = self.conv(image)
        return [
            base + float(index) * 0.1
            for index, _ in enumerate(taps)
        ]


def model_config() -> RelationModelConfig:
    return RelationModelConfig(
        image_size=8,
        max_boxes=4,
        pair_budget=6,
        hidden_dim=16,
        geometry_dim=8,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        tap_indices=(-3, -2, -1),
    )


def record(image: str, relations) -> dict[str, object]:
    return {
        "image": image,
        "width": 8,
        "height": 8,
        "boxes_xyxy": [
            [0, 0, 4, 8],
            [4, 0, 8, 8],
        ],
        "object_labels": ["left", "right"],
        "relations": relations,
    }


def write_manifest(path: Path, payloads) -> None:
    path.write_text(
        "".join(
            json.dumps(payload, separators=(",", ":")) + "\n"
            for payload in payloads
        ),
        encoding="utf-8",
    )


class FrozenBaselineTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        Image.new("RGB", (8, 8), (20, 40, 60)).save(
            self.root / "train.png"
        )
        Image.new("RGB", (8, 8), (70, 90, 110)).save(
            self.root / "validation.png"
        )
        self.vocabulary = RelationVocabulary(
            predicates=("beside", "holding")
        )

        self.train_path = self.root / "train.jsonl"
        self.validation_path = self.root / "validation.jsonl"
        write_manifest(
            self.train_path,
            [record("train.png", [[0, 0, 1], [0, 1, 1]])],
        )
        write_manifest(
            self.validation_path,
            [record("validation.png", [[1, 0, 0]])],
        )
        self.train_manifest = DatasetManifest.load(
            self.train_path, self.vocabulary
        )
        self.validation_manifest = DatasetManifest.load(
            self.validation_path, self.vocabulary
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_photometric_jitter_matches_apache_formula(self):
        image = torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
                [[0.2, 0.4], [0.6, 0.9]],
            ],
            dtype=torch.float32,
        )
        strength = 0.3
        torch.manual_seed(777)
        factors = [
            float(
                1.0
                + (
                    torch.rand(()) * 2.0
                    - 1.0
                )
                * strength
            )
            for _ in range(3)
        ]
        expected = image * factors[0]
        mean = expected.mean(
            dim=(1, 2),
            keepdim=True,
        )
        expected = (
            expected - mean
        ) * factors[1] + mean
        luma = torch.tensor(
            [0.299, 0.587, 0.114],
            dtype=torch.float32,
        ).view(3, 1, 1)
        grey = (
            expected * luma
        ).sum(
            dim=0,
            keepdim=True,
        )
        expected = (
            expected - grey
        ) * factors[2] + grey
        expected = expected.clamp(
            0.0,
            1.0,
        )

        torch.manual_seed(777)
        actual = photometric_jitter(
            image.clone(),
            strength,
        )
        self.assertTrue(
            torch.equal(
                actual,
                expected,
            )
        )
        self.assertGreaterEqual(
            float(actual.min()),
            0.0,
        )
        self.assertLessEqual(
            float(actual.max()),
            1.0,
        )
        identity = image.clone()
        self.assertIs(
            photometric_jitter(
                identity,
                0.0,
            ),
            identity,
        )

    def test_photometric_augmentation_changes_only_image_tensor(self):
        torch.manual_seed(778)
        plain = prepare_example(
            self.train_manifest.examples[0],
            image_root=self.root,
            image_size=8,
            max_boxes=4,
            predicate_count=2,
            augment=0.0,
        )
        torch.manual_seed(778)
        augmented = prepare_example(
            self.train_manifest.examples[0],
            image_root=self.root,
            image_size=8,
            max_boxes=4,
            predicate_count=2,
            augment=0.3,
        )
        self.assertFalse(
            torch.equal(
                plain["image"],
                augmented["image"],
            )
        )
        for key in (
            "boxes",
            "box_count",
            "pair_targets",
            "predicate_targets",
            "cfa_predicate_labels",
            "object_label_indices",
            "source_id",
        ):
            self.assertTrue(
                torch.equal(
                    plain[key],
                    augmented[key],
                ),
                key,
            )

    def test_prepare_example_builds_runtime_and_multilabel_targets(self):
        prepared = prepare_example(
            self.train_manifest.examples[0],
            image_root=self.root,
            image_size=8,
            max_boxes=4,
            predicate_count=2,
        )
        self.assertEqual(tuple(prepared["image"].shape), (3, 8, 8))
        self.assertEqual(tuple(prepared["boxes"].shape), (4, 4))
        self.assertEqual(int(prepared["box_count"]), 2)
        self.assertTrue(
            torch.allclose(
                prepared["boxes"][0],
                torch.tensor([0.25, 0.5, 0.5, 1.0]),
            )
        )
        self.assertEqual(float(prepared["pair_targets"][0, 1]), 1.0)
        self.assertEqual(
            prepared["predicate_targets"][0, 1].tolist(),
            [1.0, 1.0],
        )

    def test_predicate_weighting_uses_unique_positive_pairs_and_support(self):
        examples = []
        relations_by_example = (
            ((0, 0, 1),),
            ((0, 0, 1),),
            ((0, 0, 1),),
            ((0, 1, 1),),
        )
        for index, relations in enumerate(relations_by_example):
            examples.append(
                RelationExample(
                    image=f"weight-{index}.png",
                    width=8,
                    height=8,
                    boxes_xyxy=(
                        (0.0, 0.0, 4.0, 8.0),
                        (4.0, 0.0, 8.0, 8.0),
                    ),
                    object_labels=("left", "right"),
                    relations=relations,
                )
            )
        manifest = DatasetManifest(
            examples=tuple(examples),
            annotations_sha256="a" * 64,
            vocabulary_sha256=self.vocabulary.sha256(),
        )

        none = build_predicate_weighting(
            manifest,
            predicate_count=2,
            mode="none",
            cap=20.0,
        )
        balanced = build_predicate_weighting(
            manifest,
            predicate_count=2,
            mode="balanced",
            cap=2.0,
        )
        sqrt_balanced = build_predicate_weighting(
            manifest,
            predicate_count=2,
            mode="sqrt-balanced",
            cap=20.0,
        )

        self.assertEqual(none.positive_pair_count, 4)
        self.assertEqual(none.predicate_positive_counts, (3, 1))
        self.assertEqual(none.positive_weights, (1.0, 1.0))
        self.assertAlmostEqual(
            balanced.positive_weights[0],
            1.0 / 3.0,
        )
        self.assertEqual(balanced.positive_weights[1], 2.0)
        self.assertAlmostEqual(
            sqrt_balanced.positive_weights[0],
            math.sqrt(1.0 / 3.0),
        )
        self.assertAlmostEqual(
            sqrt_balanced.positive_weights[1],
            math.sqrt(3.0),
        )

        zero_support = build_predicate_weighting(
            manifest,
            predicate_count=3,
            mode="balanced",
        )
        self.assertEqual(
            zero_support.predicate_positive_counts,
            (3, 1, 0),
        )
        self.assertEqual(
            zero_support.zero_support_predicate_indices,
            (2,),
        )
        self.assertEqual(
            zero_support.positive_weights[2],
            1.0,
        )

    def test_zero_support_negative_weight_builder(self):
        weights = build_zero_support_negative_weights(
            predicate_count=5,
            zero_support_predicate_indices=(1, 4),
            zero_support_negative_weight=0.25,
        )
        self.assertEqual(
            weights.tolist(),
            [1.0, 0.25, 1.0, 1.0, 0.25],
        )

        for value in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                build_zero_support_negative_weights(
                    predicate_count=5,
                    zero_support_predicate_indices=(1,),
                    zero_support_negative_weight=value,
                )
        with self.assertRaises(ValueError):
            build_zero_support_negative_weights(
                predicate_count=5,
                zero_support_predicate_indices=(1, 1),
                zero_support_negative_weight=0.5,
            )
        with self.assertRaises(ValueError):
            build_zero_support_negative_weights(
                predicate_count=5,
                zero_support_predicate_indices=(5,),
                zero_support_negative_weight=0.5,
            )

    def test_frozen_baseline_updates_head_not_backbone_and_evaluates(self):
        seed_everything(17)
        backbone = ToyBackbone()
        model = KFRelationModel(
            backbone,
            torch.randn(2, 6),
            model_config(),
        )
        before_backbone = backbone.conv.weight.detach().clone()
        before_head = model.pair_head.weight.detach().clone()

        freeze_backbone(model)
        dataset = RelationTrainingDataset(
            self.train_manifest,
            image_root=self.root,
            image_size=8,
            max_boxes=4,
            predicate_count=2,
        )
        baseline = FrozenBaselineConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1.0e-2,
            weight_decay=0.0,
            seed=17,
        )
        loader = make_training_loader(dataset, baseline)
        optimizer = torch.optim.AdamW(
            trainable_parameters(model),
            lr=baseline.learning_rate,
            weight_decay=baseline.weight_decay,
        )

        losses = train_epoch(
            model,
            loader,
            optimizer,
            device=torch.device("cpu"),
        )
        for value in losses.values():
            self.assertTrue(math.isfinite(value))

        self.assertTrue(
            torch.equal(backbone.conv.weight.detach(), before_backbone)
        )
        self.assertFalse(
            torch.equal(model.pair_head.weight.detach(), before_head)
        )

        report = evaluate_gt_boxes(
            model,
            self.validation_manifest,
            image_root=self.root,
            device=torch.device("cpu"),
        )
        self.assertEqual(report["examples"], 1)
        self.assertEqual(report["ground_truth_pairs"], 1)
        self.assertEqual(report["ground_truth_triplets"], 1)
        self.assertEqual(
            report["annotations_sha256"],
            self.validation_manifest.annotations_sha256,
        )

    def test_ema_updates_only_on_optimizer_boundaries(self):
        torch.manual_seed(123)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(2, 6),
            model_config(),
        )
        freeze_backbone(model)
        repeated = DatasetManifest(
            examples=self.train_manifest.examples * 4,
            annotations_sha256="e" * 64,
            vocabulary_sha256=self.vocabulary.sha256(),
        )
        dataset = RelationTrainingDataset(
            repeated,
            image_root=self.root,
            image_size=8,
            max_boxes=4,
            predicate_count=2,
        )
        loader = make_training_loader(
            dataset,
            FrozenBaselineConfig(
                epochs=1,
                batch_size=1,
                learning_rate=1.0e-2,
                weight_decay=0.0,
                seed=19,
            ),
        )
        optimizer = torch.optim.AdamW(
            trainable_parameters(model),
            lr=1.0e-2,
            weight_decay=0.0,
        )
        ema = ModelEMA(model)
        initial_ema = module_state_sha256(
            ema.ema_model
        )

        report = train_epoch(
            model,
            loader,
            optimizer,
            device=torch.device("cpu"),
            grad_accum=2,
            ema=ema,
        )

        self.assertEqual(
            int(report["micro_batches"]),
            4,
        )
        self.assertEqual(
            int(report["optimizer_steps"]),
            2,
        )
        self.assertEqual(ema.updates, 2)
        self.assertNotEqual(
            module_state_sha256(ema.ema_model),
            initial_ema,
        )
        self.assertNotEqual(
            module_state_sha256(ema.ema_model),
            module_state_sha256(model),
        )

    def test_split_leakage_box_overflow_and_image_size_mismatch_fail(self):
        leaking = DatasetManifest(
            examples=self.train_manifest.examples,
            annotations_sha256="c" * 64,
            vocabulary_sha256=self.vocabulary.sha256(),
        )
        with self.assertRaises(ValueError):
            validate_disjoint_splits(
                self.train_manifest, leaking
            )

        example = self.train_manifest.examples[0]
        too_many = RelationExample(
            image=example.image,
            width=example.width,
            height=example.height,
            boxes_xyxy=tuple(
                (float(i), 0.0, float(i + 1), 1.0)
                for i in range(5)
            ),
            object_labels=("a", "b", "c", "d", "e"),
            relations=(),
        )
        with self.assertRaises(ValueError):
            prepare_example(
                too_many,
                image_root=self.root,
                image_size=8,
                max_boxes=4,
                predicate_count=2,
            )

        released_overflow = RelationExample(
            image=example.image,
            width=example.width,
            height=example.height,
            boxes_xyxy=tuple(
                (0.0, 0.0, 1.0, 1.0)
                for _ in range(41)
            ),
            object_labels=tuple(
                "a"
                for _ in range(41)
            ),
            relations=(),
        )
        with self.assertRaisesRegex(
            ValueError,
            "41 boxes",
        ):
            prepare_example(
                released_overflow,
                image_root=self.root,
                image_size=8,
                max_boxes=40,
                predicate_count=2,
            )

        wrong_size = RelationExample(
            image=example.image,
            width=9,
            height=8,
            boxes_xyxy=example.boxes_xyxy,
            object_labels=example.object_labels,
            relations=example.relations,
        )
        with self.assertRaises(ValueError):
            prepare_example(
                wrong_size,
                image_root=self.root,
                image_size=8,
                max_boxes=4,
                predicate_count=2,
            )


if __name__ == "__main__":
    unittest.main()
