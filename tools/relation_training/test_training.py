from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch
from torch import nn

from benchmark import DatasetManifest, RelationExample, RelationVocabulary
from model import BackboneAdapter, KFRelationModel, RelationModelConfig
from training import (
    FrozenBaselineConfig,
    RelationTrainingDataset,
    build_predicate_weighting,
    evaluate_gt_boxes,
    freeze_backbone,
    make_training_loader,
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

        with self.assertRaises(ValueError):
            build_predicate_weighting(
                manifest,
                predicate_count=3,
                mode="balanced",
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
