from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from apache_training_recipe import (
    ApacheTrainingRecipeConfig,
    build_reference_optimizer,
    build_reference_scheduler,
    configure_backbone_trainability,
    gradient_health,
)
from model import BackboneAdapter, KFRelationModel, RelationModelConfig
from test_model import boxes


class DummyEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 8))


class DummyDinoInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            3,
            8,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.norm = nn.LayerNorm(8)
        self.embeddings = DummyEmbeddings()


class RecipeBackbone(BackboneAdapter):
    hidden_size = 8
    patch_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.model = DummyDinoInner()

    def forward_taps(
        self,
        image: torch.Tensor,
        taps,
    ) -> list[torch.Tensor]:
        base = self.model.conv(image)
        return [
            base + float(index) * 0.1
            for index, _ in enumerate(taps)
        ]


def build_model() -> KFRelationModel:
    return KFRelationModel(
        RecipeBackbone(),
        torch.randn(3, 6),
        RelationModelConfig(
            image_size=8,
            max_boxes=4,
            pair_budget=6,
            hidden_dim=16,
            geometry_dim=8,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            tap_indices=(-3, -2, -1),
        ),
    )


class ApacheTrainingRecipeTest(unittest.TestCase):
    def test_reference_defaults(self):
        config = ApacheTrainingRecipeConfig()
        self.assertEqual(config.head_lr, 4.0e-4)
        self.assertEqual(config.backbone_lr, 5.0e-5)
        self.assertEqual(config.weight_decay, 1.0e-4)
        self.assertEqual(config.epochs, 12)
        self.assertEqual(config.warmup_steps, 500)
        self.assertEqual(config.min_lr_factor, 0.01)
        self.assertEqual(config.clip_grad, 1.0)
        self.assertEqual(config.backbone_mode, "full")

    def test_full_mode_freezes_only_unused_final_backbone_outputs(self):
        model = build_model()
        report = configure_backbone_trainability(
            model,
            mode="full",
        )
        inner = model.backbone.model

        self.assertTrue(inner.conv.weight.requires_grad)
        self.assertFalse(inner.norm.weight.requires_grad)
        self.assertFalse(inner.norm.bias.requires_grad)
        self.assertFalse(
            inner.embeddings.mask_token.requires_grad
        )
        self.assertGreater(
            int(report["trainable_parameter_count"]),
            0,
        )
        self.assertIn(
            "backbone.model.norm",
            report["frozen_unused_modules"],
        )
        self.assertIn(
            "backbone.model.embeddings.mask_token",
            report["frozen_unused_modules"],
        )

    def test_frozen_mode_has_no_trainable_backbone_parameters(self):
        model = build_model()
        report = configure_backbone_trainability(
            model,
            mode="frozen",
        )
        self.assertEqual(
            int(report["trainable_parameter_count"]),
            0,
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.backbone.parameters()
            )
        )

    def test_optimizer_uses_reference_lr_and_decay_groups(self):
        model = build_model()
        config = ApacheTrainingRecipeConfig()
        optimizer, report = build_reference_optimizer(
            model,
            config,
        )

        groups = {
            str(group["group_name"]): group
            for group in optimizer.param_groups
        }
        self.assertEqual(
            float(groups["head_decay"]["lr"]),
            config.head_lr,
        )
        self.assertEqual(
            float(groups["head_no_decay"]["lr"]),
            config.head_lr,
        )
        self.assertEqual(
            float(groups["backbone_decay"]["lr"]),
            config.backbone_lr,
        )
        self.assertEqual(
            float(groups["backbone_no_decay"]["lr"]),
            config.backbone_lr,
        )
        self.assertEqual(
            float(groups["head_no_decay"]["weight_decay"]),
            0.0,
        )
        self.assertEqual(
            float(groups["backbone_no_decay"]["weight_decay"]),
            0.0,
        )
        self.assertEqual(
            float(groups["head_decay"]["weight_decay"]),
            config.weight_decay,
        )
        self.assertEqual(
            float(groups["backbone_decay"]["weight_decay"]),
            config.weight_decay,
        )

        reported = {
            item["name"]: item
            for item in report["groups"]
        }
        self.assertGreater(
            int(reported["backbone_decay"]["parameter_count"]),
            0,
        )
        self.assertGreater(
            int(reported["head_decay"]["parameter_count"]),
            0,
        )

    def test_scheduler_matches_warmup_cosine_floor(self):
        model = build_model()
        config = ApacheTrainingRecipeConfig(
            epochs=12,
            warmup_steps=500,
            min_lr_factor=0.01,
        )
        optimizer, _ = build_reference_optimizer(
            model,
            config,
        )
        scheduler, report = build_reference_scheduler(
            optimizer,
            config,
            steps_per_epoch=100,
        )
        warmup = int(report["warmup_steps"])
        total = int(report["total_steps"])

        fn = scheduler.lr_lambdas[0]
        self.assertAlmostEqual(
            float(fn(0)),
            1.0e-3,
            places=7,
        )
        self.assertAlmostEqual(
            float(fn(warmup)),
            1.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(fn(total)),
            0.01,
            places=7,
        )

    def test_full_mode_backbone_receives_finite_nonzero_gradient(self):
        torch.manual_seed(91)
        model = build_model()
        optimizer, _ = build_reference_optimizer(
            model,
            ApacheTrainingRecipeConfig(),
        )
        image = torch.rand(1, 3, 8, 8)
        box_tensor, box_counts = boxes()

        optimizer.zero_grad(set_to_none=True)
        runtime = model(
            image,
            box_tensor[:1],
            box_counts[:1],
        )
        loss = (
            runtime[0].square().mean()
            + runtime[1][runtime[4]].square().mean()
        )
        loss.backward()

        health = gradient_health(model)
        self.assertGreater(
            float(health["backbone_gradient_norm"]),
            0.0,
        )
        self.assertGreater(
            int(health["backbone_gradient_tensors"]),
            0,
        )
        inner = model.backbone.model
        self.assertIsNotNone(inner.conv.weight.grad)
        self.assertIsNone(inner.norm.weight.grad)
        self.assertIsNone(
            inner.embeddings.mask_token.grad
        )


if __name__ == "__main__":
    unittest.main()
