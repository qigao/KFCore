from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from apache_context import (
    ApacheDeformableRelRead,
    ApacheRelationInteractionBlock,
    ApacheRelationTransformer,
)
from checkpoint import config_from_payload, load_payload, save_checkpoint
from export_onnx import check_onnx_parity, export_graph
from model import KFRelationModel, RelationModelConfig
from test_model import ToyBackbone, boxes


def context_config() -> RelationModelConfig:
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
        pair_evidence_contract="apache",
        pair_sampler_contract="apache",
        relation_context_contract="apache",
        apache_context_dropout=0.0,
        apache_box_token_dropout=0.0,
    )


class ApacheRelationContextTest(unittest.TestCase):
    def test_relation_transformer_masks_padded_box_tokens(self):
        torch.manual_seed(41)
        module = ApacheRelationTransformer(
            d_model=16,
            scene_dim=8,
            n_self_layers=2,
            n_cross_layers=2,
            n_heads=8,
            dropout=0.0,
        ).eval()
        pair = torch.randn(1, 3, 16)
        scene = torch.randn(1, 8, 2, 2)
        box_tokens = torch.randn(1, 3, 4, 16)
        padding = torch.tensor([[False, False, True]])

        changed = box_tokens.clone()
        changed[:, 2] = 1000.0
        with torch.inference_mode():
            first = module(
                pair,
                scene,
                box_tokens=box_tokens,
                pair_padding_mask=padding,
            )
            second = module(
                pair,
                scene,
                box_tokens=changed,
                pair_padding_mask=padding,
            )
        self.assertTrue(
            torch.allclose(
                first[:, :2],
                second[:, :2],
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )

    def test_relation_transformer_box_token_dropout_hides_all_box_tokens(self):
        torch.manual_seed(42)
        module = ApacheRelationTransformer(
            d_model=16,
            scene_dim=8,
            n_self_layers=1,
            n_cross_layers=2,
            n_heads=8,
            dropout=0.0,
        ).eval()
        pair = torch.randn(1, 2, 16)
        scene = torch.randn(1, 8, 2, 2)
        box_tokens = torch.randn(1, 2, 4, 16)
        changed = torch.randn_like(box_tokens) * 100.0
        drop = torch.tensor([True])

        with torch.inference_mode():
            first = module(
                pair,
                scene,
                box_tokens=box_tokens,
                box_token_drop=drop,
            )
            second = module(
                pair,
                scene,
                box_tokens=changed,
                box_token_drop=drop,
            )
        self.assertTrue(
            torch.allclose(
                first,
                second,
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )

    def test_deformable_read_starts_as_exact_identity(self):
        torch.manual_seed(43)
        module = ApacheDeformableRelRead(
            d_model=16,
            n_points=4,
            n_heads=8,
            null_slots=2,
        ).eval()
        query = torch.randn(1, 3, 16)
        scene = torch.randn(1, 16, 3, 3)
        anchors = torch.tensor(
            [
                [
                    [
                        [0.2, 0.2, 0.2, 0.2],
                        [0.4, 0.2, 0.2, 0.2],
                        [0.3, 0.2, 0.4, 0.2],
                        [0.3, 0.2, 0.1, 0.2],
                    ],
                    [
                        [0.3, 0.5, 0.2, 0.2],
                        [0.6, 0.5, 0.2, 0.2],
                        [0.45, 0.5, 0.5, 0.2],
                        [0.45, 0.5, 0.1, 0.2],
                    ],
                    [
                        [0.2, 0.8, 0.2, 0.2],
                        [0.7, 0.8, 0.2, 0.2],
                        [0.45, 0.8, 0.7, 0.2],
                        [0.45, 0.8, 0.3, 0.2],
                    ],
                ]
            ],
            dtype=torch.float32,
        )
        with torch.inference_mode():
            output = module(query, scene, anchors)
        self.assertTrue(torch.equal(output, query))
        self.assertEqual(float(module.gamma.abs().sum()), 0.0)
        self.assertGreater(
            float(module.offset_mlp.bias.abs().sum()),
            0.0,
        )
        self.assertIsNotNone(module.null_vectors)
        assert module.null_vectors is not None
        self.assertEqual(
            tuple(module.null_vectors.shape),
            (8, 4, 2, 2),
        )

    def test_deformable_read_learns_scene_residual_when_gate_opens(self):
        torch.manual_seed(44)
        module = ApacheDeformableRelRead(
            d_model=16,
            n_points=4,
            n_heads=8,
            null_slots=2,
        ).eval()
        with torch.no_grad():
            module.gamma.fill_(1.0)
        query = torch.randn(1, 1, 16)
        scene = torch.randn(1, 16, 2, 2)
        anchors = torch.tensor(
            [[[[0.5, 0.5, 0.2, 0.2]] * 4]],
            dtype=torch.float32,
        )
        with torch.inference_mode():
            output = module(query, scene, anchors)
        self.assertFalse(torch.allclose(output, query))
        self.assertTrue(torch.isfinite(output).all())

    def test_interaction_block_padding_does_not_pollute_valid_queries(self):
        torch.manual_seed(45)
        module = ApacheRelationInteractionBlock(
            d_model=16,
            scene_dim=8,
            n_dependency_layers=2,
            n_grounding_layers=1,
            n_heads=8,
            dropout=0.0,
        ).eval()
        query = torch.randn(1, 3, 16)
        changed = query.clone()
        changed[:, 2] = 1000.0
        scene = torch.randn(1, 8, 2, 2)
        padding = torch.tensor([[False, False, True]])

        with torch.inference_mode():
            first = module(
                query,
                scene,
                query_padding_mask=padding,
            )
            second = module(
                changed,
                scene,
                query_padding_mask=padding,
            )
        self.assertTrue(
            torch.allclose(
                first[:, :2],
                second[:, :2],
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        )

    def test_context_config_requires_apache_pair_stack(self):
        with self.assertRaises(ValueError):
            RelationModelConfig(
                relation_context_contract="apache",
            )
        with self.assertRaises(ValueError):
            RelationModelConfig(
                pair_evidence_contract="apache",
                relation_context_contract="apache",
            )

    def test_model_apache_context_activates_box_prompts_and_freezes_legacy_transformer(self):
        torch.manual_seed(46)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            context_config(),
        )
        assert model.apache_box_prompt_encoder is not None
        self.assertTrue(
            any(
                parameter.requires_grad
                for parameter in model.apache_box_prompt_encoder.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.relation_transformer.parameters()
            )
        )
        self.assertIsNotNone(model.apache_relation_transformer)
        self.assertIsNotNone(model.apache_deformable_read)
        self.assertIsNotNone(model.apache_relation_interaction)

    def test_model_apache_context_forward_and_onnx_parity(self):
        torch.manual_seed(47)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            context_config(),
        ).eval()
        image = torch.rand(1, 3, 8, 8)
        box_tensor, box_counts = boxes()

        with torch.inference_mode():
            runtime = model(
                image,
                box_tensor[:1],
                box_counts[:1],
            )
        self.assertEqual(tuple(runtime[0].shape), (1, 6, 3))
        self.assertEqual(tuple(runtime[1].shape), (1, 6))
        self.assertTrue(torch.isfinite(runtime[0]).all())
        self.assertTrue(
            torch.isfinite(runtime[1][runtime[4]]).all()
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apache-context.onnx"
            reference = export_graph(
                model,
                path,
                image,
                box_tensor[:1],
                box_counts[:1],
            )
            delta = check_onnx_parity(
                path,
                image,
                box_tensor[:1],
                box_counts[:1],
                reference,
            )
        self.assertLessEqual(delta, 1.0e-3)

    def test_context_checkpoint_round_trip(self):
        torch.manual_seed(48)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            context_config(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apache-context.pt"
            save_checkpoint(
                path,
                model,
                backbone_model="synthetic/test-backbone",
                predicates=["beside", "holding", "riding"],
            )
            payload = load_payload(path)
            restored = config_from_payload(payload)

        self.assertEqual(
            restored.relation_context_contract,
            "apache",
        )
        self.assertIn(
            "apache_relation_transformer.scene_proj.weight",
            payload["state_dict"],
        )
        self.assertIn(
            "apache_deformable_read.offset_mlp.weight",
            payload["state_dict"],
        )
        self.assertIn(
            "apache_relation_interaction.dependency_layers.0.self_attn.in_proj_weight",
            payload["state_dict"],
        )


if __name__ == "__main__":
    unittest.main()
