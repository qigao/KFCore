from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from apache_vocab_head import (
    SPATIAL_FLAGS_SCHEMA,
    ApacheVocabHead,
    balanced_spatial_probe_targets,
    load_predicate_spatial_flags,
)
from checkpoint import config_from_payload, load_payload, save_checkpoint
from export_onnx import (
    ENCODER_OUTPUT_NAMES,
    check_dynamic_vocabulary_parity,
    check_onnx_parity,
    export_dynamic_vocabulary_graph,
    export_encoder_graph,
    export_graph,
)
from model import KFRelationModel, RelationModelConfig
from test_model import ToyBackbone, boxes


def head_config() -> RelationModelConfig:
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
        predicate_head_contract="apache",
    )


class ApacheVocabHeadTest(unittest.TestCase):
    def test_spatial_flag_sidecar_requires_exact_vocabulary_order(self):
        payload = {
            "schema": SPATIAL_FLAGS_SCHEMA,
            "predicates": ["beside", "holding", "riding"],
            "is_spatial": [True, False, False],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spatial-flags.json"
            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            flags = load_predicate_spatial_flags(
                path,
                ["beside", "holding", "riding"],
            )
            self.assertTrue(
                torch.equal(
                    flags,
                    torch.tensor([True, False, False]),
                )
            )
            with self.assertRaises(ValueError):
                load_predicate_spatial_flags(
                    path,
                    ["holding", "beside", "riding"],
                )

            payload["is_spatial"] = [True, True, True]
            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_predicate_spatial_flags(
                    path,
                    ["beside", "holding", "riding"],
                )

    def test_routing_is_text_conditioned_and_permutation_equivariant(self):
        torch.manual_seed(51)
        head = ApacheVocabHead(
            d_model=16,
            text_dim=6,
            projection_layers=2,
        )
        bank = F.normalize(torch.randn(4, 6), dim=-1)
        alpha = head.routing_alpha(bank)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = head.routing_alpha(bank[permutation])
        self.assertTrue(
            torch.allclose(
                permuted,
                alpha[permutation],
                atol=0.0,
                rtol=0.0,
            )
        )
        novel = head.routing_alpha(
            F.normalize(torch.randn(1, 6), dim=-1)
        )
        self.assertEqual(tuple(novel.shape), (1,))
        self.assertGreaterEqual(float(novel[0]), 0.0)
        self.assertLessEqual(float(novel[0]), 1.0)

    def test_alpha_zero_and_one_select_exact_experts(self):
        torch.manual_seed(52)
        head = ApacheVocabHead(
            d_model=6,
            text_dim=6,
            projection_layers=1,
        )
        with torch.no_grad():
            head.logit_scale.fill_(0.0)
            head.logit_bias.zero_()

        bank = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            dim=-1,
        )
        semantic = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
        )
        spatial = torch.tensor(
            [[[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]]
        )

        semantic_only = head.score_query_dual(
            semantic,
            spatial,
            bank,
            alpha=torch.zeros(2),
        )
        spatial_only = head.score_query_dual(
            semantic,
            spatial,
            bank,
            alpha=torch.ones(2),
        )
        self.assertTrue(
            torch.allclose(
                semantic_only,
                torch.tensor([[[1.0, 0.0]]]),
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                spatial_only,
                torch.tensor([[[0.0, 1.0]]]),
                atol=1.0e-6,
            )
        )

    def test_scoring_does_not_modify_text_bank(self):
        torch.manual_seed(53)
        head = ApacheVocabHead(
            d_model=16,
            text_dim=6,
        )
        bank = F.normalize(torch.randn(5, 6), dim=-1)
        original = bank.clone()
        _ = head.score_query_dual(
            torch.randn(1, 2, 6),
            torch.randn(1, 2, 6),
            bank,
        )
        self.assertTrue(torch.equal(bank, original))

    def test_gate_warm_start_reduces_target_mse(self):
        torch.manual_seed(54)
        head = ApacheVocabHead(
            d_model=16,
            text_dim=6,
        )
        bank = F.normalize(torch.randn(8, 6), dim=-1)
        target = torch.tensor(
            [0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95]
        )
        with torch.no_grad():
            before = F.mse_loss(
                head.routing_alpha(bank),
                target,
            ).item()
        final = head.warm_start_gate(
            bank,
            target,
            steps=150,
            learning_rate=1.0e-2,
        )
        with torch.no_grad():
            after = F.mse_loss(
                head.routing_alpha(bank),
                target,
            ).item()
        self.assertLess(after, before)
        self.assertAlmostEqual(final, after, delta=2.0e-3)

    def test_balanced_spatial_probe_separates_simple_text_space(self):
        bank = F.normalize(
            torch.tensor(
                [
                    [-2.0, 0.0],
                    [-1.0, 0.1],
                    [-1.5, -0.1],
                    [1.0, 0.0],
                    [1.5, 0.1],
                    [2.0, -0.1],
                ],
                dtype=torch.float32,
            ),
            dim=-1,
        )
        flags = torch.tensor(
            [False, False, False, True, True, True]
        )
        target = balanced_spatial_probe_targets(
            bank,
            flags,
            steps=300,
            learning_rate=5.0e-2,
        )
        self.assertGreater(
            float(target[flags].mean()),
            float(target[~flags].mean()),
        )

    def test_model_has_real_semantic_and_spatial_queries(self):
        torch.manual_seed(55)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            head_config(),
        ).eval()
        image = torch.rand(1, 3, 8, 8)
        box_tensor, box_counts = boxes()

        with torch.inference_mode():
            (
                semantic,
                spatial,
                pair_logits,
                sub_idx,
                obj_idx,
                valid,
            ) = model.forward_encoder(
                image,
                box_tensor[:1],
                box_counts[:1],
            )
        self.assertEqual(tuple(semantic.shape), (1, 6, 6))
        self.assertEqual(tuple(spatial.shape), (1, 6, 6))
        self.assertFalse(torch.equal(semantic, spatial))
        self.assertEqual(tuple(pair_logits.shape), (1, 6))
        self.assertEqual(tuple(sub_idx.shape), (1, 6))
        self.assertEqual(tuple(obj_idx.shape), (1, 6))
        self.assertEqual(tuple(valid.shape), (1, 6))

        assert model.apache_compose_gate is not None
        self.assertTrue(
            torch.equal(
                model.apache_compose_gate.detach(),
                torch.tensor([0.1, 0.1]),
            )
        )
        self.assertFalse(model.predicate_projection.weight.requires_grad)
        self.assertFalse(model.logit_scale.requires_grad)

    def test_model_head_onnx_and_encoder_onnx_match(self):
        torch.manual_seed(56)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            head_config(),
        ).eval()
        image = torch.rand(1, 3, 8, 8)
        box_tensor, box_counts = boxes()

        with tempfile.TemporaryDirectory() as directory:
            relation_path = Path(directory) / "apache-head.onnx"
            relation_reference = export_graph(
                model,
                relation_path,
                image,
                box_tensor[:1],
                box_counts[:1],
            )
            relation_delta = check_onnx_parity(
                relation_path,
                image,
                box_tensor[:1],
                box_counts[:1],
                relation_reference,
            )

            encoder_path = Path(directory) / "apache-head-encoder.onnx"
            encoder_reference = export_encoder_graph(
                model,
                encoder_path,
                image,
                box_tensor[:1],
                box_counts[:1],
            )
            encoder_delta = check_onnx_parity(
                encoder_path,
                image,
                box_tensor[:1],
                box_counts[:1],
                encoder_reference,
                output_names=ENCODER_OUTPUT_NAMES,
            )

        self.assertLessEqual(relation_delta, 1.0e-3)
        self.assertLessEqual(encoder_delta, 1.0e-3)

    def test_dynamic_vocabulary_graph_reuses_one_onnx_for_multiple_vocab_sizes(self):
        torch.manual_seed(58)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            head_config(),
        ).eval()
        image = torch.rand(1, 3, 8, 8)
        box_tensor, box_counts = boxes()

        W3 = F.normalize(
            torch.randn(3, 6),
            dim=-1,
        )
        alpha3 = torch.tensor(
            [0.0, 0.5, 1.0],
            dtype=torch.float32,
        )
        W1 = W3[:1].clone()
        alpha1 = alpha3[:1].clone()
        W5 = F.normalize(
            torch.randn(5, 6),
            dim=-1,
        )
        assert model.apache_vocab_head is not None
        with torch.inference_mode():
            alpha5 = model.apache_vocab_head.routing_alpha(W5)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apache-dynamic-vocab.onnx"
            reference = export_dynamic_vocabulary_graph(
                model,
                path,
                image,
                box_tensor[:1],
                box_counts[:1],
                W3,
                alpha3,
            )
            self.assertEqual(
                tuple(reference[0].shape),
                (1, 6, 3),
            )
            delta = check_dynamic_vocabulary_parity(
                path,
                model,
                image,
                box_tensor[:1],
                box_counts[:1],
                [
                    (W1, alpha1),
                    (W3, alpha3),
                    (W5, alpha5),
                ],
            )

        self.assertLessEqual(delta, 1.0e-3)

    def test_dynamic_vocabulary_graph_alpha_controls_exact_expert_route(self):
        torch.manual_seed(59)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            head_config(),
        ).eval()
        image = torch.rand(1, 3, 8, 8)
        box_tensor, box_counts = boxes()
        W = F.normalize(
            torch.randn(2, 6),
            dim=-1,
        )

        with torch.inference_mode():
            (
                semantic,
                spatial,
                _,
                _,
                _,
                _,
            ) = model.forward_encoder(
                image,
                box_tensor[:1],
                box_counts[:1],
            )
            assert model.apache_vocab_head is not None
            semantic_only = model.apache_vocab_head.score_query_dual(
                semantic,
                spatial,
                W,
                alpha=torch.zeros(2),
            )
            spatial_only = model.apache_vocab_head.score_query_dual(
                semantic,
                spatial,
                W,
                alpha=torch.ones(2),
            )

        self.assertFalse(
            torch.allclose(
                semantic_only,
                spatial_only,
            )
        )

    def test_head_checkpoint_round_trip(self):
        torch.manual_seed(57)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            head_config(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apache-head.pt"
            save_checkpoint(
                path,
                model,
                backbone_model="synthetic/test-backbone",
                predicates=["beside", "holding", "riding"],
            )
            payload = load_payload(path)
            restored = config_from_payload(payload)

        self.assertEqual(
            restored.predicate_head_contract,
            "apache",
        )
        self.assertIn(
            "apache_vocab_head.gate_mlp.0.weight",
            payload["state_dict"],
        )
        self.assertIn(
            "apache_sub_text_proj.weight",
            payload["state_dict"],
        )
        self.assertIn(
            "apache_spatial_proj.0.weight",
            payload["state_dict"],
        )

    def test_apache_head_requires_context_and_fixed_text_bank(self):
        with self.assertRaises(ValueError):
            RelationModelConfig(
                predicate_head_contract="apache",
            )
        with self.assertRaises(ValueError):
            RelationModelConfig(
                pair_evidence_contract="apache",
                pair_sampler_contract="apache",
                relation_context_contract="apache",
                predicate_head_contract="apache",
                predicate_adapter_rank=1,
            )


if __name__ == "__main__":
    unittest.main()
