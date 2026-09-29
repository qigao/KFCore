from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import torch

from apache_pair_evidence import (
    BoxPromptEncoder,
    RelGeomEncoder,
    ScenePosEnc,
    SoftSpatialPool,
    contact_box,
    coverage_pair_metrics,
    union_box,
)
from checkpoint import config_from_payload, load_payload, save_checkpoint
from export_onnx import check_onnx_parity, export_graph
from model import KFRelationModel, RelationModelConfig
from test_model import ToyBackbone, boxes


def apache_config() -> RelationModelConfig:
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
    )


class ApachePairEvidenceTest(unittest.TestCase):
    def test_scene_position_encoding_starts_zero_gated(self):
        module = ScenePosEnc(8, num_freqs=4, max_octave=3.0)
        value = module(
            2,
            3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(tuple(value.shape), (1, 6, 8))
        self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_box_prompt_encoder_emits_two_corner_tokens(self):
        torch.manual_seed(1)
        module = BoxPromptEncoder(
            8,
            num_freqs=4,
            max_octave=3.0,
        )
        value = module(
            torch.tensor(
                [[[0.1, 0.2, 0.6, 0.8]]],
                dtype=torch.float32,
            )
        )
        self.assertEqual(tuple(value.shape), (1, 1, 2, 8))
        pair = module.encode_pairs(
            torch.tensor([[[0.1, 0.2, 0.4, 0.5]]]),
            torch.tensor([[[0.5, 0.2, 0.8, 0.7]]]),
        )
        self.assertEqual(tuple(pair.shape), (1, 1, 4, 8))

    def test_soft_spatial_pool_is_global_and_mask_bias_starts_disabled(self):
        torch.manual_seed(2)
        module = SoftSpatialPool(
            8,
            n_heads=2,
            num_freqs=4,
            max_octave=3.0,
        ).eval()
        boxes_t = torch.tensor(
            [[[0.20, 0.20, 0.15, 0.15]]],
            dtype=torch.float32,
        )
        feature_map = torch.zeros(1, 8, 2, 2)
        baseline = module(feature_map, boxes_t)

        changed = feature_map.clone()
        changed[:, :, 1, 1] = 5.0
        global_read = module(changed, boxes_t)
        self.assertFalse(torch.allclose(baseline, global_read))

        zeros = torch.zeros(1, 1, 4)
        ones = torch.ones(1, 1, 4)
        masked_zero = module(changed, boxes_t, coverage=zeros)
        masked_one = module(changed, boxes_t, coverage=ones)
        self.assertTrue(
            torch.allclose(
                masked_zero,
                masked_one,
                atol=0.0,
                rtol=0.0,
            )
        )
        self.assertEqual(
            float(module.cov_lambda.abs().sum()),
            0.0,
        )

    def test_union_and_contact_boxes_match_reference_gap_semantics(self):
        subject = torch.tensor(
            [[[0.20, 0.20, 0.20, 0.20]]]
        )
        object_ = torch.tensor(
            [[[0.70, 0.20, 0.20, 0.20]]]
        )
        union = union_box(subject, object_)
        contact = contact_box(subject, object_)
        self.assertTrue(
            torch.allclose(
                union,
                torch.tensor(
                    [[[0.45, 0.20, 0.70, 0.20]]]
                ),
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                contact,
                torch.tensor(
                    [[[0.45, 0.20, 0.30, 0.20]]]
                ),
                atol=1.0e-6,
            )
        )

        overlapping = torch.tensor(
            [[[0.30, 0.20, 0.20, 0.20]]]
        )
        overlap_contact = contact_box(subject, overlapping)
        self.assertTrue(
            torch.allclose(
                overlap_contact,
                torch.tensor(
                    [[[0.25, 0.20, 0.10, 0.20]]]
                ),
                atol=1.0e-6,
            )
        )

    def test_reference_geometry_has_exact_19_column_contract(self):
        subject = torch.tensor(
            [[[0.25, 0.25, 0.20, 0.20]]],
            dtype=torch.float32,
        )
        object_ = torch.tensor(
            [[[0.35, 0.25, 0.20, 0.20]]],
            dtype=torch.float32,
        )
        features = RelGeomEncoder.features(
            subject,
            object_,
        )
        self.assertEqual(tuple(features.shape), (1, 1, 19))

        def n(value: float) -> float:
            return 10.0 * math.tanh(value / 10.0)

        self.assertAlmostEqual(
            float(features[0, 0, 0]),
            n(0.5),
            places=5,
        )
        self.assertAlmostEqual(
            float(features[0, 0, 1]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(features[0, 0, 7]),
            n(1.0 / 3.0),
            places=5,
        )
        self.assertAlmostEqual(
            float(features[0, 0, 8]),
            n(0.5),
            places=5,
        )
        self.assertAlmostEqual(
            float(features[0, 0, 9]),
            n(0.5),
            places=5,
        )
        self.assertAlmostEqual(
            float(features[0, 0, 12]),
            n(1.0),
            places=5,
        )
        self.assertAlmostEqual(
            float(features[0, 0, 13]),
            0.0,
            places=5,
        )
        self.assertAlmostEqual(
            float(features[0, 0, 15]),
            n(1.0),
            places=5,
        )
        self.assertAlmostEqual(
            float(features[0, 0, 16]),
            n(1.0),
            places=5,
        )
        self.assertAlmostEqual(
            float(features[0, 0, 17]),
            n(1.0 / 3.0),
            places=5,
        )
        self.assertAlmostEqual(
            float(features[0, 0, 18]),
            n(0.5),
            places=5,
        )

    def test_region_coverage_metrics_are_pairwise(self):
        coverage = torch.tensor(
            [
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                ]
            ]
        )
        region_iou, region_contact = coverage_pair_metrics(
            coverage
        )
        self.assertAlmostEqual(
            float(region_iou[0, 0, 1]),
            1.0 / 3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(region_contact[0, 0, 1]),
            0.5,
            places=6,
        )

    def test_apache_model_exposes_reference_pair_evidence_shapes(self):
        torch.manual_seed(3)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            apache_config(),
        )
        image = torch.rand(1, 3, 8, 8)
        box_tensor, _ = boxes()
        patch_features = model._fused_patch_features(image)
        valid_boxes = torch.tensor(
            [[True, True, True, True]]
        )
        subject_index = torch.tensor(
            [[0, 1, 2]], dtype=torch.int64
        )
        object_index = torch.tensor(
            [[1, 0, 3]], dtype=torch.int64
        )
        valid = torch.tensor([[True, True, True]])

        (
            pair_tokens,
            box_tokens,
            anchors,
            geometry,
        ) = model._apache_pair_evidence(
            patch_features,
            box_tensor[:1],
            valid_boxes,
            subject_index,
            object_index,
            valid,
        )
        self.assertEqual(tuple(pair_tokens.shape), (1, 3, 16))
        self.assertEqual(tuple(box_tokens.shape), (1, 3, 4, 16))
        self.assertEqual(tuple(anchors.shape), (1, 3, 4, 4))
        self.assertEqual(tuple(geometry.shape), (1, 3, 16))
        self.assertTrue(torch.isfinite(pair_tokens).all())
        self.assertTrue(torch.isfinite(geometry).all())

    def test_apache_forward_and_onnx_export(self):
        torch.manual_seed(4)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            apache_config(),
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

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apache-pair-evidence.onnx"
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

    def test_checkpoint_round_trip_preserves_apache_contract(self):
        torch.manual_seed(5)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            apache_config(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apache.pt"
            save_checkpoint(
                path,
                model,
                backbone_model="synthetic/test-backbone",
                predicates=["beside", "holding", "riding"],
            )
            payload = load_payload(path)
            restored = config_from_payload(payload)

        self.assertEqual(
            restored.pair_evidence_contract,
            "apache",
        )
        self.assertIn(
            "apache_spatial_pool.base_query",
            payload["state_dict"],
        )
        self.assertIn(
            "apache_geometry_encoder.mlp.0.weight",
            payload["state_dict"],
        )
        self.assertIn(
            "apache_pair_projection.weight",
            payload["state_dict"],
        )

    def test_apache_contract_rejects_legacy_evidence_overlays(self):
        with self.assertRaises(ValueError):
            RelationModelConfig(
                pair_evidence_contract="apache",
                pair_visual_evidence="contact",
            )
        with self.assertRaises(ValueError):
            RelationModelConfig(
                pair_evidence_contract="apache",
                pair_geometry_evidence="rich",
            )


if __name__ == "__main__":
    unittest.main()
