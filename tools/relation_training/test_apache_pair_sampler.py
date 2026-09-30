from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from apache_pair_sampler import (
    RELEASED_FINAL_BUDGET,
    RELEASED_GEO_BUDGET,
    ApacheRelatednessPairSampler,
    PairOpportunityTable,
)
from checkpoint import config_from_payload, load_payload, save_checkpoint
from export_onnx import check_onnx_parity, export_graph
from losses import RelationLossConfig, supervised_relation_loss
from model import KFRelationModel, RelationModelConfig
from test_model import ToyBackbone, boxes


def sampler_config() -> RelationModelConfig:
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
    )


class ApacheRelatednessSamplerTest(unittest.TestCase):
    def test_released_default_budgets_are_400_to_128(self):
        sampler = ApacheRelatednessPairSampler(
            feature_dim=8,
        )
        self.assertEqual(
            sampler.geo_budget,
            RELEASED_GEO_BUDGET,
        )
        self.assertEqual(
            sampler.final_budget,
            RELEASED_FINAL_BUDGET,
        )

    def test_inference_keeps_only_valid_ordered_pairs_and_pads(self):
        torch.manual_seed(31)
        sampler = ApacheRelatednessPairSampler(
            feature_dim=8,
            geo_budget=8,
            final_budget=6,
            rel_dim=4,
        )
        box_tensor, _ = boxes()
        features = torch.randn(1, 4, 8)
        output = sampler(
            box_tensor[:1],
            features,
            torch.tensor([2], dtype=torch.int64),
        )

        self.assertEqual(tuple(output.sub_idx.shape), (1, 6))
        self.assertEqual(tuple(output.obj_idx.shape), (1, 6))
        self.assertEqual(tuple(output.pair_logits.shape), (1, 6))
        self.assertEqual(int(output.valid_mask.sum()), 2)
        selected = {
            (
                int(output.sub_idx[0, slot]),
                int(output.obj_idx[0, slot]),
            )
            for slot in range(6)
            if bool(output.valid_mask[0, slot])
        }
        self.assertEqual(selected, {(0, 1), (1, 0)})
        self.assertTrue(
            torch.isfinite(
                output.pair_logits[output.valid_mask]
            ).all()
        )

    def test_training_forces_positive_and_swapped_pair_through_both_stages(self):
        torch.manual_seed(32)
        sampler = ApacheRelatednessPairSampler(
            feature_dim=8,
            geo_budget=2,
            final_budget=2,
            rel_dim=4,
            swap_include=True,
        )
        box_tensor, _ = boxes()
        pair_targets = torch.zeros(1, 4, 4)
        pair_targets[0, 0, 3] = 1.0
        output = sampler(
            box_tensor[:1],
            torch.randn(1, 4, 8),
            torch.tensor([4], dtype=torch.int64),
            pair_targets=pair_targets,
        )

        selected = {
            (
                int(output.sub_idx[0, slot]),
                int(output.obj_idx[0, slot]),
            )
            for slot in range(2)
            if bool(output.valid_mask[0, slot])
        }
        self.assertEqual(selected, {(0, 3), (3, 0)})
        self.assertTrue(torch.isfinite(output.geo_loss))
        self.assertTrue(torch.isfinite(output.relatedness_loss))

    def test_relatedness_projection_is_asymmetric(self):
        torch.manual_seed(33)
        sampler = ApacheRelatednessPairSampler(
            feature_dim=8,
            geo_budget=4,
            final_budget=2,
            rel_dim=4,
        )
        features = torch.randn(1, 2, 8)
        subject = sampler.f_sub(features)
        object_ = sampler.f_obj(features)
        score_01 = (
            subject[0, 0] * object_[0, 1]
        ).sum() / 2.0
        score_10 = (
            subject[0, 1] * object_[0, 0]
        ).sum() / 2.0
        self.assertFalse(
            torch.allclose(score_01, score_10)
        )

    def test_pair_opportunity_table_loads_reference_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair_opportunity.npz"
            np.savez_compressed(
                path,
                rate=np.asarray(
                    [0.0, 0.1, 0.9, 0.2],
                    dtype=np.float32,
                ),
                opportunities=np.asarray(
                    [100, 50, 75, 10],
                    dtype=np.int64,
                ),
                relations=np.asarray(
                    [0, 5, 68, 2],
                    dtype=np.int64,
                ),
                num_cats=np.int32(2),
                min_support=np.int32(50),
                meta=np.asarray(["{}"]),
            )
            table = PairOpportunityTable.load(path)

        self.assertEqual(table.num_cats, 2)
        self.assertEqual(table.min_support, 50)
        self.assertTrue(
            torch.equal(
                table.trusted,
                torch.tensor([True, True, True, False]),
            )
        )
        stats = table.stats(negative_floor=0.3)
        self.assertEqual(stats["trusted_category_pairs"], 3)
        self.assertGreaterEqual(
            float(stats["median_trusted_negative_weight"]),
            0.3,
        )

    def test_statistical_pu_negative_weights_match_reference_formula(self):
        sampler = ApacheRelatednessPairSampler(
            feature_dim=8,
            geo_budget=4,
            final_budget=2,
            rel_dim=4,
            negative_weight=0.3,
        )
        sampler.set_negative_rates(
            torch.tensor([0.0, 0.1, 0.9, 0.2]),
            torch.tensor([True, True, True, False]),
            2,
        )
        weights = sampler._pu_neg_weight(
            torch.tensor([0, 0, 1, -1]),
            torch.tensor([0, 1, 0, 0]),
            torch.zeros(4),
        )
        self.assertTrue(
            torch.allclose(
                weights,
                torch.tensor([1.0, 0.9, 0.3, 0.3]),
                atol=1.0e-6,
            )
        )

    def test_statistical_weights_drive_selected_background_pair_weights(self):
        torch.manual_seed(341)
        sampler = ApacheRelatednessPairSampler(
            feature_dim=8,
            geo_budget=8,
            final_budget=6,
            rel_dim=4,
            negative_weight=0.3,
        )
        # Every trusted category pair has zero interaction rate, so all
        # unlabelled selected pairs must receive weight 1 rather than 0.3.
        sampler.set_negative_rates(
            torch.zeros(4),
            torch.ones(4, dtype=torch.bool),
            2,
        )
        box_tensor, _ = boxes()
        pair_targets = torch.zeros(1, 4, 4)
        pair_targets[0, 0, 1] = 1.0
        entity_labels = torch.tensor(
            [[0, 1, 0, 1]],
            dtype=torch.int64,
        )
        output = sampler(
            box_tensor[:1],
            torch.randn(1, 4, 8),
            torch.tensor([4], dtype=torch.int64),
            pair_targets=pair_targets,
            entity_labels=entity_labels,
        )
        assert output.pair_negative_weights is not None
        valid_weights = output.pair_negative_weights[
            output.valid_mask
        ]
        self.assertTrue(
            torch.allclose(
                valid_weights,
                torch.ones_like(valid_weights),
            )
        )

    def test_pu_negative_floor_adds_only_unlabelled_relatedness_mass(self):
        torch.manual_seed(34)
        low = ApacheRelatednessPairSampler(
            feature_dim=8,
            geo_budget=8,
            final_budget=6,
            rel_dim=4,
            negative_weight=0.0,
        )
        high = ApacheRelatednessPairSampler(
            feature_dim=8,
            geo_budget=8,
            final_budget=6,
            rel_dim=4,
            negative_weight=0.3,
        )
        high.load_state_dict(low.state_dict())

        box_tensor, _ = boxes()
        features = torch.randn(1, 4, 8)
        pair_targets = torch.zeros(1, 4, 4)
        pair_targets[0, 0, 1] = 1.0

        low_out = low(
            box_tensor[:1],
            features,
            torch.tensor([4], dtype=torch.int64),
            pair_targets=pair_targets,
        )
        high_out = high(
            box_tensor[:1],
            features,
            torch.tensor([4], dtype=torch.int64),
            pair_targets=pair_targets,
        )

        self.assertTrue(
            torch.allclose(
                low_out.geo_loss,
                high_out.geo_loss,
            )
        )
        self.assertGreaterEqual(
            float(high_out.relatedness_loss),
            float(low_out.relatedness_loss),
        )

    def test_model_training_uses_sampler_reference_losses(self):
        torch.manual_seed(35)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            sampler_config(),
        )
        image = torch.rand(1, 3, 8, 8)
        box_tensor, box_counts = boxes()
        pair_targets = torch.zeros(1, 4, 4)
        pair_targets[0, 0, 1] = 1.0
        predicate_targets = torch.zeros(1, 4, 4, 3)
        predicate_targets[0, 0, 1, 0] = 1.0

        outputs = model.forward_training(
            image,
            box_tensor[:1],
            box_counts[:1],
            pair_targets=pair_targets,
        )
        self.assertIsNotNone(outputs.sampler_geo_loss)
        self.assertIsNotNone(
            outputs.sampler_relatedness_loss
        )
        losses = supervised_relation_loss(
            outputs,
            pair_targets,
            predicate_targets,
            RelationLossConfig(),
        )
        assert outputs.sampler_geo_loss is not None
        assert outputs.sampler_relatedness_loss is not None
        self.assertTrue(
            torch.allclose(
                losses["sampler_loss"],
                outputs.sampler_geo_loss,
            )
        )
        self.assertTrue(
            torch.allclose(
                losses["pair_loss"],
                outputs.sampler_relatedness_loss,
            )
        )

    def test_model_inference_uses_stage2_pair_logits_without_targets(self):
        torch.manual_seed(36)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            sampler_config(),
        ).eval()
        image = torch.rand(1, 3, 8, 8)
        box_tensor, box_counts = boxes()
        with torch.inference_mode():
            runtime = model(
                image,
                box_tensor[:1],
                box_counts[:1],
            )
        _, pair_logits, sub_idx, obj_idx, valid = runtime
        self.assertEqual(tuple(pair_logits.shape), (1, 6))
        self.assertEqual(tuple(sub_idx.shape), (1, 6))
        self.assertEqual(tuple(obj_idx.shape), (1, 6))
        self.assertTrue(torch.isfinite(pair_logits[valid]).all())

    def test_training_requires_pair_targets_for_apache_sampler(self):
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            sampler_config(),
        )
        image = torch.rand(1, 3, 8, 8)
        box_tensor, box_counts = boxes()
        with self.assertRaises(ValueError):
            model.forward_training(
                image,
                box_tensor[:1],
                box_counts[:1],
            )

    def test_apache_sampler_onnx_matches_native_runtime(self):
        torch.manual_seed(37)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            sampler_config(),
        ).eval()
        image = torch.rand(1, 3, 8, 8)
        box_tensor, box_counts = boxes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apache-sampler.onnx"
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

    def test_checkpoint_round_trip_preserves_sampler_contract(self):
        torch.manual_seed(38)
        config = sampler_config()
        config = RelationModelConfig(
            **{
                **config.__dict__,
                "apache_pair_negative_floor": 0.42,
            }
        )
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            config,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apache-sampler.pt"
            save_checkpoint(
                path,
                model,
                backbone_model="synthetic/test-backbone",
                predicates=["beside", "holding", "riding"],
            )
            payload = load_payload(path)
            restored = config_from_payload(payload)

        self.assertEqual(
            restored.pair_sampler_contract,
            "apache",
        )
        self.assertAlmostEqual(
            restored.apache_pair_negative_floor,
            0.42,
            places=7,
        )
        self.assertIn(
            "apache_pair_sampler.geo_scorer.0.weight",
            payload["state_dict"],
        )
        self.assertIn(
            "apache_pair_sampler.f_sub.1.weight",
            payload["state_dict"],
        )
        self.assertIn(
            "apache_pair_sampler.f_obj.1.weight",
            payload["state_dict"],
        )

    def test_apache_sampler_requires_apache_pair_evidence(self):
        with self.assertRaises(ValueError):
            RelationModelConfig(
                pair_sampler_contract="apache",
            )


if __name__ == "__main__":
    unittest.main()
