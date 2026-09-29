from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from checkpoint import config_from_payload, load_payload, save_checkpoint
from export_onnx import INPUT_NAMES, OUTPUT_NAMES, check_onnx_parity, export_graph
from losses import RelationLossConfig, supervised_relation_loss
from model import (
    BackboneAdapter,
    HFDinoV3Backbone,
    KFRelationModel,
    MetaDinoV3Backbone,
    OfficialDinoV3Backbone,
    RelationModelConfig,
    TimmDinoV3Backbone,
)


class ToyBackbone(BackboneAdapter):
    hidden_size = 8
    patch_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, self.hidden_size, kernel_size=2, stride=2)

    def forward_taps(self, image: torch.Tensor, taps) -> list[torch.Tensor]:
        base = self.conv(image)
        return [base + float(index) * 0.1 for index, _ in enumerate(taps)]


class FakeOfficialModel(nn.Module):
    embed_dim = 8
    patch_size = 2
    n_blocks = 4

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, self.embed_dim, kernel_size=2, stride=2, bias=False)

    def get_intermediate_layers(self, image, *, n, reshape, norm):
        if not reshape or norm:
            raise AssertionError("official adapter requested the wrong intermediate-layer contract")
        base = self.conv(image)
        return tuple(base + float(index) * 0.1 for index in n)


class FakeHFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=2, stride=2, bias=False)

    def forward(self, pixel_values, output_hidden_states, return_dict):
        self.assert_contract(output_hidden_states, return_dict)
        patches = self.conv(pixel_values).flatten(2).transpose(1, 2)
        prefix = torch.zeros(
            patches.shape[0], 5, patches.shape[2],
            dtype=patches.dtype, device=patches.device
        )
        tokens = torch.cat((prefix, patches), dim=1)
        return SimpleNamespace(
            hidden_states=tuple(tokens * float(i + 1) for i in range(4))
        )

    @staticmethod
    def assert_contract(output_hidden_states, return_dict) -> None:
        if not output_hidden_states or not return_dict:
            raise AssertionError("adapter did not request hidden states")


class FakeMetaModel(nn.Module):
    embed_dim = 8
    patch_size = 2
    n_blocks = 8

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, self.embed_dim, kernel_size=2, stride=2, bias=False)
        self.requested_layers = None

    def get_intermediate_layers(
        self, image, *, n, reshape, return_class_token=False,
        return_extra_tokens=False, norm=True
    ):
        if not reshape or return_class_token or return_extra_tokens or norm:
            raise AssertionError("official adapter requested an unexpected contract")
        self.requested_layers = list(n)
        base = self.conv(image)
        return tuple(base + float(index) * 0.01 for index in n)


class FakeTimmModel(nn.Module):
    num_features = 8
    pretrained_cfg = {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    }

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = SimpleNamespace(patch_size=(2, 2))
        self.blocks = [object() for _ in range(8)]
        self.conv = nn.Conv2d(3, self.num_features, kernel_size=2, stride=2, bias=False)
        self.requested_layers = None

    def get_intermediate_layers(
        self, image, n=1, reshape=False, return_prefix_tokens=False,
        norm=False, attn_mask=None
    ):
        if not reshape or return_prefix_tokens or norm or attn_mask is not None:
            raise AssertionError("timm adapter requested an unexpected contract")
        self.requested_layers = list(n)
        base = self.conv(image)
        return [base + float(index) * 0.01 for index in n]


def config() -> RelationModelConfig:
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


def adapter_config() -> RelationModelConfig:
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
        predicate_adapter_rank=4,
    )


def boxes() -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.tensor(
        [
            [
                [0.20, 0.20, 0.20, 0.20],
                [0.50, 0.20, 0.20, 0.20],
                [0.20, 0.60, 0.20, 0.20],
                [0.70, 0.70, 0.20, 0.20],
            ],
            [
                [0.25, 0.25, 0.20, 0.20],
                [0.65, 0.25, 0.20, 0.20],
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
            ],
        ],
        dtype=torch.float32,
    )
    return value, torch.tensor([4, 2], dtype=torch.int64)


class RelationModelTest(unittest.TestCase):
    def test_official_adapter_extracts_multi_tap_feature_maps(self):
        adapter = OfficialDinoV3Backbone(FakeOfficialModel())
        taps = adapter.forward_taps(
            torch.rand(2, 3, 8, 8), (-3, -2, -1)
        )
        self.assertEqual(len(taps), 3)
        for tap in taps:
            self.assertEqual(tuple(tap.shape), (2, 8, 4, 4))

    def test_hf_adapter_extracts_patch_tail_from_multi_tap_hidden_states(self):
        adapter = HFDinoV3Backbone(
            FakeHFModel(),
            hidden_size=8,
            patch_size=2,
            image_mean=(0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0),
        )
        taps = adapter.forward_taps(
            torch.rand(2, 3, 8, 8), (-3, -2, -1)
        )
        self.assertEqual(len(taps), 3)
        for tap in taps:
            self.assertEqual(tuple(tap.shape), (2, 8, 4, 4))

    def test_meta_adapter_resolves_negative_taps_and_bchw_features(self):
        model = FakeMetaModel()
        adapter = MetaDinoV3Backbone(model)
        taps = adapter.forward_taps(
            torch.rand(1, 3, 8, 8), (-6, -3, -1)
        )
        self.assertEqual(model.requested_layers, [2, 5, 7])
        self.assertEqual(len(taps), 3)
        for tap in taps:
            self.assertEqual(tuple(tap.shape), (1, 8, 4, 4))

    def test_timm_adapter_resolves_negative_taps_and_bchw_features(self):
        model = FakeTimmModel()
        adapter = TimmDinoV3Backbone(model)
        taps = adapter.forward_taps(
            torch.rand(1, 3, 8, 8), (-6, -3, -1)
        )
        self.assertEqual(model.requested_layers, [2, 5, 7])
        self.assertEqual(len(taps), 3)
        for tap in taps:
            self.assertEqual(tuple(tap.shape), (1, 8, 4, 4))

    def test_adapter_rank_does_not_change_common_initialization(self):
        predicate_bank = torch.randn(3, 6)

        torch.manual_seed(20260929)
        control = KFRelationModel(
            ToyBackbone(),
            predicate_bank.clone(),
            config(),
        )
        torch.manual_seed(20260929)
        adapted = KFRelationModel(
            ToyBackbone(),
            predicate_bank.clone(),
            adapter_config(),
        )

        control_state = control.state_dict()
        adapted_state = adapted.state_dict()
        common_keys = sorted(
            key
            for key in control_state
            if not key.startswith("predicate_adapter_")
        )
        self.assertTrue(common_keys)
        self.assertEqual(
            common_keys,
            sorted(
                key
                for key in adapted_state
                if not key.startswith("predicate_adapter_")
            ),
        )
        for key in common_keys:
            self.assertTrue(
                torch.equal(control_state[key], adapted_state[key]),
                msg=f"common initialization drifted for {key}",
            )

    def test_predicate_adapter_starts_as_exact_residual_identity(self):
        torch.manual_seed(29)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            adapter_config(),
        )
        self.assertEqual(model.predicate_adapter_parameter_count(), 48)
        self.assertIsNotNone(model.predicate_adapter_down)
        self.assertIsNotNone(model.predicate_adapter_up)
        self.assertTrue(
            torch.allclose(
                model.effective_predicate_bank(),
                model.predicate_bank,
                atol=1.0e-7,
                rtol=1.0e-7,
            )
        )
        self.assertEqual(
            float(model.predicate_adapter_up.weight.abs().sum()),
            0.0,
        )

    def test_predicate_adapter_receives_finite_gradients_after_zero_init(self):
        torch.manual_seed(31)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            adapter_config(),
        )
        box_tensor, box_counts = boxes()
        pair_targets = torch.zeros(1, 4, 4)
        predicate_targets = torch.zeros(1, 4, 4, 3)

        first = model.forward_training(
            torch.rand(1, 3, 8, 8),
            box_tensor[:1],
            box_counts[:1],
        )
        _, _, sub_idx, obj_idx, valid_mask = first.runtime
        slot = int(torch.nonzero(valid_mask[0], as_tuple=False)[0, 0])
        subject = int(sub_idx[0, slot])
        object_ = int(obj_idx[0, slot])
        pair_targets[0, subject, object_] = 1.0
        predicate_targets[0, subject, object_, 0] = 1.0

        first_loss = supervised_relation_loss(
            first,
            pair_targets,
            predicate_targets,
        )["loss"]
        first_loss.backward()
        up_grad = model.predicate_adapter_up.weight.grad
        down_grad = model.predicate_adapter_down.weight.grad
        self.assertIsNotNone(up_grad)
        self.assertIsNotNone(down_grad)
        self.assertTrue(torch.isfinite(up_grad).all())
        self.assertTrue(torch.isfinite(down_grad).all())
        self.assertGreater(float(up_grad.abs().sum()), 0.0)
        # Up is zero-initialized, so the first backward cannot yet reach Down.
        self.assertEqual(float(down_grad.abs().sum()), 0.0)

        optimizer = torch.optim.SGD(
            [
                model.predicate_adapter_down.weight,
                model.predicate_adapter_up.weight,
            ],
            lr=0.1,
        )
        optimizer.step()
        model.zero_grad(set_to_none=True)

        second = model.forward_training(
            torch.rand(1, 3, 8, 8),
            box_tensor[:1],
            box_counts[:1],
        )
        second_loss = supervised_relation_loss(
            second,
            pair_targets,
            predicate_targets,
        )["loss"]
        second_loss.backward()
        down_grad = model.predicate_adapter_down.weight.grad
        up_grad = model.predicate_adapter_up.weight.grad
        self.assertIsNotNone(down_grad)
        self.assertIsNotNone(up_grad)
        self.assertTrue(torch.isfinite(down_grad).all())
        self.assertTrue(torch.isfinite(up_grad).all())
        self.assertGreater(float(down_grad.abs().sum()), 0.0)
        self.assertGreater(float(up_grad.abs().sum()), 0.0)

    def test_predicate_adapter_rank_validates_against_embedding_dimension(self):
        with self.assertRaises(ValueError):
            KFRelationModel(
                ToyBackbone(),
                torch.randn(3, 3),
                adapter_config(),
            )

    def test_runtime_shapes_and_valid_pair_indices(self):
        torch.manual_seed(3)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            config(),
        )
        box_tensor, box_counts = boxes()
        output = model(
            torch.rand(2, 3, 8, 8),
            box_tensor,
            box_counts,
        )
        pred, pair, sub, obj, valid = output
        self.assertEqual(tuple(pred.shape), (2, 6, 3))
        self.assertEqual(tuple(pair.shape), (2, 6))
        self.assertEqual(tuple(sub.shape), (2, 6))
        self.assertEqual(tuple(obj.shape), (2, 6))
        self.assertEqual(tuple(valid.shape), (2, 6))
        self.assertEqual(valid.dtype, torch.bool)

        for batch in range(2):
            count = int(box_counts[batch])
            for slot in range(6):
                if bool(valid[batch, slot]):
                    self.assertLess(int(sub[batch, slot]), count)
                    self.assertLess(int(obj[batch, slot]), count)
                    self.assertNotEqual(
                        int(sub[batch, slot]), int(obj[batch, slot])
                    )

    def test_supervised_loss_backpropagates_into_backbone_and_heads(self):
        torch.manual_seed(7)
        backbone = ToyBackbone()
        model = KFRelationModel(backbone, torch.randn(3, 6), config())
        box_tensor, box_counts = boxes()
        output = model.forward_training(
            torch.rand(2, 3, 8, 8),
            box_tensor,
            box_counts,
        )

        self.assertEqual(tuple(output.sampler_logits.shape), (2, 16))
        self.assertEqual(tuple(output.sampler_valid.shape), (2, 16))
        self.assertEqual(int(output.sampler_valid[0].sum()), 12)
        self.assertEqual(int(output.sampler_valid[1].sum()), 2)

        pair_targets = torch.zeros(2, 4, 4)
        predicate_targets = torch.zeros(
            (2, 4, 4, 3), dtype=torch.float32
        )
        for batch, count in enumerate((4, 2)):
            for subject in range(count):
                for object_ in range(count):
                    if subject == object_:
                        continue
                    pair_targets[batch, subject, object_] = 1.0
                    predicate_targets[
                        batch,
                        subject,
                        object_,
                        (subject + object_) % 3,
                    ] = 1.0

        losses = supervised_relation_loss(
            output, pair_targets, predicate_targets
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()

        self.assertIsNotNone(backbone.conv.weight.grad)
        self.assertIsNotNone(model.pair_head.weight.grad)
        self.assertIsNotNone(model.predicate_projection.weight.grad)
        sampler_grad = model.geometry_sampler[0].weight.grad
        self.assertIsNotNone(sampler_grad)
        self.assertTrue(torch.isfinite(sampler_grad).all())
        self.assertGreater(float(sampler_grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(losses["sampler_loss"]))

    def test_sampler_loss_rejects_inference_only_outputs(self):
        model = KFRelationModel(
            ToyBackbone(), torch.randn(3, 6), config()
        )
        box_tensor, box_counts = boxes()
        runtime = model(
            torch.rand(2, 3, 8, 8),
            box_tensor,
            box_counts,
        )
        pair_targets = torch.zeros(2, 4, 4)
        predicate_targets = torch.zeros(
            (2, 4, 4, 3), dtype=torch.float32
        )

        with self.assertRaises(ValueError):
            supervised_relation_loss(
                runtime, pair_targets, predicate_targets
            )

        losses = supervised_relation_loss(
            runtime,
            pair_targets,
            predicate_targets,
            RelationLossConfig(sampler_loss_weight=0.0),
        )
        self.assertEqual(float(losses["sampler_loss"]), 0.0)

    def test_predicate_loss_supports_multiple_labels_on_one_pair(self):
        torch.manual_seed(11)
        model = KFRelationModel(
            ToyBackbone(), torch.randn(3, 6), config()
        )
        box_tensor, box_counts = boxes()
        output = model.forward_training(
            torch.rand(1, 3, 8, 8),
            box_tensor[:1],
            box_counts[:1],
        )

        pred_logits, _, sub_idx, obj_idx, valid_mask = output.runtime
        valid_slots = torch.nonzero(valid_mask[0], as_tuple=False)
        self.assertGreater(valid_slots.numel(), 0)
        slot = int(valid_slots[0, 0])
        subject = int(sub_idx[0, slot])
        object_ = int(obj_idx[0, slot])

        pair_targets = torch.zeros(1, 4, 4)
        predicate_targets = torch.zeros(1, 4, 4, 3)
        pair_targets[0, subject, object_] = 1.0
        predicate_targets[0, subject, object_, 0] = 1.0
        predicate_targets[0, subject, object_, 2] = 1.0

        losses = supervised_relation_loss(
            output, pair_targets, predicate_targets
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logits[0, slot],
            torch.tensor([1.0, 0.0, 1.0]),
        )
        self.assertTrue(
            torch.allclose(losses["predicate_loss"], expected)
        )

    def test_predicate_positive_weights_change_multilabel_bce(self):
        torch.manual_seed(13)
        model = KFRelationModel(
            ToyBackbone(), torch.randn(3, 6), config()
        )
        box_tensor, box_counts = boxes()
        output = model.forward_training(
            torch.rand(1, 3, 8, 8),
            box_tensor[:1],
            box_counts[:1],
        )

        pred_logits, _, sub_idx, obj_idx, valid_mask = output.runtime
        valid_slots = torch.nonzero(valid_mask[0], as_tuple=False)
        slot = int(valid_slots[0, 0])
        subject = int(sub_idx[0, slot])
        object_ = int(obj_idx[0, slot])

        pair_targets = torch.zeros(1, 4, 4)
        predicate_targets = torch.zeros(1, 4, 4, 3)
        pair_targets[0, subject, object_] = 1.0
        predicate_targets[0, subject, object_, 0] = 1.0
        predicate_targets[0, subject, object_, 2] = 1.0
        weights = torch.tensor([2.0, 1.0, 4.0])

        losses = supervised_relation_loss(
            output,
            pair_targets,
            predicate_targets,
            predicate_positive_weights=weights,
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logits[0, slot],
            torch.tensor([1.0, 0.0, 1.0]),
            pos_weight=weights,
        )
        self.assertTrue(
            torch.allclose(losses["predicate_loss"], expected)
        )

        with self.assertRaises(ValueError):
            supervised_relation_loss(
                output,
                pair_targets,
                predicate_targets,
                predicate_positive_weights=torch.ones(2),
            )
        with self.assertRaises(ValueError):
            supervised_relation_loss(
                output,
                pair_targets,
                predicate_targets,
                predicate_positive_weights=torch.tensor(
                    [1.0, 0.0, 1.0]
                ),
            )

    def test_predicate_supervision_mask_zeroes_masked_gradients(self):
        torch.manual_seed(19)
        model = KFRelationModel(
            ToyBackbone(), torch.randn(3, 6), config()
        )
        box_tensor, box_counts = boxes()
        output = model.forward_training(
            torch.rand(1, 3, 8, 8),
            box_tensor[:1],
            box_counts[:1],
        )

        pred_logits, _, sub_idx, obj_idx, valid_mask = output.runtime
        pred_logits.retain_grad()
        valid_slots = torch.nonzero(valid_mask[0], as_tuple=False)
        slot = int(valid_slots[0, 0])
        subject = int(sub_idx[0, slot])
        object_ = int(obj_idx[0, slot])

        pair_targets = torch.zeros(1, 4, 4)
        predicate_targets = torch.zeros(1, 4, 4, 3)
        pair_targets[0, subject, object_] = 1.0
        predicate_targets[0, subject, object_, 0] = 1.0
        supervision = torch.tensor([True, True, False])

        losses = supervised_relation_loss(
            output,
            pair_targets,
            predicate_targets,
            RelationLossConfig(
                sampler_loss_weight=0.0,
                pair_loss_weight=0.0,
                predicate_loss_weight=1.0,
            ),
            predicate_supervision_mask=supervision,
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logits[0, slot, :2],
            torch.tensor([1.0, 0.0]),
        )
        self.assertTrue(
            torch.allclose(losses["predicate_loss"], expected)
        )

        losses["loss"].backward()
        self.assertIsNotNone(pred_logits.grad)
        self.assertEqual(
            float(pred_logits.grad[..., 2].abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(pred_logits.grad[..., :2].abs().sum()),
            0.0,
        )

    def test_predicate_supervision_mask_fails_fast_on_invalid_contract(self):
        torch.manual_seed(23)
        model = KFRelationModel(
            ToyBackbone(), torch.randn(3, 6), config()
        )
        box_tensor, box_counts = boxes()
        output = model.forward_training(
            torch.rand(1, 3, 8, 8),
            box_tensor[:1],
            box_counts[:1],
        )
        _, _, sub_idx, obj_idx, valid_mask = output.runtime
        valid_slots = torch.nonzero(valid_mask[0], as_tuple=False)
        slot = int(valid_slots[0, 0])
        subject = int(sub_idx[0, slot])
        object_ = int(obj_idx[0, slot])

        pair_targets = torch.zeros(1, 4, 4)
        predicate_targets = torch.zeros(1, 4, 4, 3)
        pair_targets[0, subject, object_] = 1.0
        predicate_targets[0, subject, object_, 2] = 1.0

        invalid_masks = (
            torch.tensor([True, False]),
            torch.tensor([1.0, 1.0, 0.0]),
            torch.tensor([False, False, False]),
        )
        for supervision in invalid_masks:
            with self.assertRaises(ValueError):
                supervised_relation_loss(
                    output,
                    pair_targets,
                    predicate_targets,
                    predicate_supervision_mask=supervision,
                )

        with self.assertRaises(ValueError):
            supervised_relation_loss(
                output,
                pair_targets,
                predicate_targets,
                predicate_supervision_mask=torch.tensor(
                    [True, True, False]
                ),
            )

    def test_predicate_targets_fail_fast_on_inconsistent_supervision(self):
        model = KFRelationModel(
            ToyBackbone(), torch.randn(3, 6), config()
        )
        box_tensor, box_counts = boxes()
        output = model.forward_training(
            torch.rand(1, 3, 8, 8),
            box_tensor[:1],
            box_counts[:1],
        )

        pair_targets = torch.zeros(1, 4, 4)
        predicate_targets = torch.zeros(1, 4, 4, 3)
        pair_targets[0, 0, 1] = 1.0
        with self.assertRaises(ValueError):
            supervised_relation_loss(
                output, pair_targets, predicate_targets
            )

        pair_targets.zero_()
        predicate_targets[0, 0, 1, 0] = 1.0
        with self.assertRaises(ValueError):
            supervised_relation_loss(
                output, pair_targets, predicate_targets
            )

        with self.assertRaises(ValueError):
            supervised_relation_loss(
                output,
                torch.zeros(1, 4, 4),
                torch.zeros(1, 4, 4, 2),
            )

    def test_synthetic_onnx_matches_native_runtime_contract(self):
        torch.manual_seed(8)
        model = KFRelationModel(
            ToyBackbone(), torch.randn(3, 6), config()
        )
        box_tensor, box_counts = boxes()
        image = torch.rand(1, 3, 8, 8)
        export_boxes = box_tensor[:1].contiguous()
        export_counts = box_counts[:1].contiguous()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relation.onnx"
            reference = export_graph(
                model,
                path,
                image,
                export_boxes,
                export_counts,
                opset=18,
            )
            delta = check_onnx_parity(
                path,
                image,
                export_boxes,
                export_counts,
                reference,
            )
            self.assertLessEqual(delta, 1.0e-3)

            import onnxruntime as ort

            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self.assertEqual(
                [item.name for item in session.get_inputs()],
                INPUT_NAMES,
            )
            self.assertEqual(
                [item.name for item in session.get_outputs()],
                OUTPUT_NAMES,
            )

    def test_adapter_enabled_onnx_matches_native_runtime_contract(self):
        torch.manual_seed(37)
        model = KFRelationModel(
            ToyBackbone(), torch.randn(3, 6), adapter_config()
        )
        with torch.no_grad():
            model.predicate_adapter_up.weight.normal_(0.0, 0.01)

        box_tensor, box_counts = boxes()
        image = torch.rand(1, 3, 8, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relation-adapter.onnx"
            reference = export_graph(
                model,
                path,
                image,
                box_tensor[:1].contiguous(),
                box_counts[:1].contiguous(),
                opset=18,
            )
            delta = check_onnx_parity(
                path,
                image,
                box_tensor[:1].contiguous(),
                box_counts[:1].contiguous(),
                reference,
            )
        self.assertLessEqual(delta, 1.0e-3)

    def test_adapter_checkpoint_round_trip_preserves_rank_and_state(self):
        torch.manual_seed(41)
        model = KFRelationModel(
            ToyBackbone(), torch.randn(3, 6), adapter_config()
        )
        with torch.no_grad():
            model.predicate_adapter_up.weight.normal_(0.0, 0.01)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_checkpoint(
                path,
                model,
                backbone_model="synthetic/test-backbone",
                predicates=["beside", "holding", "riding"],
            )
            payload = load_payload(path)
            restored_config = config_from_payload(payload)

        self.assertEqual(restored_config.predicate_adapter_rank, 4)
        self.assertIn("predicate_adapter_down.weight", payload["state_dict"])
        self.assertIn("predicate_adapter_up.weight", payload["state_dict"])
        self.assertTrue(
            torch.equal(
                payload["state_dict"]["predicate_adapter_up.weight"],
                model.predicate_adapter_up.weight.detach().cpu(),
            )
        )

    def test_checkpoint_round_trip_preserves_runtime_configuration(self):
        torch.manual_seed(9)
        model = KFRelationModel(
            ToyBackbone(), torch.randn(3, 6), config()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                path,
                model,
                backbone_model="synthetic/test-backbone",
                predicates=["beside", "holding", "riding"],
                extra={"purpose": "unit-test"},
            )
            payload = load_payload(path)
            restored = config_from_payload(payload)
            self.assertEqual(restored, config())
            self.assertEqual(
                payload["predicates"],
                ["beside", "holding", "riding"],
            )
            self.assertEqual(
                tuple(payload["predicate_embeddings"].shape), (3, 6)
            )


if __name__ == "__main__":
    unittest.main()
