from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from checkpoint import config_from_payload, load_payload, save_checkpoint
from export_onnx import (
    ENCODER_OUTPUT_NAMES,
    INPUT_NAMES,
    OUTPUT_NAMES,
    check_onnx_parity,
    export_encoder_graph,
    export_graph,
)
from losses import RelationLossConfig, supervised_relation_loss
from model import (
    BackboneAdapter,
    HFDinoV3Backbone,
    KFRelationModel,
    MetaDinoV3Backbone,
    OfficialDinoV3Backbone,
    RelationModelConfig,
    RelationTrainingOutputs,
    TimmDinoV3Backbone,
    _pair_union_contact_boxes,
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


def visual_config(mode: str) -> RelationModelConfig:
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
        pair_visual_evidence=mode,
    )


def geometry_config(mode: str) -> RelationModelConfig:
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
        pair_visual_evidence="contact",
        pair_geometry_evidence=mode,
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
    def test_rich_pair_geometry_has_bounded_normalized_features(self):
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            geometry_config("rich"),
        )
        box_tensor = torch.tensor(
            [
                [
                    [0.25, 0.25, 0.20, 0.20],
                    [0.35, 0.25, 0.20, 0.20],
                ]
            ],
            dtype=torch.float32,
        )
        rich = model._rich_pair_geometry(box_tensor)
        expected = torch.tensor(
            [
                0.5, 0.0,
                0.5, 0.0,
                0.5, 0.5,
                0.0, 0.0,
                1.0, 0.0,
            ],
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.allclose(
                rich[0, 0, 1],
                expected,
                atol=1.0e-5,
            )
        )

        separated = torch.tensor(
            [
                [
                    [0.20, 0.20, 0.20, 0.20],
                    [0.80, 0.20, 0.20, 0.20],
                ]
            ],
            dtype=torch.float32,
        )
        rich_separated = model._rich_pair_geometry(separated)
        self.assertAlmostEqual(
            float(rich_separated[0, 0, 1, 6]),
            0.4,
            places=5,
        )
        self.assertEqual(
            float(rich_separated[0, 0, 1, 4]),
            0.0,
        )
        self.assertAlmostEqual(
            float(rich_separated[0, 0, 1, 8]),
            1.0,
            places=5,
        )

    def test_rich_geometry_keeps_common_initialization_and_output_identical(self):
        def build(mode: str) -> KFRelationModel:
            torch.manual_seed(66)
            backbone = ToyBackbone()
            embeddings = torch.randn(3, 6)
            return KFRelationModel(
                backbone,
                embeddings,
                geometry_config(mode),
            )

        basic = build("basic")
        rich = build("rich")

        basic_state = basic.state_dict()
        rich_state = rich.state_dict()
        for name, value in basic_state.items():
            self.assertIn(name, rich_state)
            self.assertTrue(
                torch.equal(value, rich_state[name]),
                msg=f"common parameter drift: {name}",
            )

        self.assertIsNone(basic.rich_geometry_projection)
        self.assertIsNone(basic.rich_geometry_sampler)
        self.assertIsNotNone(rich.rich_geometry_projection)
        self.assertIsNotNone(rich.rich_geometry_sampler)
        self.assertEqual(
            float(rich.rich_geometry_projection.weight.abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(rich.rich_geometry_sampler.weight.abs().sum()),
            0.0,
        )

        box_tensor, box_counts = boxes()
        torch.manual_seed(67)
        image = torch.rand(1, 3, 8, 8)
        with torch.inference_mode():
            basic_output = basic(
                image, box_tensor[:1], box_counts[:1]
            )
            rich_output = rich(
                image, box_tensor[:1], box_counts[:1]
            )
        for left, right in zip(basic_output, rich_output):
            self.assertTrue(torch.equal(left, right))

    def test_union_and_contact_box_geometry(self):
        subject = torch.tensor(
            [
                [
                    [0.25, 0.25, 0.20, 0.20],
                    [0.20, 0.20, 0.20, 0.20],
                    [0.25, 0.25, 0.20, 0.20],
                ]
            ],
            dtype=torch.float32,
        )
        object_ = torch.tensor(
            [
                [
                    [0.35, 0.25, 0.20, 0.20],
                    [0.80, 0.80, 0.20, 0.20],
                    [0.45, 0.25, 0.20, 0.20],
                ]
            ],
            dtype=torch.float32,
        )
        valid = torch.tensor([[True, True, True]])

        union, contact, contact_valid = _pair_union_contact_boxes(
            subject, object_, valid
        )

        self.assertTrue(
            torch.allclose(
                union[0, 0],
                torch.tensor([0.30, 0.25, 0.30, 0.20]),
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                contact[0, 0],
                torch.tensor([0.30, 0.25, 0.10, 0.20]),
                atol=1.0e-6,
            )
        )
        self.assertTrue(bool(contact_valid[0, 0]))

        self.assertFalse(bool(contact_valid[0, 1]))
        self.assertTrue(torch.equal(
            contact[0, 1], torch.zeros(4)
        ))

        # Pair 2 touches exactly at one vertical edge. Zero-area contact is
        # deliberately represented as invalid/zero rather than a thin ROI.
        self.assertFalse(bool(contact_valid[0, 2]))
        self.assertTrue(torch.equal(
            contact[0, 2], torch.zeros(4)
        ))

    def test_visual_evidence_modes_keep_common_initialization_identical(self):
        def build(mode: str) -> KFRelationModel:
            torch.manual_seed(61)
            backbone = ToyBackbone()
            embeddings = torch.randn(3, 6)
            return KFRelationModel(
                backbone,
                embeddings,
                visual_config(mode),
            )

        endpoint = build("endpoint")
        union = build("union")
        contact = build("contact")
        union_contact = build("union-contact")

        endpoint_state = endpoint.state_dict()
        for candidate in (union, contact, union_contact):
            state = candidate.state_dict()
            for name, value in endpoint_state.items():
                self.assertIn(name, state)
                self.assertTrue(
                    torch.equal(value, state[name]),
                    msg=f"common parameter drift: {name}",
                )

        self.assertIsNone(endpoint.union_projection)
        self.assertIsNone(endpoint.contact_projection)
        self.assertIsNotNone(union.union_projection)
        self.assertIsNone(union.contact_projection)
        self.assertIsNone(contact.union_projection)
        self.assertIsNotNone(contact.contact_projection)
        self.assertIsNotNone(union_contact.union_projection)
        self.assertIsNotNone(union_contact.contact_projection)
        self.assertEqual(
            float(union.union_projection.weight.abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(contact.contact_projection.weight.abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(union_contact.union_projection.weight.abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(union_contact.contact_projection.weight.abs().sum()),
            0.0,
        )

        box_tensor, box_counts = boxes()
        torch.manual_seed(62)
        image = torch.rand(1, 3, 8, 8)
        with torch.inference_mode():
            expected = endpoint(
                image, box_tensor[:1], box_counts[:1]
            )
            union_output = union(
                image, box_tensor[:1], box_counts[:1]
            )
            contact_output = contact(
                image, box_tensor[:1], box_counts[:1]
            )
            union_contact_output = union_contact(
                image, box_tensor[:1], box_counts[:1]
            )
        for left, union_value, contact_value, final in zip(
            expected,
            union_output,
            contact_output,
            union_contact_output,
        ):
            self.assertTrue(torch.equal(left, union_value))
            self.assertTrue(torch.equal(left, contact_value))
            self.assertTrue(torch.equal(left, final))

    def test_open_vocabulary_encoder_is_independent_of_predicate_count(self):
        torch.manual_seed(31)
        model = KFRelationModel(
            ToyBackbone(), torch.randn(3, 6), config()
        )
        image = torch.rand(1, 3, 8, 8)
        box_tensor, box_counts = boxes()

        with torch.inference_mode():
            first = model.forward_encoder(
                image, box_tensor[:1], box_counts[:1]
            )

        self.assertEqual(len(first), 6)
        self.assertEqual(tuple(first[0].shape), (1, 6, 6))
        self.assertEqual(tuple(first[1].shape), (1, 6, 6))
        self.assertTrue(torch.equal(first[0], first[1]))

        model.predicate_bank = torch.nn.functional.normalize(
            torch.randn(7, 6), dim=-1
        )
        with torch.inference_mode():
            second = model.forward_encoder(
                image, box_tensor[:1], box_counts[:1]
            )

        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))

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

    def test_released_apache_graph_uses_512_width_and_40_boxes(self):
        cfg = RelationModelConfig(
            image_size=8,
            max_boxes=40,
            pair_budget=6,
            hidden_dim=512,
            geometry_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            tap_indices=(-3, -2, -1),
            pair_evidence_contract="apache",
            pair_sampler_contract="apache",
            relation_context_contract="apache",
            predicate_head_contract="apache",
        )
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            cfg,
        )
        image = torch.randn(1, 3, 8, 8)
        box_tensor = torch.zeros(
            (1, 40, 4),
            dtype=torch.float32,
        )
        box_tensor[0, :4] = torch.tensor(
            [
                [0.20, 0.20, 0.20, 0.20],
                [0.50, 0.20, 0.20, 0.20],
                [0.20, 0.60, 0.20, 0.20],
                [0.70, 0.70, 0.20, 0.20],
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            output = model(
                image,
                box_tensor,
                torch.tensor([4], dtype=torch.int64),
            )

        self.assertEqual(model.config.hidden_dim, 512)
        self.assertEqual(model.config.max_boxes, 40)
        self.assertEqual(model.apache_pair_projection.out_features, 512)
        self.assertEqual(
            model.apache_relation_transformer.scene_proj.out_features,
            512,
        )
        self.assertEqual(
            tuple(model.apache_deformable_read.norm.normalized_shape),
            (512,),
        )
        self.assertEqual(
            model.apache_relation_interaction.scene_proj.out_features,
            512,
        )
        self.assertEqual(
            model.apache_vocab_head.d_model,
            512,
        )
        self.assertEqual(
            tuple(output[0].shape),
            (1, 6, 3),
        )

    def test_reference_40_box_tensor_contract(self):
        cfg = RelationModelConfig(
            image_size=8,
            max_boxes=40,
            pair_budget=6,
            hidden_dim=16,
            geometry_dim=8,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            tap_indices=(-3, -2, -1),
        )
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            cfg,
        )
        image = torch.randn(1, 3, 8, 8)
        box_tensor = torch.zeros(
            (1, 40, 4),
            dtype=torch.float32,
        )
        box_tensor[0, :4] = torch.tensor(
            [
                [0.20, 0.20, 0.20, 0.20],
                [0.50, 0.20, 0.20, 0.20],
                [0.20, 0.60, 0.20, 0.20],
                [0.70, 0.70, 0.20, 0.20],
            ],
            dtype=torch.float32,
        )
        output = model(
            image,
            box_tensor,
            torch.tensor([4], dtype=torch.int64),
        )
        self.assertEqual(
            tuple(output[0].shape),
            (1, 6, 3),
        )

        with self.assertRaisesRegex(
            ValueError,
            "max_boxes",
        ):
            model(
                image,
                torch.zeros(
                    (1, 41, 4),
                    dtype=torch.float32,
                ),
                torch.tensor([4], dtype=torch.int64),
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

    def test_predicate_negative_weights_scale_only_negative_terms(self):
        torch.manual_seed(17)
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

        negative_weights = torch.tensor([0.0, 0.25, 0.0])
        losses = supervised_relation_loss(
            output,
            pair_targets,
            predicate_targets,
            RelationLossConfig(
                sampler_loss_weight=0.0,
                pair_loss_weight=0.0,
                predicate_loss_weight=1.0,
            ),
            predicate_negative_weights=negative_weights,
        )

        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logits[0, slot],
            torch.tensor([1.0, 0.0, 0.0]),
            reduction="none",
        )
        expected = (raw[0] + 0.25 * raw[1]) / 1.25
        self.assertTrue(
            torch.allclose(losses["predicate_loss"], expected)
        )

        losses["loss"].backward()
        self.assertIsNotNone(pred_logits.grad)
        self.assertGreater(
            float(pred_logits.grad[0, slot, 0].abs()),
            0.0,
        )
        self.assertGreater(
            float(pred_logits.grad[0, slot, 1].abs()),
            0.0,
        )
        self.assertEqual(
            float(pred_logits.grad[0, slot, 2].abs()),
            0.0,
        )

    def test_zero_negative_weight_matches_zero_support_mask(self):
        torch.manual_seed(18)
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
        slot = int(torch.nonzero(valid_mask[0], as_tuple=False)[0, 0])
        subject = int(sub_idx[0, slot])
        object_ = int(obj_idx[0, slot])

        pair_targets = torch.zeros(1, 4, 4)
        predicate_targets = torch.zeros(1, 4, 4, 3)
        pair_targets[0, subject, object_] = 1.0
        predicate_targets[0, subject, object_, 0] = 1.0
        config_value = RelationLossConfig(
            sampler_loss_weight=0.0,
            pair_loss_weight=0.0,
            predicate_loss_weight=1.0,
        )

        weighted = supervised_relation_loss(
            output,
            pair_targets,
            predicate_targets,
            config_value,
            predicate_negative_weights=torch.tensor(
                [1.0, 1.0, 0.0]
            ),
        )
        masked = supervised_relation_loss(
            output,
            pair_targets,
            predicate_targets,
            config_value,
            predicate_supervision_mask=torch.tensor(
                [True, True, False]
            ),
        )
        self.assertTrue(
            torch.allclose(
                weighted["predicate_loss"],
                masked["predicate_loss"],
            )
        )

    def test_predicate_negative_weights_fail_fast_on_invalid_contract(self):
        torch.manual_seed(20)
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
        slot = int(torch.nonzero(valid_mask[0], as_tuple=False)[0, 0])
        subject = int(sub_idx[0, slot])
        object_ = int(obj_idx[0, slot])

        pair_targets = torch.zeros(1, 4, 4)
        predicate_targets = torch.zeros(1, 4, 4, 3)
        pair_targets[0, subject, object_] = 1.0
        predicate_targets[0, subject, object_, 0] = 1.0

        invalid = (
            torch.ones(2),
            torch.tensor([1.0, -0.1, 1.0]),
            torch.tensor([1.0, float("nan"), 1.0]),
        )
        for weights in invalid:
            with self.assertRaises(ValueError):
                supervised_relation_loss(
                    output,
                    pair_targets,
                    predicate_targets,
                    predicate_negative_weights=weights,
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

        supervision = torch.tensor([True, True, False])
        invalid_holdouts = (
            torch.tensor([False, True, False]),
            torch.tensor([False, False]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([False, False, False]),
        )
        for holdout in invalid_holdouts:
            with self.assertRaises(ValueError):
                supervised_relation_loss(
                    output,
                    pair_targets,
                    predicate_targets,
                    predicate_supervision_mask=supervision,
                    explicit_holdout_mask=holdout,
                )

        with self.assertRaises(ValueError):
            supervised_relation_loss(
                output,
                pair_targets,
                predicate_targets,
                explicit_holdout_mask=torch.tensor(
                    [False, False, True]
                ),
            )

    def test_explicit_holdout_can_hide_positive_predicate_targets(self):
        torch.manual_seed(24)
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
        slot = int(torch.nonzero(valid_mask[0], as_tuple=False)[0, 0])
        subject = int(sub_idx[0, slot])
        object_ = int(obj_idx[0, slot])

        pair_targets = torch.zeros(1, 4, 4)
        predicate_targets = torch.zeros(1, 4, 4, 3)
        pair_targets[0, subject, object_] = 1.0
        predicate_targets[0, subject, object_, 2] = 1.0
        supervision = torch.tensor([True, True, False])

        # The ordinary supervision-mask path still rejects hidden positives.
        with self.assertRaises(ValueError):
            supervised_relation_loss(
                output,
                pair_targets,
                predicate_targets,
                predicate_supervision_mask=supervision,
            )

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
            explicit_holdout_mask=~supervision,
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logits[0, slot, :2],
            torch.zeros(2),
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

    def test_holdout_only_row_policy_skips_only_unknown_predicate_rows(self):
        torch.manual_seed(25)
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
        slot = int(torch.nonzero(valid_mask[0], as_tuple=False)[0, 0])
        subject = int(sub_idx[0, slot])
        object_ = int(obj_idx[0, slot])

        pair_targets = torch.zeros(1, 4, 4)
        predicate_targets = torch.zeros(1, 4, 4, 3)
        pair_targets[0, subject, object_] = 1.0
        predicate_targets[0, subject, object_, 2] = 1.0
        supervision = torch.tensor([True, True, False])
        holdout = torch.tensor([False, False, True])
        loss_config = RelationLossConfig(
            sampler_loss_weight=0.0,
            pair_loss_weight=0.0,
            predicate_loss_weight=1.0,
        )

        skipped = supervised_relation_loss(
            output,
            pair_targets,
            predicate_targets,
            loss_config,
            predicate_supervision_mask=supervision,
            explicit_holdout_mask=holdout,
            explicit_holdout_row_policy="skip-holdout-only",
        )
        self.assertEqual(float(skipped["predicate_rows"]), 1.0)
        self.assertEqual(
            float(skipped["predicate_rows_skipped"]),
            1.0,
        )
        self.assertEqual(float(skipped["predicate_loss"]), 0.0)

        predicate_targets[0, subject, object_, 0] = 1.0
        mixed = supervised_relation_loss(
            output,
            pair_targets,
            predicate_targets,
            loss_config,
            predicate_supervision_mask=supervision,
            explicit_holdout_mask=holdout,
            explicit_holdout_row_policy="skip-holdout-only",
        )
        self.assertEqual(float(mixed["predicate_rows"]), 1.0)
        self.assertEqual(
            float(mixed["predicate_rows_skipped"]),
            0.0,
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logits[0, slot, :2],
            torch.tensor([1.0, 0.0]),
        )
        self.assertTrue(
            torch.allclose(mixed["predicate_loss"], expected)
        )

        with self.assertRaises(ValueError):
            supervised_relation_loss(
                output,
                pair_targets,
                predicate_targets,
                loss_config,
                predicate_supervision_mask=supervision,
                explicit_holdout_row_policy="skip-holdout-only",
            )
        with self.assertRaises(ValueError):
            supervised_relation_loss(
                output,
                pair_targets,
                predicate_targets,
                loss_config,
                explicit_holdout_row_policy="unsupported",
            )

    def test_batch_local_infonce_uses_only_observed_predicate_columns(self):
        pred_logits = torch.zeros(1, 2, 3, requires_grad=True)
        pair_logits = torch.zeros(1, 2, requires_grad=True)
        sub_idx = torch.tensor([[0, 1]], dtype=torch.int64)
        obj_idx = torch.tensor([[1, 0]], dtype=torch.int64)
        valid = torch.tensor([[True, True]])
        query = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            requires_grad=True,
        )
        raw_query = query.clone()
        bank = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            requires_grad=True,
        )
        outputs = RelationTrainingOutputs(
            runtime=(
                pred_logits,
                pair_logits,
                sub_idx,
                obj_idx,
                valid,
            ),
            sampler_logits=torch.zeros(1, 4),
            sampler_valid=torch.ones(1, 4, dtype=torch.bool),
            predicate_query=query,
            predicate_query_raw=raw_query,
            predicate_bank=bank,
        )
        pair_targets = torch.zeros(1, 2, 2)
        pair_targets[0, 0, 1] = 1.0
        pair_targets[0, 1, 0] = 1.0
        predicate_targets = torch.zeros(1, 2, 2, 3)
        predicate_targets[0, 0, 1, 0] = 1.0
        predicate_targets[0, 1, 0, 1] = 1.0

        losses = supervised_relation_loss(
            outputs,
            pair_targets,
            predicate_targets,
            RelationLossConfig(
                sampler_loss_weight=0.0,
                pair_loss_weight=0.0,
                predicate_loss_weight=1.0,
                predicate_objective="batch-local-infonce",
                predicate_contrastive_temperature=1.0,
            ),
        )
        expected = torch.logsumexp(
            torch.tensor([1.0, 0.0]), dim=0
        ) - 1.0
        self.assertTrue(
            torch.allclose(losses["predicate_loss"], expected)
        )
        self.assertEqual(
            float(losses["predicate_contrast_set_size"]),
            2.0,
        )
        self.assertAlmostEqual(
            float(losses["predicate_unobserved_column_fraction"]),
            1.0 / 3.0,
            places=6,
        )

        losses["loss"].backward()
        self.assertIsNotNone(query.grad)
        self.assertGreater(float(query.grad.abs().sum()), 0.0)

    def test_batch_local_infonce_adds_bounded_hard_negatives(self):
        pred_logits = torch.zeros(1, 1, 4)
        pair_logits = torch.zeros(1, 1)
        outputs = RelationTrainingOutputs(
            runtime=(
                pred_logits,
                pair_logits,
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([[1]], dtype=torch.int64),
                torch.tensor([[True]]),
            ),
            sampler_logits=torch.zeros(1, 4),
            sampler_valid=torch.ones(1, 4, dtype=torch.bool),
            predicate_query=torch.tensor(
                [[[1.0, 0.0]]], requires_grad=True
            ),
            predicate_query_raw=torch.tensor(
                [[[1.0, 0.0]]], requires_grad=True
            ),
            predicate_bank=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.8, 0.6],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ]
            ),
        )
        pair_targets = torch.zeros(1, 2, 2)
        pair_targets[0, 0, 1] = 1.0
        predicate_targets = torch.zeros(1, 2, 2, 4)
        predicate_targets[0, 0, 1, 0] = 1.0

        losses = supervised_relation_loss(
            outputs,
            pair_targets,
            predicate_targets,
            RelationLossConfig(
                sampler_loss_weight=0.0,
                pair_loss_weight=0.0,
                predicate_loss_weight=1.0,
                predicate_objective="batch-local-infonce",
                predicate_contrastive_temperature=1.0,
                predicate_contrastive_hard_negative_count=1,
            ),
            predicate_contrastive_negative_mask=torch.tensor(
                [False, True, True, False]
            ),
        )
        expected = torch.logsumexp(
            torch.tensor([1.0, 0.8]), dim=0
        ) - 1.0
        self.assertTrue(
            torch.allclose(losses["predicate_loss"], expected)
        )
        self.assertEqual(
            float(losses["predicate_contrast_set_size"]),
            2.0,
        )
        self.assertEqual(
            float(losses["predicate_hard_negative_count"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(losses["predicate_unobserved_column_fraction"]),
            0.5,
            places=6,
        )

    def test_batch_local_infonce_hard_negatives_respect_supervision_mask(self):
        pred_logits = torch.zeros(1, 1, 3)
        pair_logits = torch.zeros(1, 1)
        outputs = RelationTrainingOutputs(
            runtime=(
                pred_logits,
                pair_logits,
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([[1]], dtype=torch.int64),
                torch.tensor([[True]]),
            ),
            sampler_logits=torch.zeros(1, 4),
            sampler_valid=torch.ones(1, 4, dtype=torch.bool),
            predicate_query=torch.tensor([[[1.0, 0.0]]]),
            predicate_query_raw=torch.tensor([[[1.0, 0.0]]]),
            predicate_bank=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ]
            ),
        )
        pair_targets = torch.zeros(1, 2, 2)
        pair_targets[0, 0, 1] = 1.0
        predicate_targets = torch.zeros(1, 2, 2, 3)
        predicate_targets[0, 0, 1, 0] = 1.0
        supervision = torch.tensor([True, False, True])

        losses = supervised_relation_loss(
            outputs,
            pair_targets,
            predicate_targets,
            RelationLossConfig(
                sampler_loss_weight=0.0,
                pair_loss_weight=0.0,
                predicate_loss_weight=1.0,
                predicate_objective="batch-local-infonce",
                predicate_contrastive_temperature=1.0,
                predicate_contrastive_hard_negative_count=1,
            ),
            predicate_supervision_mask=supervision,
            explicit_holdout_mask=~supervision,
            predicate_contrastive_negative_mask=torch.tensor(
                [False, True, False]
            ),
        )
        self.assertEqual(
            float(losses["predicate_hard_negative_count"]),
            0.0,
        )
        self.assertEqual(
            float(losses["predicate_contrast_set_size"]),
            1.0,
        )
        self.assertEqual(float(losses["predicate_loss"]), 0.0)

    def test_batch_local_infonce_preserves_multilabel_positives(self):
        pred_logits = torch.zeros(1, 1, 3)
        pair_logits = torch.zeros(1, 1)
        sub_idx = torch.tensor([[0]], dtype=torch.int64)
        obj_idx = torch.tensor([[1]], dtype=torch.int64)
        valid = torch.tensor([[True]])
        query = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
        bank = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )
        outputs = RelationTrainingOutputs(
            runtime=(
                pred_logits,
                pair_logits,
                sub_idx,
                obj_idx,
                valid,
            ),
            sampler_logits=torch.zeros(1, 4),
            sampler_valid=torch.ones(1, 4, dtype=torch.bool),
            predicate_query=query,
            predicate_query_raw=query.clone(),
            predicate_bank=bank,
        )
        pair_targets = torch.zeros(1, 2, 2)
        pair_targets[0, 0, 1] = 1.0
        predicate_targets = torch.zeros(1, 2, 2, 3)
        predicate_targets[0, 0, 1, 0] = 1.0
        predicate_targets[0, 0, 1, 1] = 1.0

        losses = supervised_relation_loss(
            outputs,
            pair_targets,
            predicate_targets,
            RelationLossConfig(
                sampler_loss_weight=0.0,
                pair_loss_weight=0.0,
                predicate_loss_weight=1.0,
                predicate_objective="batch-local-infonce",
                predicate_contrastive_temperature=1.0,
            ),
        )
        denominator = torch.logsumexp(
            torch.tensor([1.0, 0.0]), dim=0
        )
        expected = (
            (denominator - 1.0)
            + (denominator - 0.0)
        ) * 0.5
        self.assertTrue(
            torch.allclose(losses["predicate_loss"], expected)
        )
        self.assertEqual(
            float(losses["predicate_contrast_set_size"]),
            2.0,
        )

    def test_batch_local_infonce_skips_explicit_holdout_only_rows(self):
        pred_logits = torch.zeros(1, 2, 3)
        pair_logits = torch.zeros(1, 2)
        sub_idx = torch.tensor([[0, 1]], dtype=torch.int64)
        obj_idx = torch.tensor([[1, 0]], dtype=torch.int64)
        valid = torch.tensor([[True, True]])
        query = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            requires_grad=True,
        )
        bank = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )
        outputs = RelationTrainingOutputs(
            runtime=(
                pred_logits,
                pair_logits,
                sub_idx,
                obj_idx,
                valid,
            ),
            sampler_logits=torch.zeros(1, 4),
            sampler_valid=torch.ones(1, 4, dtype=torch.bool),
            predicate_query=query,
            predicate_query_raw=query.clone(),
            predicate_bank=bank,
        )
        pair_targets = torch.zeros(1, 2, 2)
        pair_targets[0, 0, 1] = 1.0
        pair_targets[0, 1, 0] = 1.0
        predicate_targets = torch.zeros(1, 2, 2, 3)
        predicate_targets[0, 0, 1, 0] = 1.0
        predicate_targets[0, 1, 0, 1] = 1.0
        supervision = torch.tensor([True, False, True])

        losses = supervised_relation_loss(
            outputs,
            pair_targets,
            predicate_targets,
            RelationLossConfig(
                sampler_loss_weight=0.0,
                pair_loss_weight=0.0,
                predicate_loss_weight=1.0,
                predicate_objective="batch-local-infonce",
                predicate_contrastive_temperature=1.0,
            ),
            predicate_supervision_mask=supervision,
            explicit_holdout_mask=~supervision,
        )
        self.assertEqual(
            float(losses["predicate_rows_skipped"]),
            1.0,
        )
        self.assertEqual(
            float(losses["predicate_contrast_set_size"]),
            1.0,
        )
        self.assertEqual(float(losses["predicate_loss"]), 0.0)

    def test_batch_local_infonce_rejects_bce_negative_reweighting(self):
        pred_logits = torch.zeros(1, 1, 2)
        pair_logits = torch.zeros(1, 1)
        outputs = RelationTrainingOutputs(
            runtime=(
                pred_logits,
                pair_logits,
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([[1]], dtype=torch.int64),
                torch.tensor([[True]]),
            ),
            sampler_logits=torch.zeros(1, 4),
            sampler_valid=torch.ones(1, 4, dtype=torch.bool),
            predicate_query=torch.tensor([[[1.0, 0.0]]]),
            predicate_query_raw=torch.tensor([[[1.0, 0.0]]]),
            predicate_bank=torch.eye(2),
        )
        pair_targets = torch.zeros(1, 2, 2)
        pair_targets[0, 0, 1] = 1.0
        predicate_targets = torch.zeros(1, 2, 2, 2)
        predicate_targets[0, 0, 1, 0] = 1.0

        with self.assertRaises(ValueError):
            supervised_relation_loss(
                outputs,
                pair_targets,
                predicate_targets,
                RelationLossConfig(
                    sampler_loss_weight=0.0,
                    pair_loss_weight=0.0,
                    predicate_loss_weight=1.0,
                    predicate_objective="batch-local-infonce",
                ),
                predicate_negative_weights=torch.tensor([1.0, 0.1]),
            )

    def test_infonce_calibration_uses_safe_seen_columns_only(self):
        pred_logits = torch.tensor(
            [[[2.0, -2.0, 10.0]]],
            requires_grad=True,
        )
        pair_logits = torch.zeros(1, 1)
        query = torch.tensor(
            [[[1.0, 0.0]]], requires_grad=True
        )
        outputs = RelationTrainingOutputs(
            runtime=(
                pred_logits,
                pair_logits,
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([[1]], dtype=torch.int64),
                torch.tensor([[True]]),
            ),
            sampler_logits=torch.zeros(1, 4),
            sampler_valid=torch.ones(1, 4, dtype=torch.bool),
            predicate_query=query,
            predicate_query_raw=query.clone(),
            predicate_bank=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ]
            ),
        )
        pair_targets = torch.zeros(1, 2, 2)
        pair_targets[0, 0, 1] = 1.0
        predicate_targets = torch.zeros(1, 2, 2, 3)
        predicate_targets[0, 0, 1, 0] = 1.0
        supervision = torch.tensor([True, True, False])

        losses = supervised_relation_loss(
            outputs,
            pair_targets,
            predicate_targets,
            RelationLossConfig(
                sampler_loss_weight=0.0,
                pair_loss_weight=0.0,
                predicate_loss_weight=1.0,
                predicate_objective="batch-local-infonce",
                predicate_contrastive_temperature=1.0,
                predicate_calibration_loss_weight=1.0,
            ),
            predicate_supervision_mask=supervision,
            explicit_holdout_mask=~supervision,
            predicate_contrastive_negative_mask=torch.tensor(
                [True, True, True]
            ),
        )

        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logits[0, 0, :2],
            torch.tensor([1.0, 0.0]),
        )
        self.assertTrue(
            torch.allclose(losses["predicate_calibration_loss"], expected)
        )
        self.assertTrue(
            torch.allclose(losses["predicate_loss"], expected)
        )
        self.assertEqual(
            float(losses["predicate_calibration_rows"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(losses["predicate_calibration_column_fraction"]),
            2.0 / 3.0,
            places=6,
        )

        losses["loss"].backward()
        self.assertIsNotNone(pred_logits.grad)
        self.assertGreater(
            float(pred_logits.grad[0, 0, :2].abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(pred_logits.grad[0, 0, 2].abs()),
            0.0,
        )

    def test_infonce_calibration_skips_holdout_only_rows(self):
        pred_logits = torch.tensor(
            [[[0.0, 0.0, 2.0]]],
            requires_grad=True,
        )
        pair_logits = torch.zeros(1, 1)
        query = torch.tensor(
            [[[1.0, 0.0]]], requires_grad=True
        )
        outputs = RelationTrainingOutputs(
            runtime=(
                pred_logits,
                pair_logits,
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([[1]], dtype=torch.int64),
                torch.tensor([[True]]),
            ),
            sampler_logits=torch.zeros(1, 4),
            sampler_valid=torch.ones(1, 4, dtype=torch.bool),
            predicate_query=query,
            predicate_query_raw=query.clone(),
            predicate_bank=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ]
            ),
        )
        pair_targets = torch.zeros(1, 2, 2)
        pair_targets[0, 0, 1] = 1.0
        predicate_targets = torch.zeros(1, 2, 2, 3)
        predicate_targets[0, 0, 1, 2] = 1.0
        supervision = torch.tensor([True, True, False])

        losses = supervised_relation_loss(
            outputs,
            pair_targets,
            predicate_targets,
            RelationLossConfig(
                sampler_loss_weight=0.0,
                pair_loss_weight=0.0,
                predicate_loss_weight=1.0,
                predicate_objective="batch-local-infonce",
                predicate_contrastive_temperature=1.0,
                predicate_calibration_loss_weight=1.0,
            ),
            predicate_supervision_mask=supervision,
            explicit_holdout_mask=~supervision,
            predicate_contrastive_negative_mask=torch.tensor(
                [True, True, False]
            ),
        )
        self.assertEqual(float(losses["predicate_loss"]), 0.0)
        self.assertEqual(
            float(losses["predicate_calibration_loss"]),
            0.0,
        )
        self.assertEqual(
            float(losses["predicate_calibration_rows"]),
            0.0,
        )
        self.assertEqual(
            float(losses["predicate_calibration_rows_skipped"]),
            1.0,
        )

    def test_predicate_calibration_requires_infonce(self):
        with self.assertRaises(ValueError):
            RelationLossConfig(
                predicate_objective="bce",
                predicate_calibration_loss_weight=0.25,
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

    def test_union_contact_onnx_matches_native_runtime_contract(self):
        torch.manual_seed(63)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            visual_config("union-contact"),
        )
        # Exercise a non-zero learned residual rather than allowing the
        # zero-initialized branch to become an export no-op.
        with torch.no_grad():
            model.union_projection.weight.normal_(0.0, 0.01)
            model.contact_projection.weight.normal_(0.0, 0.01)

        box_tensor, box_counts = boxes()
        image = torch.rand(1, 3, 8, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relation-union-contact.onnx"
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

    def test_rich_geometry_checkpoint_round_trip_preserves_mode(self):
        torch.manual_seed(68)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            geometry_config("rich"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rich-geometry.pt"
            save_checkpoint(
                path,
                model,
                backbone_model="synthetic/test-backbone",
                predicates=["beside", "holding", "riding"],
            )
            payload = load_payload(path)
            restored = config_from_payload(payload)

        self.assertEqual(
            restored.pair_geometry_evidence,
            "rich",
        )
        self.assertIn(
            "rich_geometry_projection.weight",
            payload["state_dict"],
        )
        self.assertIn(
            "rich_geometry_sampler.weight",
            payload["state_dict"],
        )

    def test_contact_checkpoint_round_trip_preserves_mode(self):
        torch.manual_seed(63)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            visual_config("contact"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contact.pt"
            save_checkpoint(
                path,
                model,
                backbone_model="synthetic/test-backbone",
                predicates=["beside", "holding", "riding"],
            )
            payload = load_payload(path)
            restored = config_from_payload(payload)

        self.assertEqual(
            restored.pair_visual_evidence,
            "contact",
        )
        self.assertNotIn(
            "union_projection.weight",
            payload["state_dict"],
        )
        self.assertIn(
            "contact_projection.weight",
            payload["state_dict"],
        )

    def test_union_contact_checkpoint_round_trip_preserves_mode(self):
        torch.manual_seed(64)
        model = KFRelationModel(
            ToyBackbone(),
            torch.randn(3, 6),
            visual_config("union-contact"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "union-contact.pt"
            save_checkpoint(
                path,
                model,
                backbone_model="synthetic/test-backbone",
                predicates=["beside", "holding", "riding"],
            )
            payload = load_payload(path)
            restored = config_from_payload(payload)

        self.assertEqual(
            restored.pair_visual_evidence,
            "union-contact",
        )
        self.assertIn(
            "union_projection.weight",
            payload["state_dict"],
        )
        self.assertIn(
            "contact_projection.weight",
            payload["state_dict"],
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
