from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from apache_objective import (
    SOURCE_ALLOW_SCHEMA,
    ApacheObjectiveConfig,
    ApacheReferenceObjective,
    BatchLocalInfoNCE,
    PredicateOntology,
    load_source_column_allow,
    swap_direction_hinge_dense,
)
from model import RelationTrainingOutputs


def ontology() -> PredicateOntology:
    pos_w = torch.eye(4, dtype=torch.float32)
    pos_w[0, 1] = 0.5
    neg_lw = torch.zeros(4, 4, dtype=torch.float32)
    neg_lw[0, 3] = math.log(0.5)
    inverse = torch.zeros(4, 4, dtype=torch.bool)
    inverse[0, 2] = True
    inverse[2, 0] = True
    sym = torch.tensor([0.0, 1.0, 0.0, 0.5])
    return PredicateOntology(
        predicates=("above", "over", "below", "holding"),
        pos_w=pos_w,
        neg_lw=neg_lw,
        sym=sym,
        inverse_mask=inverse,
    )


def training_outputs() -> RelationTrainingOutputs:
    pred_logits = torch.tensor(
        [
            [
                [2.0, 0.5, -1.0, 3.0],
                [-0.5, 0.2, 1.5, 2.5],
                [0.1, -0.2, 1.8, 2.0],
                [0.3, 0.1, -0.5, 2.2],
            ]
        ],
        requires_grad=True,
    )
    pair_logits = torch.tensor(
        [[1.0, -0.5, 0.8, -0.2]],
        requires_grad=True,
    )
    sub_idx = torch.tensor([[0, 1, 1, 2]], dtype=torch.int64)
    obj_idx = torch.tensor([[1, 0, 2, 1]], dtype=torch.int64)
    valid = torch.ones(1, 4, dtype=torch.bool)

    semantic = torch.tensor(
        [
            [
                [0.2, 0.9, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.2, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    spatial = torch.tensor(
        [
            [
                [0.1, 1.0, 0.0, 0.0],
                [1.0, 0.1, 0.0, 0.0],
                [0.0, 0.1, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.1],
            ]
        ],
        requires_grad=True,
    )
    bank = torch.eye(4, dtype=torch.float32)
    alpha = torch.tensor(
        [0.0, 0.25, 1.0, 0.5],
        requires_grad=True,
    )

    object_subject = torch.tensor(
        [
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    object_object = object_subject.clone().detach().requires_grad_(True)

    return RelationTrainingOutputs(
        runtime=(
            pred_logits,
            pair_logits,
            sub_idx,
            obj_idx,
            valid,
        ),
        sampler_logits=torch.zeros(1, 9),
        sampler_valid=torch.ones(1, 9, dtype=torch.bool),
        predicate_query=F.normalize(semantic, dim=-1),
        predicate_query_raw=semantic,
        predicate_bank=bank,
        predicate_spatial_query=spatial,
        predicate_alpha=alpha,
        object_subject_query=object_subject,
        object_object_query=object_object,
        sampler_geo_loss=torch.tensor(
            0.2,
            requires_grad=True,
        ),
        sampler_relatedness_loss=torch.tensor(
            0.3,
            requires_grad=True,
        ),
        sampler_pair_negative_weights=torch.tensor(
            [[1.0, 0.3, 1.0, 0.3]]
        ),
    )


def dense_targets() -> torch.Tensor:
    targets = torch.zeros(1, 3, 3, 4)
    targets[0, 0, 1, 0] = 1.0
    targets[0, 1, 2, 2] = 1.0
    return targets


class ApacheObjectiveTest(unittest.TestCase):
    def test_soft_supervision_loader_restores_inverse_full_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "meta.json"
            npz = root / "soft.npz"
            meta.write_text(
                json.dumps(
                    {
                        "predicates": [
                            "above",
                            "over",
                            "below",
                            "holding",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            np.savez(
                npz,
                predicates=np.asarray(
                    ["above", "over", "below", "holding"]
                ),
                pos_i=np.asarray([0]),
                pos_j=np.asarray([1]),
                pos_w=np.asarray([0.5], dtype=np.float16),
                neg_i=np.asarray([0, 0]),
                neg_j=np.asarray([2, 3]),
                neg_lw=np.asarray(
                    [math.log(0.2), math.log(0.5)],
                    dtype=np.float16,
                ),
                inv_i=np.asarray([0]),
                inv_j=np.asarray([2]),
                inv_w=np.asarray([1.0], dtype=np.float32),
                sym=np.asarray([0.0, 1.0, 0.0, 0.5]),
            )
            loaded = PredicateOntology.from_soft_supervision(
                meta,
                npz,
            )

        self.assertEqual(
            float(loaded.pos_w[0, 0]),
            1.0,
        )
        self.assertGreater(
            float(loaded.pos_w[0, 1]),
            0.0,
        )
        self.assertTrue(bool(loaded.inverse_mask[0, 2]))
        self.assertEqual(
            float(loaded.neg_lw[0, 2]),
            0.0,
        )
        self.assertLess(
            float(loaded.neg_lw[0, 3]),
            0.0,
        )

    def test_source_allow_requires_exact_known_predicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": SOURCE_ALLOW_SCHEMA,
                        "sources": [
                            {
                                "name": "source-a",
                                "predicates": [
                                    "above",
                                    "below",
                                ],
                            },
                            {
                                "name": "source-b",
                                "predicates": [
                                    "holding",
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            names, allow = load_source_column_allow(
                path,
                ("above", "over", "below", "holding"),
            )

        self.assertEqual(names, ("source-a", "source-b"))
        self.assertTrue(
            torch.equal(
                allow,
                torch.tensor(
                    [
                        [True, False, True, False],
                        [False, False, False, True],
                    ]
                ),
            )
        )

    def test_infonce_contrast_set_includes_inverse(self):
        objective = BatchLocalInfoNCE(
            ontology(),
            temp=0.07,
            n_neg=0,
            hard_frac=0.5,
        )
        W = torch.eye(4)
        labels = torch.tensor(
            [[True, False, False, False]]
        )
        contrast = objective.build_set(
            labels.nonzero(as_tuple=True)[1],
            W,
        )
        self.assertEqual(
            set(contrast.tolist()),
            {0, 2},
        )

    def test_source_mask_cannot_hide_observed_positive(self):
        outputs = training_outputs()
        objective = ApacheReferenceObjective(
            ontology(),
            config=ApacheObjectiveConfig(
                n_neg=0,
                lambda_obj=0.0,
                lambda_swap=0.0,
                lambda_sigmoid=0.0,
                lambda_bg=0.0,
            ),
            source_column_allow=torch.tensor(
                [[False, True, True, True]]
            ),
        )
        with self.assertRaises(ValueError):
            objective(
                outputs,
                dense_targets(),
                source_ids=torch.tensor([0]),
            )

    def test_swap_hinge_uses_swapped_slot_and_symmetry_weight(self):
        W = torch.eye(4)
        q_sem = torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        q_spa = q_sem.clone()
        alpha = torch.zeros(4)
        sub = torch.tensor([[0, 1]])
        obj = torch.tensor([[1, 0]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        targets = torch.zeros(1, 2, 2, 4)
        targets[0, 0, 1, 0] = 1.0

        loss = swap_direction_hinge_dense(
            q_sem,
            q_spa,
            alpha,
            W,
            sub,
            obj,
            valid,
            targets,
            torch.tensor([0.0, 0.0, 0.0, 0.0]),
            margin=0.05,
        )
        self.assertAlmostEqual(
            float(loss),
            1.05,
            places=5,
        )

        symmetric = swap_direction_hinge_dense(
            q_sem,
            q_spa,
            alpha,
            W,
            sub,
            obj,
            valid,
            targets,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            margin=0.05,
        )
        self.assertEqual(float(symmetric), 0.0)

    def test_full_reference_objective_is_finite_weighted_and_differentiable(self):
        outputs = training_outputs()
        config = ApacheObjectiveConfig(
            n_neg=2,
            hard_frac=0.5,
            lambda_obj=0.10,
            lambda_swap=0.50,
            lambda_sigmoid=0.25,
            lambda_bg=0.05,
            lambda_geo=1.0,
            lambda_rel=1.0,
            bg_topk=2,
        )
        objective = ApacheReferenceObjective(
            ontology(),
            config=config,
            source_column_allow=torch.tensor(
                [[True, True, True, False]]
            ),
            object_text_bank=torch.eye(4)[:3],
        )
        labels = torch.tensor([[0, 1, -1]])
        losses = objective(
            outputs,
            dense_targets(),
            source_ids=torch.tensor([0]),
            object_label_indices=labels,
        )

        for key, value in losses.items():
            self.assertEqual(
                value.ndim,
                0,
                msg=key,
            )
            self.assertTrue(
                torch.isfinite(value),
                msg=key,
            )

        expected = (
            losses["loss_nce"]
            + 0.10 * losses["loss_obj"]
            + 0.50 * losses["loss_swap"]
            + 0.25 * losses["loss_sigmoid"]
            + 0.05 * losses["loss_background"]
            + losses["loss_geo"]
            + losses["loss_relatedness"]
        )
        self.assertTrue(
            torch.allclose(
                losses["loss"],
                expected,
            )
        )
        self.assertGreater(
            float(losses["contrast_set_size"]),
            0.0,
        )
        self.assertGreater(
            float(losses["synonym_positive_mass"]),
            0.0,
        )
        self.assertGreater(
            float(losses["source_masked_column_fraction"]),
            0.0,
        )
        self.assertGreater(
            float(losses["background_slots"]),
            0.0,
        )

        losses["loss"].backward()
        self.assertIsNotNone(
            outputs.predicate_query_raw.grad
        )
        assert outputs.predicate_spatial_query is not None
        self.assertIsNotNone(
            outputs.predicate_spatial_query.grad
        )
        assert outputs.predicate_alpha is not None
        self.assertIsNotNone(
            outputs.predicate_alpha.grad
        )
        assert outputs.object_subject_query is not None
        self.assertIsNotNone(
            outputs.object_subject_query.grad
        )


if __name__ == "__main__":
    unittest.main()
