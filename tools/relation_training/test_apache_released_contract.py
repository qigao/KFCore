from __future__ import annotations

import copy
import unittest

import torch

from apache_released_contract import (
    RELEASED_MODEL_SCALARS,
    RELEASED_OBJECTIVE_SCALARS,
    RELEASED_RECIPE_SCALARS,
    RELEASED_STRUCTURE,
    SCALAR_CONTRACT_SCHEMA,
    validate_released_scalar_contract,
    validate_released_structure,
    validate_released_structure_report,
)
from model import KFRelationModel, RelationModelConfig
from test_model import ToyBackbone


def drift(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.001
    if isinstance(value, str):
        return value + "-drift"
    raise TypeError(type(value))


def released_model() -> KFRelationModel:
    return KFRelationModel(
        ToyBackbone(),
        torch.randn(3, 512),
        RelationModelConfig(
            image_size=448,
            max_boxes=40,
            pair_budget=128,
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
            apache_context_dropout=0.2,
            apache_box_token_dropout=0.3,
            apache_pair_negative_floor=0.3,
            apache_cfa_prob=0.5,
            apache_cfa_alpha=1.0,
            allow_training_multiscale=True,
        ),
    )


class ApacheReleasedScalarContractTest(
    unittest.TestCase
):
    def test_all_released_scalars_match(self):
        report = validate_released_scalar_contract(
            recipe=RELEASED_RECIPE_SCALARS,
            model=RELEASED_MODEL_SCALARS,
            objective=RELEASED_OBJECTIVE_SCALARS,
        )
        self.assertEqual(
            report["schema"],
            SCALAR_CONTRACT_SCHEMA,
        )
        self.assertTrue(
            report["matches_released"]
        )

    def test_every_recipe_scalar_drift_is_rejected(self):
        for key, expected in RELEASED_RECIPE_SCALARS.items():
            with self.subTest(key=key):
                values = dict(
                    RELEASED_RECIPE_SCALARS
                )
                values[key] = drift(expected)
                with self.assertRaisesRegex(
                    ValueError,
                    key,
                ):
                    validate_released_scalar_contract(
                        recipe=values,
                        model=RELEASED_MODEL_SCALARS,
                        objective=RELEASED_OBJECTIVE_SCALARS,
                    )

    def test_every_model_scalar_drift_is_rejected(self):
        for key, expected in RELEASED_MODEL_SCALARS.items():
            with self.subTest(key=key):
                values = dict(
                    RELEASED_MODEL_SCALARS
                )
                values[key] = drift(expected)
                with self.assertRaisesRegex(
                    ValueError,
                    key,
                ):
                    validate_released_scalar_contract(
                        recipe=RELEASED_RECIPE_SCALARS,
                        model=values,
                        objective=RELEASED_OBJECTIVE_SCALARS,
                    )

    def test_every_objective_scalar_drift_is_rejected(self):
        for key, expected in RELEASED_OBJECTIVE_SCALARS.items():
            with self.subTest(key=key):
                values = dict(
                    RELEASED_OBJECTIVE_SCALARS
                )
                values[key] = drift(expected)
                with self.assertRaisesRegex(
                    ValueError,
                    key,
                ):
                    validate_released_scalar_contract(
                        recipe=RELEASED_RECIPE_SCALARS,
                        model=RELEASED_MODEL_SCALARS,
                        objective=values,
                    )

    def test_hard_coded_structure_matches_release(self):
        torch.manual_seed(172)
        model = released_model()
        report = validate_released_structure(
            model
        )
        self.assertTrue(
            report["matches_released"]
        )
        for key, expected in RELEASED_STRUCTURE.items():
            self.assertEqual(
                report[key],
                expected,
                key,
            )
        self.assertEqual(
            report["box_pe_num_freqs"],
            16,
        )
        self.assertEqual(
            report["box_pe_max_octave"],
            7.0,
        )

    def test_persisted_structure_drift_is_rejected(self):
        torch.manual_seed(173)
        report = validate_released_structure(
            released_model()
        )
        for key, expected in RELEASED_STRUCTURE.items():
            with self.subTest(key=key):
                changed = copy.deepcopy(
                    report
                )
                changed[key] = drift(expected)
                with self.assertRaisesRegex(
                    ValueError,
                    key,
                ):
                    validate_released_structure_report(
                        changed
                    )

        changed = copy.deepcopy(report)
        changed["box_pe_num_freqs"] = 15
        with self.assertRaisesRegex(
            ValueError,
            "box positional",
        ):
            validate_released_structure_report(
                changed
            )


if __name__ == "__main__":
    unittest.main()
