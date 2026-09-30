from __future__ import annotations

import copy
import unittest

from apache_release_scalars import (
    MODEL_SCALARS,
    OBJECTIVE_SCALARS,
    RECIPE_SCALARS,
    SCHEMA,
    validate_released_scalar_contract,
)


def drift(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.01
    if isinstance(value, str):
        return value + "-drift"
    raise TypeError(type(value))


class ApacheReleasedScalarContractTest(unittest.TestCase):
    def test_exact_released_scalars_qualify(self):
        report = validate_released_scalar_contract(
            dict(RECIPE_SCALARS),
            dict(MODEL_SCALARS),
            dict(OBJECTIVE_SCALARS),
        )
        self.assertEqual(
            report["schema"],
            SCHEMA,
        )
        self.assertTrue(
            report["matches_released"]
        )
        self.assertEqual(
            report["recipe"],
            RECIPE_SCALARS,
        )
        self.assertEqual(
            report["model"],
            MODEL_SCALARS,
        )
        self.assertEqual(
            report["objective"],
            OBJECTIVE_SCALARS,
        )

    def test_every_released_scalar_drift_fails_closed(self):
        cases = (
            ("recipe", RECIPE_SCALARS),
            ("model", MODEL_SCALARS),
            ("objective", OBJECTIVE_SCALARS),
        )
        for block_name, expected in cases:
            for key, value in expected.items():
                with self.subTest(
                    block=block_name,
                    key=key,
                ):
                    recipe = dict(RECIPE_SCALARS)
                    model = dict(MODEL_SCALARS)
                    objective = dict(
                        OBJECTIVE_SCALARS
                    )
                    target = {
                        "recipe": recipe,
                        "model": model,
                        "objective": objective,
                    }[block_name]
                    target[key] = drift(value)
                    with self.assertRaisesRegex(
                        ValueError,
                        key,
                    ):
                        validate_released_scalar_contract(
                            recipe,
                            model,
                            objective,
                        )

    def test_missing_released_scalar_fails_closed(self):
        for block_name, expected in (
            ("recipe", RECIPE_SCALARS),
            ("model", MODEL_SCALARS),
            ("objective", OBJECTIVE_SCALARS),
        ):
            key = next(iter(expected))
            recipe = dict(RECIPE_SCALARS)
            model = dict(MODEL_SCALARS)
            objective = dict(OBJECTIVE_SCALARS)
            target = {
                "recipe": recipe,
                "model": model,
                "objective": objective,
            }[block_name]
            del target[key]
            with self.subTest(
                block=block_name,
                key=key,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    key,
                ):
                    validate_released_scalar_contract(
                        recipe,
                        model,
                        objective,
                    )


if __name__ == "__main__":
    unittest.main()
