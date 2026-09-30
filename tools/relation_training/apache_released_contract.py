from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


SCALAR_CONTRACT_SCHEMA = "kfcore.apache-released-scalars/1"

RELEASED_RECIPE_SCALARS: dict[str, object] = {
    "head_lr": 4.0e-4,
    "backbone_lr": 5.0e-5,
    "weight_decay": 1.0e-4,
    "warmup_steps": 500,
    "min_lr_factor": 0.01,
    "clip_grad": 1.0,
    "multi_scale": "0.5,1.5",
    "multi_scale_n": 7,
    "cfa_prob": 0.5,
    "cfa_alpha": 1.0,
}

RELEASED_MODEL_SCALARS: dict[str, object] = {
    "apache_context_dropout": 0.2,
    "apache_box_token_dropout": 0.3,
    "apache_pair_negative_floor": 0.3,
}

RELEASED_OBJECTIVE_SCALARS: dict[str, object] = {
    "infonce_temp": 0.07,
    "n_neg": 512,
    "hard_frac": 0.5,
    "lambda_obj": 0.10,
    "lambda_swap": 0.50,
    "lambda_sigmoid": 0.25,
    "lambda_bg": 0.05,
    "lambda_geo": 1.0,
    "lambda_rel": 1.0,
    "bg_topk": 5,
    "swap_margin": 0.05,
    "pair_negative_floor": 0.30,
}


def _value(
    source: Mapping[str, Any] | object,
    key: str,
) -> Any:
    if isinstance(source, Mapping):
        if key not in source:
            raise ValueError(
                f"released scalar contract is missing {key}"
            )
        return source[key]
    if not hasattr(source, key):
        raise ValueError(
            f"released scalar contract is missing {key}"
        )
    return getattr(source, key)


def _matches(
    actual: object,
    expected: object,
) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, int)
            and actual == expected
        )
    if isinstance(expected, float):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, (int, float))
            and math.isfinite(float(actual))
            and math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
    return actual == expected


def _validate_group(
    name: str,
    source: Mapping[str, Any] | object,
    expected: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, target in expected.items():
        actual = _value(source, key)
        if not _matches(actual, target):
            raise ValueError(
                f"Apache released {name} scalar {key} "
                f"must be {target!r}; got {actual!r}"
            )
        result[key] = actual
    return result


def validate_released_scalar_contract(
    *,
    recipe: Mapping[str, Any] | object,
    model: Mapping[str, Any] | object,
    objective: Mapping[str, Any] | object,
) -> dict[str, object]:
    """Fail closed on any mutable released scalar-hyperparameter drift."""
    return {
        "schema": SCALAR_CONTRACT_SCHEMA,
        "matches_released": True,
        "recipe": _validate_group(
            "recipe",
            recipe,
            RELEASED_RECIPE_SCALARS,
        ),
        "model": _validate_group(
            "model",
            model,
            RELEASED_MODEL_SCALARS,
        ),
        "objective": _validate_group(
            "objective",
            objective,
            RELEASED_OBJECTIVE_SCALARS,
        ),
    }
