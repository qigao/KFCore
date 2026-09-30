from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


SCHEMA = "kfcore.apache-released-scalar-contract/1"

RELEASED_HEAD_LR = 4.0e-4
RELEASED_BACKBONE_LR = 5.0e-5
RELEASED_WEIGHT_DECAY = 1.0e-4
RELEASED_WARMUP_STEPS = 500
RELEASED_MIN_LR_FACTOR = 0.01
RELEASED_CLIP_GRAD = 1.0
RELEASED_MULTI_SCALE = "0.5,1.5"
RELEASED_MULTI_SCALE_N = 7
RELEASED_CFA_PROB = 0.5
RELEASED_CFA_ALPHA = 1.0

RELEASED_CONTEXT_DROPOUT = 0.2
RELEASED_BOX_TOKEN_DROPOUT = 0.3
RELEASED_PAIR_NEGATIVE_FLOOR = 0.3

RELEASED_INFONCE_TEMP = 0.07
RELEASED_N_NEG = 512
RELEASED_HARD_FRAC = 0.5
RELEASED_LAMBDA_OBJ = 0.10
RELEASED_LAMBDA_SWAP = 0.50
RELEASED_LAMBDA_SIGMOID = 0.25
RELEASED_LAMBDA_BG = 0.05
RELEASED_LAMBDA_GEO = 1.0
RELEASED_LAMBDA_REL = 1.0
RELEASED_BG_TOPK = 5
RELEASED_SWAP_MARGIN = 0.05

RECIPE_SCALARS: dict[str, object] = {
    "head_lr": RELEASED_HEAD_LR,
    "backbone_lr": RELEASED_BACKBONE_LR,
    "weight_decay": RELEASED_WEIGHT_DECAY,
    "warmup_steps": RELEASED_WARMUP_STEPS,
    "min_lr_factor": RELEASED_MIN_LR_FACTOR,
    "clip_grad": RELEASED_CLIP_GRAD,
    "multi_scale": RELEASED_MULTI_SCALE,
    "multi_scale_n": RELEASED_MULTI_SCALE_N,
    "cfa_prob": RELEASED_CFA_PROB,
    "cfa_alpha": RELEASED_CFA_ALPHA,
}

MODEL_SCALARS: dict[str, object] = {
    "apache_context_dropout": RELEASED_CONTEXT_DROPOUT,
    "apache_box_token_dropout": RELEASED_BOX_TOKEN_DROPOUT,
    "apache_pair_negative_floor": RELEASED_PAIR_NEGATIVE_FLOOR,
    "apache_cfa_prob": RELEASED_CFA_PROB,
    "apache_cfa_alpha": RELEASED_CFA_ALPHA,
    "allow_training_multiscale": True,
}

OBJECTIVE_SCALARS: dict[str, object] = {
    "infonce_temp": RELEASED_INFONCE_TEMP,
    "n_neg": RELEASED_N_NEG,
    "hard_frac": RELEASED_HARD_FRAC,
    "lambda_obj": RELEASED_LAMBDA_OBJ,
    "lambda_swap": RELEASED_LAMBDA_SWAP,
    "lambda_sigmoid": RELEASED_LAMBDA_SIGMOID,
    "lambda_bg": RELEASED_LAMBDA_BG,
    "lambda_geo": RELEASED_LAMBDA_GEO,
    "lambda_rel": RELEASED_LAMBDA_REL,
    "bg_topk": RELEASED_BG_TOPK,
    "swap_margin": RELEASED_SWAP_MARGIN,
    "pair_negative_floor": RELEASED_PAIR_NEGATIVE_FLOOR,
}


def _equal(actual: object, expected: object) -> bool:
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


def _validate_block(
    name: str,
    payload: Mapping[str, Any],
    expected: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, target in expected.items():
        if key not in payload:
            raise ValueError(
                f"Apache released {name} scalar is missing: {key}"
            )
        actual = payload[key]
        if not _equal(actual, target):
            raise ValueError(
                f"Apache released {name} scalar drift: "
                f"{key}={actual!r}, expected {target!r}"
            )
        result[key] = actual
    return result


def validate_released_scalar_contract(
    recipe: Mapping[str, Any],
    model: Mapping[str, Any],
    objective: Mapping[str, Any],
) -> dict[str, object]:
    if not isinstance(recipe, Mapping):
        raise ValueError("released recipe scalar block must be a mapping")
    if not isinstance(model, Mapping):
        raise ValueError("released model scalar block must be a mapping")
    if not isinstance(objective, Mapping):
        raise ValueError("released objective scalar block must be a mapping")

    return {
        "schema": SCHEMA,
        "matches_released": True,
        "recipe": _validate_block(
            "recipe",
            recipe,
            RECIPE_SCALARS,
        ),
        "model": _validate_block(
            "model",
            model,
            MODEL_SCALARS,
        ),
        "objective": _validate_block(
            "objective",
            objective,
            OBJECTIVE_SCALARS,
        ),
    }
