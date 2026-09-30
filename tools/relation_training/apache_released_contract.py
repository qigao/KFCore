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




RELEASED_STRUCTURE: dict[str, object] = {
    "context_self_layers": 2,
    "context_cross_layers": 2,
    "interaction_dependency_layers": 2,
    "interaction_grounding_layers": 1,
    "attention_heads": 8,
    "ffn_ratio": 2.0,
    "deformable_points": 4,
    "deformable_heads": 8,
    "deformable_nulls": 2,
    "pe_num_freqs": 16,
    "pe_max_octave": 7.0,
    "vocab_projection_layers": 2,
    "logit_scale_init": 5.0,
}


def validate_released_structure(
    model: object,
) -> dict[str, object]:
    """Validate the Apache hard-coded structure without exposing new knobs."""
    relation = getattr(
        model,
        "apache_relation_transformer",
        None,
    )
    deformable = getattr(
        model,
        "apache_deformable_read",
        None,
    )
    interaction = getattr(
        model,
        "apache_relation_interaction",
        None,
    )
    spatial_pool = getattr(
        model,
        "apache_spatial_pool",
        None,
    )
    box_prompt = getattr(
        model,
        "apache_box_prompt_encoder",
        None,
    )
    vocab = getattr(
        model,
        "apache_vocab_head",
        None,
    )
    if any(
        value is None
        for value in (
            relation,
            deformable,
            interaction,
            spatial_pool,
            box_prompt,
            vocab,
        )
    ):
        raise ValueError(
            "released structure requires the full Apache model stack"
        )

    context_self_layers = len(
        relation.self_layers
    )
    context_cross_layers = (
        len(relation.cross_layers) + 1
    )
    dependency_layers = len(
        interaction.dependency_layers
    )
    grounding_layers = len(
        interaction.grounding_layers
    )
    heads = {
        int(layer.self_attn.num_heads)
        for layer in relation.self_layers
    }
    heads.update(
        int(layer.self_attn.num_heads)
        for layer in relation.cross_layers
    )
    heads.add(
        int(relation.last_cross.self_attn.num_heads)
    )
    heads.add(
        int(relation.last_cross.cross_attn.num_heads)
    )
    heads.update(
        int(layer.self_attn.num_heads)
        for layer in interaction.dependency_layers
    )
    heads.update(
        int(layer.self_attn.num_heads)
        for layer in interaction.grounding_layers
    )
    heads.add(
        int(spatial_pool.n_heads)
    )
    heads.add(
        int(deformable.n_heads)
    )
    if heads != {8}:
        raise ValueError(
            "Apache released structure requires 8 attention heads"
        )

    d_model = int(vocab.d_model)
    ffn_dims = {
        int(layer.linear1.out_features)
        for layer in relation.self_layers
    }
    ffn_dims.update(
        int(layer.linear1.out_features)
        for layer in relation.cross_layers
    )
    ffn_dims.update(
        int(layer.linear1.out_features)
        for layer in interaction.dependency_layers
    )
    ffn_dims.update(
        int(layer.linear1.out_features)
        for layer in interaction.grounding_layers
    )
    ffn_dims.add(
        int(
            relation.last_cross.ffn[0].out_features
        )
    )
    expected_ffn_dim = int(
        d_model
        * float(
            RELEASED_STRUCTURE["ffn_ratio"]
        )
    )
    if ffn_dims != {expected_ffn_dim}:
        raise ValueError(
            "Apache released structure requires ffn_ratio=2.0"
        )

    projection = vocab.proj
    projection_layers = sum(
        1
        for module in projection.modules()
        if module is not projection
        and module.__class__.__name__ == "Linear"
    )
    logit_scale_init = float(
        vocab.logit_scale.detach().exp().cpu().item()
    )

    report = {
        "context_self_layers": context_self_layers,
        "context_cross_layers": context_cross_layers,
        "interaction_dependency_layers": dependency_layers,
        "interaction_grounding_layers": grounding_layers,
        "attention_heads": next(iter(heads)),
        "ffn_ratio": (
            float(expected_ffn_dim)
            / float(d_model)
        ),
        "deformable_points": int(
            deformable.n_points
        ),
        "deformable_heads": int(
            deformable.n_heads
        ),
        "deformable_nulls": int(
            deformable.null_slots
        ),
        "pe_num_freqs": int(
            spatial_pool.scene_pe.num_freqs
        ),
        "pe_max_octave": float(
            spatial_pool.scene_pe.max_octave
        ),
        "box_pe_num_freqs": int(
            box_prompt.num_freqs
        ),
        "box_pe_max_octave": float(
            box_prompt.max_octave
        ),
        "vocab_projection_layers": (
            projection_layers
        ),
        "logit_scale_init": logit_scale_init,
    }

    return validate_released_structure_report(
        {
            "schema": "kfcore.apache-released-structure/1",
            "matches_released": True,
            **report,
        }
    )


def validate_released_structure_report(
    report: Mapping[str, Any],
) -> dict[str, object]:
    if (
        report.get("schema")
        != "kfcore.apache-released-structure/1"
        or report.get("matches_released") is not True
    ):
        raise ValueError(
            "invalid Apache released structure evidence"
        )
    expected = dict(RELEASED_STRUCTURE)
    for key, target in expected.items():
        actual = _value(report, key)
        if not _matches(actual, target):
            raise ValueError(
                f"Apache released structure {key} "
                f"must be {target!r}; got {actual!r}"
            )
    if (
        _value(report, "box_pe_num_freqs")
        != RELEASED_STRUCTURE["pe_num_freqs"]
        or not _matches(
            _value(report, "box_pe_max_octave"),
            RELEASED_STRUCTURE["pe_max_octave"],
        )
    ):
        raise ValueError(
            "Apache released box positional encoding differs from 16/7"
        )
    return dict(report)


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
