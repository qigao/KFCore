from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from apache_pair_sampler import (
    RELEASED_FINAL_BUDGET,
    RELEASED_GEO_BUDGET,
)
from apache_released_contract import (
    validate_released_scalar_contract,
    validate_released_structure_report,
)
from apache_spatial_flags import (
    DERIVATION_ALGORITHM,
    DERIVATION_SCHEMA,
    MAJORITY_COMPARATOR,
    MAJORITY_THRESHOLD,
    SPATIAL_BIT,
)
from apache_source_allow import (
    DERIVATION_ALGORITHM as SOURCE_ALLOW_DERIVATION_ALGORITHM,
    DERIVATION_SCHEMA as SOURCE_ALLOW_DERIVATION_SCHEMA,
    RELEASED_RESTRICTED_SOURCES,
)
from apache_training_recipe import (
    RELEASED_AMP,
    RELEASED_AMP_DTYPE,
    RELEASED_D_MODEL,
    RELEASED_EMA_DECAY,
    RELEASED_IMAGE_SIZE,
    RELEASED_MAX_BOXES,
    RELEASED_PHOTOMETRIC_AUGMENT,
    RELEASED_TEXT_DIM,
)
from apache_mixture import (
    RELEASED_MICRO_BATCH_SIZE,
    RELEASED_MIX_FRACTIONS,
    RELEASED_SAMPLES_PER_EPOCH,
    RELEASED_SEED,
    RELEASED_SOURCE_NAMES,
    RELEASED_WORLD_SIZE,
)


CORPUS_SCHEMA = "kfcore.apache-released-corpus/1"
QUALIFICATION_SCHEMA = "kfcore.apache-release-training-qualification/1"
TRAINING_SCHEMA = "kfcore.relation-training-run/1"
REFERENCE_SOURCE_COMMIT = (
    "4a07de9d06f2e3f14309753b7907cf1d3a263b08"
)
RELEASED_EPOCHS = 12
RELEASED_EFFECTIVE_BATCH_SIZE = (
    RELEASED_MICRO_BATCH_SIZE * RELEASED_WORLD_SIZE
)
PROVENANCE_KINDS = {
    "published-pack",
    "deterministic-rebuild",
    "unresolved",
}


def stable_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        stable_json_bytes(payload)
    ).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise ValueError(
            f"{path}: expected a JSON object"
        )
    return value


def require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            char not in "0123456789abcdef"
            for char in value
        )
    ):
        raise ValueError(
            f"{name} must be lowercase SHA-256 hex"
        )
    return value


def require_non_empty_string(
    value: object,
    name: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{name} must be a non-empty string"
        )
    return value


def _require_positive_int(
    value: object,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer"
        )
    return value


def _require_non_negative_int(
    value: object,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )
    return value


def _finite_number(
    value: object,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"{name} must be finite"
        )
    return float(value)


def _validate_spatial_derivation(
    payload: object,
    *,
    sidecar_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "predicate spatial-flags derivation must be an object"
        )
    if payload.get("schema") != DERIVATION_SCHEMA:
        raise ValueError(
            "unsupported predicate spatial-flags derivation schema"
        )
    if payload.get("algorithm") != DERIVATION_ALGORITHM:
        raise ValueError(
            "predicate spatial-flags derivation algorithm differs from Apache"
        )
    if payload.get("spatial_bit") != SPATIAL_BIT:
        raise ValueError(
            "predicate spatial-flags derivation must use relation flag bit 0"
        )
    threshold = _finite_number(
        payload.get("majority_threshold"),
        "predicate spatial majority threshold",
    )
    if not math.isclose(
        threshold,
        MAJORITY_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "predicate spatial majority threshold must be 0.5"
        )
    if (
        payload.get("majority_comparator")
        != MAJORITY_COMPARATOR
    ):
        raise ValueError(
            "predicate spatial majority comparator must be >="
        )
    derived_sha = require_sha256(
        payload.get("sidecar_sha256"),
        "predicate spatial sidecar_sha256",
    )
    if derived_sha != sidecar_sha256:
        raise ValueError(
            "predicate spatial derivation sidecar hash does not match corpus"
        )

    sources = payload.get("sources")
    if (
        not isinstance(sources, list)
        or len(sources)
        != len(RELEASED_SOURCE_NAMES)
    ):
        raise ValueError(
            "predicate spatial derivation must describe all released sources"
        )
    normalized_sources: list[
        dict[str, Any]
    ] = []
    names: list[str] = []
    for index, source in enumerate(
        sources
    ):
        if not isinstance(source, dict):
            raise ValueError(
                f"predicate spatial source {index} must be an object"
            )
        name = require_non_empty_string(
            source.get("source_name"),
            f"predicate spatial source {index} name",
        )
        names.append(name)
        local_predicate_count = _require_positive_int(
            source.get(
                "local_predicate_count"
            ),
            f"{name} local_predicate_count",
        )
        supported = _require_non_negative_int(
            source.get(
                "supported_predicate_count"
            ),
            f"{name} supported_predicate_count",
        )
        spatial = _require_non_negative_int(
            source.get(
                "local_spatial_majority_count"
            ),
            f"{name} local_spatial_majority_count",
        )
        relations = _require_non_negative_int(
            source.get("relations"),
            f"{name} relations",
        )
        if (
            supported > local_predicate_count
            or spatial > supported
        ):
            raise ValueError(
                f"{name} predicate spatial counts are inconsistent"
            )
        ignored = source.get(
            "ignored_predicates"
        )
        if (
            not isinstance(ignored, list)
            or any(
                not isinstance(value, str)
                or not value
                for value in ignored
            )
        ):
            raise ValueError(
                f"{name} ignored_predicates must be strings"
            )
        normalized_sources.append(
            {
                "source_name": name,
                "pack_split": require_non_empty_string(
                    source.get("pack_split"),
                    f"{name} pack_split",
                ),
                "meta_sha256": require_sha256(
                    source.get("meta_sha256"),
                    f"{name} meta_sha256",
                ),
                "rels_sha256": require_sha256(
                    source.get("rels_sha256"),
                    f"{name} rels_sha256",
                ),
                "relations": relations,
                "local_predicate_count": (
                    local_predicate_count
                ),
                "supported_predicate_count": (
                    supported
                ),
                "local_spatial_majority_count": (
                    spatial
                ),
                "ignored_predicates": list(
                    ignored
                ),
            }
        )
    if tuple(names) != RELEASED_SOURCE_NAMES:
        raise ValueError(
            "predicate spatial derivation source order differs from release"
        )

    union_count = _require_positive_int(
        payload.get(
            "union_predicate_count"
        ),
        "union_predicate_count",
    )
    supported_union = _require_positive_int(
        payload.get(
            "supported_union_predicate_count"
        ),
        "supported_union_predicate_count",
    )
    spatial_union = _require_positive_int(
        payload.get(
            "spatial_union_predicate_count"
        ),
        "spatial_union_predicate_count",
    )
    if (
        supported_union > union_count
        or spatial_union > supported_union
    ):
        raise ValueError(
            "predicate spatial union counts are inconsistent"
        )
    unsupported = payload.get(
        "unsupported_predicates"
    )
    if (
        not isinstance(unsupported, list)
        or any(
            not isinstance(value, str)
            or not value
            for value in unsupported
        )
    ):
        raise ValueError(
            "unsupported_predicates must be strings"
        )

    return {
        "schema": DERIVATION_SCHEMA,
        "algorithm": DERIVATION_ALGORITHM,
        "spatial_bit": SPATIAL_BIT,
        "majority_threshold": (
            MAJORITY_THRESHOLD
        ),
        "majority_comparator": (
            MAJORITY_COMPARATOR
        ),
        "sources": normalized_sources,
        "union_predicate_count": union_count,
        "supported_union_predicate_count": (
            supported_union
        ),
        "spatial_union_predicate_count": (
            spatial_union
        ),
        "unsupported_predicates": list(
            unsupported
        ),
        "sidecar_sha256": derived_sha,
    }


def _validate_source_allow_derivation(
    payload: object,
    *,
    sidecar_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "source-column derivation must be an object"
        )
    if (
        payload.get("schema")
        != SOURCE_ALLOW_DERIVATION_SCHEMA
    ):
        raise ValueError(
            "unsupported source-column derivation schema"
        )
    if (
        payload.get("algorithm")
        != SOURCE_ALLOW_DERIVATION_ALGORITHM
    ):
        raise ValueError(
            "source-column derivation algorithm differs from Apache"
        )
    restricted = payload.get(
        "restricted_sources"
    )
    if restricted != list(
        RELEASED_RESTRICTED_SOURCES
    ):
        raise ValueError(
            "released source-column derivation must restrict only hicodet"
        )
    union_count = _require_positive_int(
        payload.get(
            "union_predicate_count"
        ),
        "source-column union_predicate_count",
    )
    derived_sha = require_sha256(
        payload.get("sidecar_sha256"),
        "source-column sidecar_sha256",
    )
    if derived_sha != sidecar_sha256:
        raise ValueError(
            "source-column derivation sidecar hash does not match corpus"
        )

    sources = payload.get("sources")
    if (
        not isinstance(sources, list)
        or len(sources)
        != len(RELEASED_SOURCE_NAMES)
    ):
        raise ValueError(
            "source-column derivation must describe all released sources"
        )
    normalized_sources: list[
        dict[str, Any]
    ] = []
    names: list[str] = []
    for index, source in enumerate(
        sources
    ):
        if not isinstance(source, dict):
            raise ValueError(
                f"source-column source {index} must be an object"
            )
        name = require_non_empty_string(
            source.get("source_name"),
            f"source-column source {index} name",
        )
        names.append(name)
        local_count = _require_positive_int(
            source.get(
                "local_predicate_count"
            ),
            f"{name} local_predicate_count",
        )
        allowed_count = _require_positive_int(
            source.get(
                "allowed_predicate_count"
            ),
            f"{name} allowed_predicate_count",
        )
        if allowed_count > union_count:
            raise ValueError(
                f"{name} allowed predicate count exceeds union vocabulary"
            )
        expected_restricted = (
            name
            in RELEASED_RESTRICTED_SOURCES
        )
        if (
            source.get("restricted")
            is not expected_restricted
        ):
            raise ValueError(
                f"{name} restricted-source evidence differs from release"
            )
        if (
            not expected_restricted
            and allowed_count
            != union_count
        ):
            raise ValueError(
                f"{name} unrestricted source must allow every union predicate"
            )
        ignored = source.get(
            "ignored_predicates"
        )
        if (
            not isinstance(ignored, list)
            or any(
                not isinstance(value, str)
                or not value
                for value in ignored
            )
        ):
            raise ValueError(
                f"{name} ignored_predicates must be strings"
            )
        normalized_sources.append(
            {
                "source_name": name,
                "pack_split": require_non_empty_string(
                    source.get("pack_split"),
                    f"{name} pack_split",
                ),
                "meta_sha256": require_sha256(
                    source.get("meta_sha256"),
                    f"{name} meta_sha256",
                ),
                "local_predicate_count": (
                    local_count
                ),
                "restricted": (
                    expected_restricted
                ),
                "allowed_predicate_count": (
                    allowed_count
                ),
                "ignored_predicates": list(
                    ignored
                ),
            }
        )
    if tuple(names) != RELEASED_SOURCE_NAMES:
        raise ValueError(
            "source-column derivation source order differs from release"
        )

    return {
        "schema": SOURCE_ALLOW_DERIVATION_SCHEMA,
        "algorithm": SOURCE_ALLOW_DERIVATION_ALGORITHM,
        "restricted_sources": list(
            RELEASED_RESTRICTED_SOURCES
        ),
        "union_predicate_count": (
            union_count
        ),
        "sources": normalized_sources,
        "sidecar_sha256": derived_sha,
    }


def validate_corpus(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("schema") != CORPUS_SCHEMA:
        raise ValueError(
            "unsupported released corpus schema"
        )
    if (
        payload.get("reference_source_commit")
        != REFERENCE_SOURCE_COMMIT
    ):
        raise ValueError(
            "released corpus uses the wrong Apache source reference"
        )

    sources = payload.get("sources")
    if (
        not isinstance(sources, list)
        or len(sources) != len(RELEASED_SOURCE_NAMES)
    ):
        raise ValueError(
            "released corpus must describe exactly three sources"
        )

    normalized_sources: list[dict[str, Any]] = []
    for index, (source, expected_name) in enumerate(
        zip(sources, RELEASED_SOURCE_NAMES)
    ):
        if not isinstance(source, dict):
            raise ValueError(
                f"source {index} must be an object"
            )
        name = require_non_empty_string(
            source.get("name"),
            f"source {index} name",
        )
        if name != expected_name:
            raise ValueError(
                "released corpus source order/name mismatch"
            )
        provenance_kind = require_non_empty_string(
            source.get("provenance_kind"),
            f"{name} provenance_kind",
        )
        if provenance_kind not in PROVENANCE_KINDS:
            raise ValueError(
                f"unsupported provenance kind for {name}"
            )

        normalized: dict[str, Any] = {
            "name": name,
            "provenance_kind": provenance_kind,
        }

        if provenance_kind == "unresolved":
            normalized["note"] = require_non_empty_string(
                source.get("note"),
                f"{name} unresolved note",
            )
            normalized["resolved"] = False
        else:
            normalized["origin"] = require_non_empty_string(
                source.get("origin"),
                f"{name} origin",
            )
            normalized["revision"] = require_non_empty_string(
                source.get("revision"),
                f"{name} revision",
            )
            if provenance_kind == "deterministic-rebuild":
                normalized["recipe"] = (
                    require_non_empty_string(
                        source.get("recipe"),
                        f"{name} rebuild recipe",
                    )
                )
            normalized["annotations_sha256"] = (
                require_sha256(
                    source.get("annotations_sha256"),
                    f"{name} annotations_sha256",
                )
            )
            normalized["post_exclusion_count"] = (
                _require_positive_int(
                    source.get("post_exclusion_count"),
                    f"{name} post_exclusion_count",
                )
            )
            normalized["resolved"] = True

        normalized_sources.append(normalized)

    normalized_payload = {
        "schema": CORPUS_SCHEMA,
        "reference_source_commit": (
            REFERENCE_SOURCE_COMMIT
        ),
        "sources": normalized_sources,
        "exclude_ids_sha256": require_sha256(
            payload.get("exclude_ids_sha256"),
            "exclude_ids_sha256",
        ),
        "source_column_allow_sha256": require_sha256(
            payload.get("source_column_allow_sha256"),
            "source_column_allow_sha256",
        ),
        "ontology_meta_sha256": require_sha256(
            payload.get("ontology_meta_sha256"),
            "ontology_meta_sha256",
        ),
        "ontology_npz_sha256": require_sha256(
            payload.get("ontology_npz_sha256"),
            "ontology_npz_sha256",
        ),
        "neg_rate_table_sha256": require_sha256(
            payload.get("neg_rate_table_sha256"),
            "neg_rate_table_sha256",
        ),
        "predicate_embeddings_sha256": require_sha256(
            payload.get("predicate_embeddings_sha256"),
            "predicate_embeddings_sha256",
        ),
        "object_embeddings_sha256": require_sha256(
            payload.get("object_embeddings_sha256"),
            "object_embeddings_sha256",
        ),
        "predicate_spatial_flags_sha256": require_sha256(
            payload.get("predicate_spatial_flags_sha256"),
            "predicate_spatial_flags_sha256",
        ),
        "vocabulary_sha256": require_sha256(
            payload.get("vocabulary_sha256"),
            "vocabulary_sha256",
        ),
    }
    normalized_payload[
        "predicate_spatial_flags_derivation"
    ] = _validate_spatial_derivation(
        payload.get(
            "predicate_spatial_flags_derivation"
        ),
        sidecar_sha256=normalized_payload[
            "predicate_spatial_flags_sha256"
        ],
    )
    normalized_payload[
        "source_column_allow_derivation"
    ] = _validate_source_allow_derivation(
        payload.get(
            "source_column_allow_derivation"
        ),
        sidecar_sha256=normalized_payload[
            "source_column_allow_sha256"
        ],
    )
    return normalized_payload


def unresolved_sources(
    corpus: dict[str, Any],
) -> tuple[str, ...]:
    normalized = validate_corpus(corpus)
    return tuple(
        source["name"]
        for source in normalized["sources"]
        if not source["resolved"]
    )


def _require_released_fractions(
    values: object,
) -> None:
    if (
        not isinstance(values, list)
        or len(values) != len(RELEASED_MIX_FRACTIONS)
    ):
        raise ValueError(
            "training target fractions are missing"
        )
    for actual, expected in zip(
        values,
        RELEASED_MIX_FRACTIONS,
    ):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "training target fractions differ from released recipe"
            )


def validate_training_run(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("schema") != TRAINING_SCHEMA:
        raise ValueError(
            "unsupported relation training run schema"
        )
    if payload.get("frozen_backbone") is not False:
        raise ValueError(
            "released qualification requires full backbone fine-tuning"
        )

    recipe = payload.get("training_recipe")
    if not isinstance(recipe, dict):
        raise ValueError(
            "training recipe evidence is missing"
        )
    if recipe.get("name") != "apache-reference":
        raise ValueError(
            "training run is not the Apache reference recipe"
        )
    if recipe.get("matches_released_epoch_count") is not True:
        raise ValueError(
            "training run did not execute the released 12 epochs"
        )
    config = recipe.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            "Apache recipe config is missing"
        )
    if config.get("epochs") != RELEASED_EPOCHS:
        raise ValueError(
            "Apache recipe epoch count differs from release"
        )
    if config.get("micro_batch_size") != RELEASED_MICRO_BATCH_SIZE:
        raise ValueError(
            "Apache micro-batch size differs from release"
        )
    if config.get("grad_accum") != RELEASED_WORLD_SIZE:
        raise ValueError(
            "logical rank/gradient accumulation differs from release"
        )
    if config.get("text_dim") != RELEASED_TEXT_DIM:
        raise ValueError(
            "Apache recipe config requires text_dim=512"
        )

    if config.get("image_size") != RELEASED_IMAGE_SIZE:
        raise ValueError(
            "Apache recipe config requires img_size=448"
        )
    if config.get("geo_budget") != RELEASED_GEO_BUDGET:
        raise ValueError(
            "Apache recipe config requires geo_budget=400"
        )
    if config.get("final_budget") != RELEASED_FINAL_BUDGET:
        raise ValueError(
            "Apache recipe config requires final_budget=128"
        )
    if config.get("amp") is not RELEASED_AMP:
        raise ValueError(
            "Apache recipe config requires amp=true"
        )
    if config.get("amp_dtype") != RELEASED_AMP_DTYPE:
        raise ValueError(
            "Apache recipe config requires amp_dtype=bf16"
        )

    config_augment = config.get("augment")
    if (
        isinstance(config_augment, bool)
        or not isinstance(config_augment, (int, float))
        or not math.isclose(
            float(config_augment),
            RELEASED_PHOTOMETRIC_AUGMENT,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "Apache recipe config requires augment=0.3"
        )

    config_ema_decay = config.get("ema_decay")
    if (
        isinstance(config_ema_decay, bool)
        or not isinstance(config_ema_decay, (int, float))
        or not math.isclose(
            float(config_ema_decay),
            RELEASED_EMA_DECAY,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "Apache recipe config requires ema_decay=0.9998"
        )
    if (
        recipe.get("effective_batch_size")
        != RELEASED_EFFECTIVE_BATCH_SIZE
    ):
        raise ValueError(
            "effective batch size differs from release"
        )
    if recipe.get("weight_source") != "ema":
        raise ValueError(
            "released qualification requires EMA checkpoint weights"
        )
    ema = recipe.get("ema")
    if not isinstance(ema, dict):
        raise ValueError(
            "released qualification requires EMA evidence"
        )
    if ema.get("enabled") is not True:
        raise ValueError(
            "released qualification requires EMA enabled"
        )
    decay = ema.get("decay")
    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or not math.isclose(
            float(decay),
            RELEASED_EMA_DECAY,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "released qualification requires ema_decay=0.9998"
        )
    updates = ema.get("updates")
    if (
        isinstance(updates, bool)
        or not isinstance(updates, int)
        or updates <= 0
    ):
        raise ValueError(
            "released qualification requires positive EMA update count"
        )
    require_sha256(
        ema.get("state_sha256"),
        "EMA state_sha256",
    )
    require_sha256(
        ema.get("raw_state_sha256"),
        "raw state_sha256",
    )
    if ema["state_sha256"] == ema["raw_state_sha256"]:
        raise ValueError(
            "released EMA state must differ from raw training state"
        )

    augmentation = recipe.get("augmentation")
    if not isinstance(augmentation, dict):
        raise ValueError(
            "released qualification requires augmentation evidence"
        )
    if augmentation.get("kind") != "brightness-contrast-saturation":
        raise ValueError(
            "released qualification requires photometric augmentation contract"
        )
    strength = augmentation.get("strength")
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isclose(
            float(strength),
            RELEASED_PHOTOMETRIC_AUGMENT,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "released qualification requires augment=0.3"
        )
    if (
        augmentation.get("horizontal_flip") is not False
        or augmentation.get("geometry_transform") is not False
    ):
        raise ValueError(
            "released augmentation must preserve geometry and direction"
        )
    if augmentation.get("rng_source") != "ambient-torch-rng":
        raise ValueError(
            "released augmentation RNG source differs from Apache contract"
        )
    if augmentation.get("rng_equivalence") != "stochastic-distribution":
        raise ValueError(
            "released augmentation must claim stochastic-distribution equivalence"
        )

    precision = recipe.get("precision")
    if not isinstance(precision, dict):
        raise ValueError(
            "released qualification requires precision evidence"
        )
    if precision.get("amp_requested") is not True:
        raise ValueError(
            "released qualification requires AMP enabled"
        )
    if precision.get("amp_dtype") != RELEASED_AMP_DTYPE:
        raise ValueError(
            "released qualification requires BF16 AMP"
        )
    if precision.get("autocast_device_type") != "cuda":
        raise ValueError(
            "released qualification requires CUDA autocast"
        )
    if precision.get("autocast_executed") is not True:
        raise ValueError(
            "released qualification requires executed autocast"
        )
    if precision.get("grad_scaler") is not False:
        raise ValueError(
            "released BF16 contract must not use GradScaler"
        )
    if precision.get("released_cuda_execution") is not True:
        raise ValueError(
            "released qualification requires CUDA BF16 execution"
        )
    device = payload.get("device")
    if (
        not isinstance(device, str)
        or not device.startswith("cuda")
    ):
        raise ValueError(
            "released qualification requires a CUDA training device"
        )

    sampler_budget = recipe.get("sampler_budget")
    if not isinstance(sampler_budget, dict):
        raise ValueError(
            "released qualification requires sampler budget evidence"
        )
    if (
        sampler_budget.get("geo_budget") != RELEASED_GEO_BUDGET
        or sampler_budget.get("final_budget") != RELEASED_FINAL_BUDGET
        or sampler_budget.get("matches_released") is not True
    ):
        raise ValueError(
            "released qualification requires sampler budget 400->128"
        )

    mixture = payload.get("train_mixture")
    if not isinstance(mixture, dict):
        raise ValueError(
            "training mixture evidence is missing"
        )
    if mixture.get("source_names") != list(
        RELEASED_SOURCE_NAMES
    ):
        raise ValueError(
            "training source order differs from release"
        )
    _require_released_fractions(
        mixture.get("target_fractions")
    )
    if mixture.get("seed") != RELEASED_SEED:
        raise ValueError(
            "training mixture seed differs from release"
        )
    if (
        mixture.get("draws_per_epoch")
        != RELEASED_SAMPLES_PER_EPOCH
    ):
        raise ValueError(
            "training draw count differs from release"
        )
    if (
        mixture.get("matches_released_sampling_stream")
        is not True
    ):
        raise ValueError(
            "released sampling stream was not qualified"
        )

    history = payload.get("history")
    if (
        not isinstance(history, list)
        or len(history) != RELEASED_EPOCHS
    ):
        raise ValueError(
            "training history must contain all 12 released epochs"
        )
    expected_epochs = list(
        range(1, RELEASED_EPOCHS + 1)
    )
    actual_epochs = [
        row.get("epoch")
        if isinstance(row, dict)
        else None
        for row in history
    ]
    if actual_epochs != expected_epochs:
        raise ValueError(
            "training history epoch sequence is incomplete"
        )

    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(
            "model config evidence is missing"
        )
    if model_config.get("image_size") != RELEASED_IMAGE_SIZE:
        raise ValueError(
            "Apache released qualification requires image_size=448"
        )
    if model_config.get("max_boxes") != RELEASED_MAX_BOXES:
        raise ValueError(
            "Apache released qualification requires max_boxes=40"
        )
    if model_config.get("pair_budget") != RELEASED_FINAL_BUDGET:
        raise ValueError(
            "Apache released qualification requires pair_budget=128"
        )
    if model_config.get("hidden_dim") != RELEASED_D_MODEL:
        raise ValueError(
            "Apache released qualification requires hidden_dim=512"
        )
    for key in (
        "pair_evidence_contract",
        "pair_sampler_contract",
        "relation_context_contract",
        "predicate_head_contract",
    ):
        if model_config.get(key) != "apache":
            raise ValueError(
                f"{key} is not the Apache contract"
            )

    require_sha256(
        payload.get("checkpoint_sha256"),
        "checkpoint_sha256",
    )
    predicate_shape = payload.get(
        "predicate_embedding_shape"
    )
    if (
        not isinstance(predicate_shape, list)
        or len(predicate_shape) != 2
        or predicate_shape[1] != RELEASED_TEXT_DIM
    ):
        raise ValueError(
            "released qualification requires predicate text_dim=512"
        )
    require_sha256(
        payload.get("predicate_embeddings_sha256"),
        "predicate_embeddings_sha256",
    )
    require_sha256(
        payload.get("vocabulary_sha256"),
        "vocabulary_sha256",
    )

    objective = payload.get(
        "apache_reference_objective"
    )
    if not isinstance(objective, dict):
        raise ValueError(
            "Apache reference objective evidence is missing"
        )
    objective_config = objective.get("config")
    if not isinstance(objective_config, dict):
        raise ValueError(
            "Apache reference objective config is missing"
        )
    scalar_contract = (
        validate_released_scalar_contract(
            recipe=config,
            model=model_config,
            objective=objective_config,
        )
    )
    if recipe.get("scalar_contract") != scalar_contract:
        raise ValueError(
            "released scalar-contract evidence does not match resolved config"
        )
    structure_contract = recipe.get(
        "structure_contract"
    )
    if not isinstance(structure_contract, dict):
        raise ValueError(
            "released structure-contract evidence is missing"
        )
    validate_released_structure_report(
        structure_contract
    )

    assets = objective.get("assets")
    if not isinstance(assets, dict):
        raise ValueError(
            "Apache objective asset evidence is missing"
        )
    for key in (
        "source_column_allow_sha256",
        "ontology_meta_sha256",
        "ontology_npz_sha256",
        "neg_rate_table_sha256",
        "object_embeddings_sha256",
    ):
        require_sha256(
            assets.get(key),
            key,
        )

    if assets.get("source_names") != list(
        RELEASED_SOURCE_NAMES
    ):
        raise ValueError(
            "released source-column table source order differs from release"
        )

    object_shape = assets.get(
        "object_embeddings_shape"
    )
    object_order = assets.get(
        "object_label_order"
    )
    if (
        not isinstance(object_shape, list)
        or len(object_shape) != 2
        or object_shape[1] != RELEASED_TEXT_DIM
        or not isinstance(object_shape[0], int)
        or isinstance(object_shape[0], bool)
        or object_shape[0] <= 0
    ):
        raise ValueError(
            "released qualification requires object embeddings [O,512]"
        )
    if (
        not isinstance(object_order, list)
        or len(object_order) != object_shape[0]
        or any(
            not isinstance(name, str) or not name
            for name in object_order
        )
        or len(set(object_order)) != len(object_order)
    ):
        raise ValueError(
            "released qualification requires exact object-label order evidence"
        )
    object_bank = assets.get("object_bank")
    if (
        not isinstance(object_bank, dict)
        or object_bank.get("schema")
        != "kfcore.apache-object-text-bank/1"
        or object_bank.get("artifact_sha256")
        != assets["object_embeddings_sha256"]
        or object_bank.get("shape")
        != object_shape
        or object_bank.get("object_label_order")
        != object_order
        or object_bank.get("text_dim")
        != RELEASED_TEXT_DIM
    ):
        raise ValueError(
            "released qualification requires named object-bank provenance"
        )

    source_hashes = mixture.get(
        "source_annotation_sha256"
    )
    source_counts = mixture.get("source_counts")
    if (
        not isinstance(source_hashes, list)
        or len(source_hashes)
        != len(RELEASED_SOURCE_NAMES)
    ):
        raise ValueError(
            "source annotation hashes are missing"
        )
    if (
        not isinstance(source_counts, list)
        or len(source_counts)
        != len(RELEASED_SOURCE_NAMES)
    ):
        raise ValueError(
            "source post-exclusion counts are missing"
        )
    for index, value in enumerate(source_hashes):
        require_sha256(
            value,
            f"source {index} annotation SHA-256",
        )
    for index, value in enumerate(source_counts):
        _require_positive_int(
            value,
            f"source {index} count",
        )
    require_sha256(
        mixture.get("exclude_ids_sha256"),
        "training exclude_ids_sha256",
    )

    routing = payload.get(
        "routing_warm_start"
    )
    if not isinstance(routing, dict):
        raise ValueError(
            "released qualification requires routing warm-start evidence"
        )
    if (
        routing.get("schema")
        != "kfcore.apache-routing-warm-start/1"
    ):
        raise ValueError(
            "unsupported routing warm-start schema"
        )
    require_sha256(
        routing.get(
            "spatial_flags_sha256"
        ),
        "routing spatial_flags_sha256",
    )
    spatial_count = _require_positive_int(
        routing.get("spatial_count"),
        "routing spatial_count",
    )
    semantic_count = _require_positive_int(
        routing.get("semantic_count"),
        "routing semantic_count",
    )
    if (
        spatial_count
        + semantic_count
        != predicate_shape[0]
    ):
        raise ValueError(
            "routing spatial/semantic counts do not match predicate vocabulary"
        )
    for key in (
        "target_alpha_mean",
        "target_alpha_spatial_mean",
        "target_alpha_semantic_mean",
        "mse_before",
        "mse_after",
        "reported_final_mse",
    ):
        value = _finite_number(
            routing.get(key),
            f"routing {key}",
        )
        if key.startswith("target_alpha") and not (
            0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"routing {key} must be within [0,1]"
            )
        if key.startswith("mse_") or key == "reported_final_mse":
            if value < 0.0:
                raise ValueError(
                    f"routing {key} must be non-negative"
                )
    if not (
        float(routing["mse_after"])
        < float(routing["mse_before"])
    ):
        raise ValueError(
            "routing warm start must improve MSE"
        )

    return dict(payload)


def qualify_training_run(
    corpus: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    corpus = validate_corpus(corpus)
    training = validate_training_run(training)

    missing = tuple(
        source["name"]
        for source in corpus["sources"]
        if not source["resolved"]
    )
    if missing:
        raise ValueError(
            "released corpus provenance is unresolved for: "
            + ", ".join(missing)
        )

    mixture = training["train_mixture"]
    expected_hashes = [
        source["annotations_sha256"]
        for source in corpus["sources"]
    ]
    expected_counts = [
        source["post_exclusion_count"]
        for source in corpus["sources"]
    ]
    if (
        mixture["source_annotation_sha256"]
        != expected_hashes
    ):
        raise ValueError(
            "training source annotation hashes do not match corpus manifest"
        )
    if mixture["source_counts"] != expected_counts:
        raise ValueError(
            "training source counts do not match corpus manifest"
        )
    if (
        mixture["exclude_ids_sha256"]
        != corpus["exclude_ids_sha256"]
    ):
        raise ValueError(
            "training exclusion artifact does not match corpus manifest"
        )

    objective_assets = training[
        "apache_reference_objective"
    ]["assets"]
    asset_pairs = (
        (
            "source_column_allow_sha256",
            "source_column_allow_sha256",
        ),
        (
            "ontology_meta_sha256",
            "ontology_meta_sha256",
        ),
        (
            "ontology_npz_sha256",
            "ontology_npz_sha256",
        ),
        (
            "neg_rate_table_sha256",
            "neg_rate_table_sha256",
        ),
        (
            "object_embeddings_sha256",
            "object_embeddings_sha256",
        ),
    )
    for training_key, corpus_key in asset_pairs:
        if (
            objective_assets[training_key]
            != corpus[corpus_key]
        ):
            raise ValueError(
                f"{training_key} does not match corpus manifest"
            )

    if (
        training["predicate_embeddings_sha256"]
        != corpus["predicate_embeddings_sha256"]
    ):
        raise ValueError(
            "predicate embedding asset does not match corpus manifest"
        )
    if (
        training["vocabulary_sha256"]
        != corpus["vocabulary_sha256"]
    ):
        raise ValueError(
            "vocabulary asset does not match corpus manifest"
        )
    routing = training[
        "routing_warm_start"
    ]
    if (
        routing["spatial_flags_sha256"]
        != corpus[
            "predicate_spatial_flags_sha256"
        ]
    ):
        raise ValueError(
            "predicate spatial-flags hash does not match corpus manifest"
        )
    derivation = corpus[
        "predicate_spatial_flags_derivation"
    ]
    predicate_count = training[
        "predicate_embedding_shape"
    ][0]
    if (
        derivation["union_predicate_count"]
        != predicate_count
    ):
        raise ValueError(
            "predicate spatial derivation vocabulary size does not match training"
        )
    if (
        derivation[
            "spatial_union_predicate_count"
        ]
        != routing["spatial_count"]
    ):
        raise ValueError(
            "predicate spatial derivation count does not match routing evidence"
        )
    if (
        predicate_count
        - derivation[
            "spatial_union_predicate_count"
        ]
        != routing["semantic_count"]
    ):
        raise ValueError(
            "predicate semantic derivation count does not match routing evidence"
        )

    return {
        "schema": QUALIFICATION_SCHEMA,
        "qualified": True,
        "reference_source_commit": (
            REFERENCE_SOURCE_COMMIT
        ),
        "corpus_sha256": payload_sha256(corpus),
        "training_sha256": payload_sha256(training),
        "checkpoint_sha256": training[
            "checkpoint_sha256"
        ],
        "source_names": list(
            RELEASED_SOURCE_NAMES
        ),
        "source_counts": expected_counts,
        "source_annotation_sha256": (
            expected_hashes
        ),
        "exclude_ids_sha256": corpus[
            "exclude_ids_sha256"
        ],
        "epochs": RELEASED_EPOCHS,
        "draws_per_epoch": (
            RELEASED_SAMPLES_PER_EPOCH
        ),
        "effective_batch_size": (
            RELEASED_EFFECTIVE_BATCH_SIZE
        ),
        "weight_source": "ema",
        "ema_decay": RELEASED_EMA_DECAY,
        "ema_updates": training[
            "training_recipe"
        ]["ema"]["updates"],
        "ema_state_sha256": training[
            "training_recipe"
        ]["ema"]["state_sha256"],
        "augmentation": training[
            "training_recipe"
        ]["augmentation"],
        "text_dim": RELEASED_TEXT_DIM,
        "image_size": RELEASED_IMAGE_SIZE,
        "geo_budget": RELEASED_GEO_BUDGET,
        "final_budget": RELEASED_FINAL_BUDGET,
        "object_embeddings_sha256": corpus[
            "object_embeddings_sha256"
        ],
        "object_embeddings_shape": training[
            "apache_reference_objective"
        ]["assets"]["object_embeddings_shape"],
        "predicate_spatial_flags_sha256": corpus[
            "predicate_spatial_flags_sha256"
        ],
        "predicate_spatial_flags_derivation": corpus[
            "predicate_spatial_flags_derivation"
        ],
        "source_column_allow_sha256": corpus[
            "source_column_allow_sha256"
        ],
        "source_column_allow_derivation": corpus[
            "source_column_allow_derivation"
        ],
        "precision": training[
            "training_recipe"
        ]["precision"],
        "scalar_contract": training[
            "training_recipe"
        ]["scalar_contract"],
        "structure_contract": training[
            "training_recipe"
        ]["structure_contract"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    corpus = load_json(args.corpus)
    training = load_json(args.training)
    result = qualify_training_run(
        corpus,
        training,
    )
    Path(args.out).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
