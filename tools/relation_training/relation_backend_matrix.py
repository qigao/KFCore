from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from make_relation_qualification_report import (
    context_digest as release_context_digest,
    validate_context as validate_release_context,
    validate_latency as validate_release_latency,
    validate_quality as validate_release_quality,
)


MATRIX_SCHEMA = "kfcore.relation-backend-matrix/1"
PROVENANCE_SCHEMA = "kfcore.relation-backend-provenance/1"
ORT_CPU_SCHEMA = "kfcore.relation-ort-cpu-cpp-qualification/1"
TRT_SCHEMA = "kfcore.tensorrt-dynamic-vocab-qualification/1"
QUALITY_SCHEMA = "kfcore.detector-box-relation-qualification/1"
EXPORT_SCHEMA = "kfcore.relation-onnx/2"
RELEASE_REPORT_SCHEMA = "kfcore.relation-qualification-report/1"

_REQUIRED_CASES = (
    ("v1", 1),
    ("v3", 3),
    ("default", 4),
    ("large", 64),
    ("v1-repeat", 1),
)

_POLICY_KEYS = (
    "logit_scale",
    "logit_bias",
    "pair_weight",
    "calibration_a",
    "calibration_b",
    "threshold",
    "top_k",
    "weight_ranking_by_detector_score",
)


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def stable_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_policy(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("score_decode_policy must be an object")
    for key in _POLICY_KEYS:
        if key not in payload:
            raise ValueError(f"score_decode_policy is missing {key}")

    result: dict[str, object] = {
        "logit_scale": _number(
            payload["logit_scale"],
            "policy logit_scale",
        ),
        "logit_bias": _number(
            payload["logit_bias"],
            "policy logit_bias",
            minimum=-float("inf"),
        ),
        "pair_weight": _number(
            payload["pair_weight"],
            "policy pair_weight",
            minimum=-float("inf"),
        ),
        "calibration_a": _number(
            payload["calibration_a"],
            "policy calibration_a",
        ),
        "calibration_b": _number(
            payload["calibration_b"],
            "policy calibration_b",
            minimum=-float("inf"),
        ),
        "threshold": _number(
            payload["threshold"],
            "policy threshold",
        ),
        "top_k": _positive_int(
            payload["top_k"],
            "policy top_k",
        ),
    }
    if float(result["logit_scale"]) <= 0.0:
        raise ValueError("policy logit_scale must be positive")
    if float(result["calibration_a"]) <= 0.0:
        raise ValueError("policy calibration_a must be positive")
    if float(result["threshold"]) > 1.0:
        raise ValueError("policy threshold must be <= 1")

    detector_weight = payload["weight_ranking_by_detector_score"]
    if not isinstance(detector_weight, bool):
        raise ValueError(
            "weight_ranking_by_detector_score must be boolean"
        )
    result["weight_ranking_by_detector_score"] = detector_weight
    return result


def validate_export_pair(
    dynamic_export: Mapping[str, object],
    encoder_export: Mapping[str, object],
) -> dict[str, object]:
    for label, payload in (
        ("dynamic", dynamic_export),
        ("encoder", encoder_export),
    ):
        if payload.get("schema") != EXPORT_SCHEMA:
            raise ValueError(f"{label} export uses unsupported schema")

    if dynamic_export.get("model_type") != "relation.open-vocabulary":
        raise ValueError("dynamic export must be relation.open-vocabulary")
    if dynamic_export.get("output_kind") != "dynamic-vocabulary-logits":
        raise ValueError("dynamic export must emit dynamic-vocabulary-logits")
    if encoder_export.get("model_type") != "relation.open-vocabulary-encoder":
        raise ValueError(
            "encoder export must be relation.open-vocabulary-encoder"
        )
    if encoder_export.get("output_kind") != "relation-queries":
        raise ValueError("encoder export must emit relation queries")

    common = (
        "checkpoint_sha256",
        "backbone_model",
        "image_size",
        "max_boxes",
        "final_budget",
        "query_dim",
        "score_logit_scale",
        "score_logit_bias",
        "score_contract",
    )
    for key in common:
        if dynamic_export.get(key) != encoder_export.get(key):
            raise ValueError(
                f"dynamic/encoder export provenance differs for {key}"
            )

    return {
        "checkpoint_sha256": _sha(
            dynamic_export.get("checkpoint_sha256"),
            "export checkpoint_sha256",
        ),
        "dynamic_onnx_sha256": _sha(
            dynamic_export.get("onnx_sha256"),
            "dynamic onnx_sha256",
        ),
        "encoder_onnx_sha256": _sha(
            encoder_export.get("onnx_sha256"),
            "encoder onnx_sha256",
        ),
        "backbone_model": str(dynamic_export["backbone_model"]),
        "image_size": _positive_int(
            dynamic_export.get("image_size"),
            "export image_size",
        ),
        "max_boxes": _positive_int(
            dynamic_export.get("max_boxes"),
            "export max_boxes",
        ),
        "final_budget": _positive_int(
            dynamic_export.get("final_budget"),
            "export final_budget",
        ),
        "query_dim": _positive_int(
            dynamic_export.get("query_dim"),
            "export query_dim",
        ),
        "score_logit_scale": _number(
            dynamic_export.get("score_logit_scale"),
            "export score_logit_scale",
        ),
        "score_logit_bias": _number(
            dynamic_export.get("score_logit_bias"),
            "export score_logit_bias",
            minimum=-float("inf"),
        ),
        "score_contract": str(dynamic_export["score_contract"]),
    }


def validate_quality(
    payload: Mapping[str, object],
    lineage: Mapping[str, object],
) -> dict[str, object]:
    if payload.get("schema") != QUALITY_SCHEMA:
        raise ValueError("unsupported detector-box quality report schema")
    relation = payload.get("relation")
    if not isinstance(relation, dict):
        raise ValueError("quality report relation provenance is missing")
    if (
        relation.get("checkpoint_sha256")
        != lineage["checkpoint_sha256"]
    ):
        raise ValueError(
            "quality report and backend exports use different checkpoints"
        )
    for key in (
        "config_sha256",
        "vocabulary_sha256",
    ):
        _sha(relation.get(key), f"quality relation {key}")

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("quality report evaluation block is missing")
    pair_weight = _number(
        evaluation.get("pair_weight"),
        "quality pair_weight",
        minimum=-float("inf"),
    )

    quality = payload.get("quality")
    if not isinstance(quality, dict):
        raise ValueError("quality metrics block is missing")
    for key in (
        "object_recoverability_ceiling",
        "directed_pair_recoverability_ceiling",
        "sampler_recall_conditional_on_recoverable_pairs",
        "sampler_pair_recall_end_to_end",
    ):
        value = _number(quality.get(key), f"quality {key}")
        if value > 1.0:
            raise ValueError(f"quality {key} must be <= 1")

    return {
        "checkpoint_sha256": relation["checkpoint_sha256"],
        "config_sha256": relation["config_sha256"],
        "vocabulary_sha256": relation["vocabulary_sha256"],
        "pair_weight": pair_weight,
        "report": dict(payload),
    }


def validate_provenance(payload: Mapping[str, object]) -> dict[str, object]:
    if payload.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError("unsupported backend provenance schema")
    if payload.get("backend") != "onnxruntime-cpu":
        raise ValueError("ORT CPU provenance has wrong backend id")
    hardware = payload.get("hardware")
    software = payload.get("software")
    if not isinstance(hardware, dict) or not hardware:
        raise ValueError("backend hardware provenance is missing")
    if not isinstance(software, dict) or not software:
        raise ValueError("backend software provenance is missing")
    return {
        "hardware": dict(hardware),
        "software": dict(software),
    }


def _validate_pair_keys(value: object, name: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} pair_keys must be an array")
    result: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                for index in item
            )
        ):
            raise ValueError(f"{name} contains invalid pair key")
        pair = (item[0], item[1])
        if pair in seen:
            raise ValueError(f"{name} contains duplicate pair key")
        seen.add(pair)
        result.append([pair[0], pair[1]])
    return result


def _normalize_ort_case(
    case: Mapping[str, object],
    *,
    backend_scoring: bool,
) -> dict[str, object]:
    label = case.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("ORT case label must be non-empty")
    predicate_count = _positive_int(
        case.get("predicate_count"),
        f"{label} predicate_count",
    )
    region_count = _positive_int(
        case.get("region_count"),
        f"{label} region_count",
    )
    valid_pair_count = _positive_int(
        case.get("valid_pair_count"),
        f"{label} valid_pair_count",
    )
    pair_keys = _validate_pair_keys(
        case.get("pair_keys"),
        label,
    )
    if len(pair_keys) != int(case.get("edge_count", -1)):
        raise ValueError(f"{label} pair key count differs from edge count")

    timing = {
        "preprocess_ms": _number(
            case.get("preprocess_ms"),
            f"{label} preprocess_ms",
        ),
        "backbone_context_ms": (
            None
            if backend_scoring
            else _number(
                case.get("backbone_context_ms"),
                f"{label} backbone_context_ms",
            )
        ),
        "predicate_scoring_ms": (
            None
            if backend_scoring
            else _number(
                case.get("predicate_scoring_ms"),
                f"{label} predicate_scoring_ms",
            )
        ),
        "backend_runtime_ms": _number(
            case.get("backend_ms"),
            f"{label} backend_ms",
        ),
        "host_decode_ms": _number(
            case.get("decode_ms"),
            f"{label} decode_ms",
        ),
        "total_relation_ms": _number(
            case.get("total_ms"),
            f"{label} total_ms",
        ),
    }

    return {
        "label": label,
        "vocabulary_size": predicate_count,
        "image_size": None,
        "box_count": {
            "min": region_count,
            "max": region_count,
            "mean": float(region_count),
        },
        "selected_pair_count": valid_pair_count,
        "pair_keys": pair_keys,
        "timing": timing,
        "predicate_scoring_in_backend": backend_scoring,
    }


def normalize_ort_cpu(
    payload: Mapping[str, object],
    provenance: Mapping[str, object],
    lineage: Mapping[str, object],
    quality: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if payload.get("schema") != ORT_CPU_SCHEMA:
        raise ValueError("unsupported ORT CPU qualification schema")
    if payload.get("passed") is not True:
        raise ValueError("ORT CPU qualification is not passing")
    if payload.get("provider") != "onnxruntime" or payload.get("device") != "cpu":
        raise ValueError("ORT CPU report has wrong execution route")
    if payload.get("host_fallback_reference") is not True:
        raise ValueError("ORT CPU report lacks host fallback reference")
    if payload.get("backend_host_pair_keyed_parity") is not True:
        raise ValueError("ORT CPU report lacks backend/host pair parity")
    if payload.get("model_sha256") != lineage["dynamic_onnx_sha256"]:
        raise ValueError("ORT CPU dynamic model does not match export")
    if payload.get("host_model_sha256") != lineage["encoder_onnx_sha256"]:
        raise ValueError("ORT CPU host model does not match encoder export")

    policy = validate_policy(payload.get("score_decode_policy"))
    if (
        abs(float(policy["logit_scale"]) - float(lineage["score_logit_scale"]))
        > 1.0e-6
        or abs(float(policy["logit_bias"]) - float(lineage["score_logit_bias"]))
        > 1.0e-6
    ):
        raise ValueError("ORT CPU score policy differs from export")
    if abs(float(policy["pair_weight"]) - float(quality["pair_weight"])) > 1.0e-6:
        raise ValueError("ORT CPU pair weight differs from quality report")

    cases_raw = payload.get("cases")
    host_raw = payload.get("host_cases")
    if not isinstance(cases_raw, list) or not isinstance(host_raw, list):
        raise ValueError("ORT CPU backend/host cases are missing")
    if len(cases_raw) != len(_REQUIRED_CASES) or len(host_raw) != len(_REQUIRED_CASES):
        raise ValueError("ORT CPU qualification has wrong case count")

    backend_cases: list[dict[str, object]] = []
    host_cases: list[dict[str, object]] = []
    for index, (label, size) in enumerate(_REQUIRED_CASES):
        backend_value = cases_raw[index]
        host_value = host_raw[index]
        if not isinstance(backend_value, dict) or not isinstance(host_value, dict):
            raise ValueError("ORT CPU case must be an object")
        backend_case = _normalize_ort_case(
            backend_value,
            backend_scoring=True,
        )
        host_case = _normalize_ort_case(
            host_value,
            backend_scoring=False,
        )
        if (
            backend_case["label"] != label
            or host_case["label"] != label
            or backend_case["vocabulary_size"] != size
            or host_case["vocabulary_size"] != size
        ):
            raise ValueError(f"ORT CPU case order/size differs at {label}")
        if backend_case["pair_keys"] != host_case["pair_keys"]:
            raise ValueError(f"ORT CPU pair keys differ at {label}")
        backend_case["image_size"] = lineage["image_size"]
        host_case["image_size"] = lineage["image_size"]
        backend_cases.append(backend_case)
        host_cases.append(host_case)

    if (
        backend_cases[0]["pair_keys"] != backend_cases[-1]["pair_keys"]
        or host_cases[0]["pair_keys"] != host_cases[-1]["pair_keys"]
    ):
        raise ValueError("repeated V=1 changed pair-key semantics")

    prov = validate_provenance(provenance)
    common = {
        "score_decode_policy": policy,
        "hardware": prov["hardware"],
        "software": prov["software"],
    }
    backend = {
        "id": "onnxruntime-cpu",
        "kind": "backend-scoring",
        **common,
        "cases": backend_cases,
    }
    host = {
        "id": "host-query-scorer-reference",
        "kind": "host-scoring-reference",
        **common,
        "cases": host_cases,
    }
    return backend, host


def normalize_released_deployment(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if payload.get("schema") != RELEASE_REPORT_SCHEMA:
        raise ValueError(
            "unsupported released deployment qualification schema"
        )

    raw_context = payload.get("context")
    raw_quality = payload.get("quality")
    raw_latency = payload.get("latency")
    if not isinstance(raw_context, dict):
        raise ValueError("released deployment context is missing")
    if not isinstance(raw_quality, dict):
        raise ValueError("released deployment quality is missing")
    if not isinstance(raw_latency, dict):
        raise ValueError("released deployment latency is missing")

    context = validate_release_context(raw_context)
    quality = validate_release_quality(raw_quality)
    latency = validate_release_latency(raw_latency)
    digest = release_context_digest(context)
    if payload.get("context_sha256") != digest:
        raise ValueError("released deployment report context digest differs")
    if latency.get("context_sha256") != digest:
        raise ValueError("released deployment latency context digest differs")

    if context.get("backend") != "onnxruntime":
        raise ValueError("released deployment backend must be onnxruntime")
    if context.get("device") != "cpu":
        raise ValueError("released deployment device must be cpu")
    if context.get("relation_model_type") != "relation.open-vocabulary":
        raise ValueError(
            "released deployment relation model type is unsupported"
        )
    if context.get("quality_input_contract") != "detector-boxes":
        raise ValueError(
            "released deployment must use detector-box quality evidence"
        )

    for key in (
        "predicate_bank_sha256",
        "detector_predictions_sha256",
        "annotations_sha256",
        "image_corpus_sha256",
        "dataset_source_parquet_sha256",
    ):
        _sha(context.get(key), f"released context {key}")

    pair_weight = _number(
        context.get("pair_weight"),
        "released context pair_weight",
        minimum=-float("inf"),
    )
    top_ks = context.get("top_ks")
    if (
        not isinstance(top_ks, list)
        or not top_ks
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in top_ks
        )
        or top_ks != sorted(set(top_ks))
    ):
        raise ValueError("released context top_ks must be sorted positive ints")

    quality_iou = _number(
        context.get("quality_iou_threshold"),
        "released context quality_iou_threshold",
    )
    if quality_iou <= 0.0 or quality_iou > 1.0:
        raise ValueError(
            "released context quality_iou_threshold must be within (0,1]"
        )

    detector = quality.get("detector")
    if not isinstance(detector, dict):
        raise ValueError("released quality detector block is missing")
    if detector.get("id") != context["detector_id"]:
        raise ValueError("released quality detector id differs from context")
    if detector.get("model_sha256") != context["detector_model_sha256"]:
        raise ValueError(
            "released quality detector model differs from context"
        )
    if detector.get("config_sha256") != context["detector_config_sha256"]:
        raise ValueError(
            "released quality detector config differs from context"
        )
    if (
        quality.get("detector_predictions_sha256")
        != context["detector_predictions_sha256"]
    ):
        raise ValueError(
            "released quality detector predictions differ from context"
        )
    if quality.get("annotations_sha256") != context["annotations_sha256"]:
        raise ValueError(
            "released quality annotations differ from context"
        )
    if abs(
        float(detector.get("iou_threshold", -1.0))
        - quality_iou
    ) > 1.0e-9:
        raise ValueError(
            "released quality IoU threshold differs from context"
        )

    decomp = quality.get("failure_decomposition_at_max_k")
    if not isinstance(decomp, dict):
        raise ValueError("released quality failure decomposition is missing")
    if decomp.get("max_k") != top_ks[-1]:
        raise ValueError(
            "released quality max-K differs from context top_ks"
        )

    runtime_provenance = context.get("runtime_provenance")
    if not isinstance(runtime_provenance, dict):
        raise ValueError("released runtime provenance is missing")
    hardware = runtime_provenance.get("hardware")
    software = runtime_provenance.get("software")
    if not isinstance(hardware, dict) or not hardware:
        raise ValueError("released hardware provenance is missing")
    if not isinstance(software, dict) or not software:
        raise ValueError("released software provenance is missing")
    python_ort = software.get("onnxruntime_python")
    sdk_ort = software.get("onnxruntime_sdk")
    if (
        not isinstance(python_ort, str)
        or not python_ort
        or python_ort != sdk_ort
    ):
        raise ValueError(
            "released Python and C++ ONNX Runtime versions differ"
        )

    return {
        "id": "upstream-released-onnxruntime-cpu",
        "kind": "measured-full-production-deployment",
        "lineage": {
            "relation_model_sha256": context["relation_model_sha256"],
            "relation_config_sha256": context["relation_config_sha256"],
            "vocabulary_sha256": context["vocabulary_sha256"],
            "predicate_bank_sha256": context["predicate_bank_sha256"],
            "detector_model_sha256": context["detector_model_sha256"],
            "detector_config_sha256": context["detector_config_sha256"],
            "detector_predictions_sha256": context[
                "detector_predictions_sha256"
            ],
            "annotations_sha256": context["annotations_sha256"],
            "image_corpus_sha256": context["image_corpus_sha256"],
            "dataset_revision": context.get("dataset_revision"),
            "dataset_source_parquet_sha256": context[
                "dataset_source_parquet_sha256"
            ],
        },
        "score_decode_policy": {
            "pair_weight": pair_weight,
            "top_ks": list(top_ks),
            "quality_iou_threshold": quality_iou,
            "relation_config_sha256": context[
                "relation_config_sha256"
            ],
        },
        "quality": quality,
        "latency": latency,
        "runtime_provenance": runtime_provenance,
        "context_sha256": digest,
        "boundary": {
            "host_query_scorer_reference_available": False,
            "encoder_query_export_available": False,
            "reason": (
                "upstream released artifact publishes the dynamic-logit "
                "deployment graph but no paired encoder/query export"
            ),
            "training_reproduction_complete": False,
        },
    }


def make_released_deployment_matrix(
    released_deployment_report: Mapping[str, object],
) -> dict[str, object]:
    deployment = normalize_released_deployment(
        released_deployment_report
    )
    return {
        "schema": MATRIX_SCHEMA,
        "mode": "upstream-released-deployment",
        "lineage": deployment["lineage"],
        "quality": deployment["quality"],
        "score_decode_policy": deployment["score_decode_policy"],
        "backends": [deployment],
        "optional_engine_qualifications": [],
        "availability": {
            "onnxruntime_cpu": True,
            "host_query_scorer_reference": False,
            "onnxruntime_cuda": False,
            "tensorrt_full_relation": False,
            "tensorrt_engine_proof": False,
        },
        "acceptance": {
            "quality_and_latency_together": True,
            "score_decode_policy_shared": True,
            "hardware_software_provenance_explicit": True,
            "released_context_digest_valid": True,
            "baseline_pair_keyed_output_parity_applicable": False,
        },
        "boundary": deployment["boundary"],
    }


def normalize_tensorrt_engine(
    payload: Mapping[str, object],
    lineage: Mapping[str, object],
) -> dict[str, object]:
    if payload.get("schema") != TRT_SCHEMA:
        raise ValueError("unsupported TensorRT qualification schema")
    for key in (
        "hardware_executed",
        "same_engine_reused",
        "same_context_reused",
        "passed",
    ):
        if payload.get(key) is not True:
            raise ValueError(f"TensorRT report does not satisfy {key}")
    if payload.get("source_onnx_sha256") != lineage["dynamic_onnx_sha256"]:
        raise ValueError("TensorRT source ONNX differs from matrix export")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("TensorRT qualification cases are missing")
    normalized: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("TensorRT case must be an object")
        if case.get("passed") is not True or case.get("pair_set_equal") is not True:
            raise ValueError(f"TensorRT parity failed at case {index}")
        latency = case.get("tensorrt_latency")
        if not isinstance(latency, dict):
            raise ValueError("TensorRT latency summary is missing")
        normalized.append({
            "label": str(case.get("label")),
            "vocabulary_size": _positive_int(
                case.get("vocabulary_size"),
                "TensorRT vocabulary_size",
            ),
            "pair_set_equal": True,
            "reference_valid_pairs": _positive_int(
                case.get("reference_valid_pairs"),
                "TensorRT reference_valid_pairs",
            ),
            "actual_valid_pairs": _positive_int(
                case.get("actual_valid_pairs"),
                "TensorRT actual_valid_pairs",
            ),
            "engine_latency_ms": {
                key: _number(latency.get(key), f"TensorRT {key}")
                for key in (
                    "mean_ms",
                    "median_ms",
                    "p95_ms",
                    "min_ms",
                    "max_ms",
                )
            },
        })

    hardware = payload.get("hardware")
    if not isinstance(hardware, dict) or not hardware:
        raise ValueError("TensorRT hardware/software provenance is missing")
    return {
        "id": "tensorrt-engine",
        "measurement_scope": "engine-only",
        "full_relation_stage_latency": False,
        "same_engine_reused": True,
        "profile": payload.get("profile"),
        "hardware_software": dict(hardware),
        "cases": normalized,
    }


def make_matrix(
    *,
    quality_report: Mapping[str, object],
    dynamic_export: Mapping[str, object],
    encoder_export: Mapping[str, object],
    ort_cpu_report: Mapping[str, object],
    ort_cpu_provenance: Mapping[str, object],
    tensorrt_report: Mapping[str, object] | None = None,
    released_deployment_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    lineage = validate_export_pair(dynamic_export, encoder_export)
    quality = validate_quality(quality_report, lineage)
    ort_cpu, host = normalize_ort_cpu(
        ort_cpu_report,
        ort_cpu_provenance,
        lineage,
        quality,
    )

    optional: list[dict[str, object]] = []
    if tensorrt_report is not None:
        optional.append(
            normalize_tensorrt_engine(tensorrt_report, lineage)
        )

    measured_deployments: list[dict[str, object]] = []
    if released_deployment_report is not None:
        measured_deployments.append(
            normalize_released_deployment(
                released_deployment_report
            )
        )

    return {
        "schema": MATRIX_SCHEMA,
        "lineage": {
            **lineage,
            "relation_config_sha256": quality["config_sha256"],
            "vocabulary_sha256": quality["vocabulary_sha256"],
        },
        "quality": quality["report"],
        "score_decode_policy": ort_cpu["score_decode_policy"],
        "backends": [
            host,
            ort_cpu,
        ],
        "optional_engine_qualifications": optional,
        "measured_deployments": measured_deployments,
        "availability": {
            "onnxruntime_cpu": True,
            "host_query_scorer_reference": True,
            "onnxruntime_cuda": False,
            "tensorrt_full_relation": False,
            "tensorrt_engine_proof": bool(optional),
        },
        "acceptance": {
            "quality_and_latency_together": True,
            "score_decode_policy_shared": True,
            "pair_keyed_output_parity": True,
            "hardware_software_provenance_explicit": True,
            "released_deployment_attachments_valid": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic KFCore relation backend matrix from "
            "either the KFCore paired-export parity lineage or a measured "
            "upstream released deployment qualification."
        )
    )
    parser.add_argument("--quality")
    parser.add_argument("--dynamic-export")
    parser.add_argument("--encoder-export")
    parser.add_argument("--ort-cpu")
    parser.add_argument("--ort-cpu-provenance")
    parser.add_argument("--tensorrt")
    parser.add_argument("--released-deployment")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline_args = (
        args.quality,
        args.dynamic_export,
        args.encoder_export,
        args.ort_cpu,
        args.ort_cpu_provenance,
    )
    baseline_selected = any(value is not None for value in baseline_args)
    if baseline_selected and not all(
        value is not None for value in baseline_args
    ):
        parser.error(
            "paired-export matrix mode requires --quality, "
            "--dynamic-export, --encoder-export, --ort-cpu and "
            "--ort-cpu-provenance together"
        )
    if not baseline_selected and not args.released_deployment:
        parser.error(
            "select paired-export inputs or --released-deployment"
        )
    if args.tensorrt and not baseline_selected:
        parser.error(
            "--tensorrt requires the paired-export baseline lineage"
        )

    released = (
        load_json(args.released_deployment)
        if args.released_deployment
        else None
    )
    if baseline_selected:
        matrix = make_matrix(
            quality_report=load_json(args.quality),
            dynamic_export=load_json(args.dynamic_export),
            encoder_export=load_json(args.encoder_export),
            ort_cpu_report=load_json(args.ort_cpu),
            ort_cpu_provenance=load_json(args.ort_cpu_provenance),
            tensorrt_report=(
                load_json(args.tensorrt)
                if args.tensorrt
                else None
            ),
            released_deployment_report=released,
        )
    else:
        assert released is not None
        matrix = make_released_deployment_matrix(released)

    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(stable_json(matrix), encoding="utf-8")
    print(json.dumps(matrix, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
