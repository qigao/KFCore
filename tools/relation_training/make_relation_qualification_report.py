from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CONTEXT_SCHEMA = "kfcore.relation-qualification-context/1"
LATENCY_SCHEMA = "kfcore.scene-behavior-latency/1"
REPORT_SCHEMA = "kfcore.relation-qualification-report/1"
QUALITY_SCHEMA = "kfcore.detector-relation-ceiling/1"


def stable_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def validate_context(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != CONTEXT_SCHEMA:
        raise ValueError("unsupported qualification context schema")
    result = dict(payload)
    for key in (
        "relation_model_sha256",
        "vocabulary_sha256",
        "relation_config_sha256",
        "detector_model_sha256",
        "detector_config_sha256",
    ):
        result[key] = require_sha256(payload.get(key), key)
    for key in (
        "backend",
        "device",
        "relation_model_type",
        "detector_id",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"context {key} must be non-empty")
        result[key] = value
    for key in ("max_boxes", "vocabulary_size"):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"context {key} must be positive")
        result[key] = value
    return result


def context_digest(context: dict[str, Any]) -> str:
    normalized = validate_context(context)
    return hashlib.sha256(stable_json_bytes(normalized)).hexdigest()


def validate_latency(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != LATENCY_SCHEMA:
        raise ValueError("unsupported latency schema")
    require_sha256(
        payload.get("context_sha256"),
        "latency context_sha256",
    )
    samples = payload.get("samples")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples <= 0
    ):
        raise ValueError("latency samples must be positive")
    stages = payload.get("stages_ms")
    if not isinstance(stages, dict):
        raise ValueError("latency stages_ms must be an object")
    for stage in (
        "detector",
        "tracker",
        "region_prepare",
        "relation",
        "assembly",
        "temporal",
        "total",
    ):
        stats = stages.get(stage)
        if not isinstance(stats, dict):
            raise ValueError(f"missing latency stage: {stage}")
        previous = -1.0
        for key in ("p50", "p90", "p95", "p99"):
            value = stats.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < previous
            ):
                raise ValueError(
                    f"latency percentiles invalid for {stage}.{key}"
                )
            previous = float(value)
        mean = stats.get("mean")
        if (
            isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or not math.isfinite(float(mean))
            or float(mean) < 0.0
        ):
            raise ValueError(f"latency mean invalid for {stage}")
    cardinality = payload.get("cardinality")
    if not isinstance(cardinality, dict):
        raise ValueError("latency cardinality must be an object")
    for key in (
        "detections_mean",
        "tracked_objects_mean",
        "relation_edges_mean",
        "events_mean",
        "pair_states_mean",
    ):
        value = cardinality.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"invalid latency cardinality: {key}")
    return dict(payload)


def validate_quality(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != QUALITY_SCHEMA:
        raise ValueError("unsupported detector-quality schema")
    detector = payload.get("detector")
    if not isinstance(detector, dict):
        raise ValueError("quality detector block is missing")
    examples = payload.get("examples")
    if (
        isinstance(examples, bool)
        or not isinstance(examples, int)
        or examples <= 0
    ):
        raise ValueError("quality examples must be positive")
    for key in (
        "object_recoverability_ceiling",
        "directed_pair_recoverability_ceiling",
        "sampler_recall_conditional_on_recoverable_pairs",
        "sampler_pair_recall_end_to_end",
    ):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"quality metric out of range: {key}")
    decomp = payload.get("failure_decomposition_at_max_k")
    if not isinstance(decomp, dict):
        raise ValueError("quality failure decomposition is missing")
    pieces = tuple(
        int(decomp.get(key, -1))
        for key in (
            "detector_miss",
            "sampler_miss",
            "predicate_miss",
            "recovered",
        )
    )
    if any(value < 0 for value in pieces):
        raise ValueError("quality failure decomposition is invalid")
    if sum(pieces) != int(payload.get("ground_truth_triplets", -1)):
        raise ValueError(
            "quality failure decomposition does not sum to GT triplets"
        )
    return dict(payload)


def make_report(
    context: dict[str, Any],
    quality: dict[str, Any],
    latency: dict[str, Any],
) -> dict[str, Any]:
    context = validate_context(context)
    quality = validate_quality(quality)
    latency = validate_latency(latency)
    digest = context_digest(context)
    if latency["context_sha256"] != digest:
        raise ValueError(
            "latency artifact uses a different qualification context"
        )
    detector = quality["detector"]
    if detector.get("id") != context["detector_id"]:
        raise ValueError("quality detector id does not match context")
    if (
        detector.get("model_sha256")
        != context["detector_model_sha256"]
    ):
        raise ValueError(
            "quality detector model does not match context"
        )
    if (
        detector.get("config_sha256")
        != context["detector_config_sha256"]
    ):
        raise ValueError(
            "quality detector config does not match context"
        )
    return {
        "schema": REPORT_SCHEMA,
        "context_sha256": digest,
        "context": context,
        "quality": quality,
        "latency": latency,
    }


def render_markdown(report: dict[str, Any]) -> str:
    context = report["context"]
    quality = report["quality"]
    latency = report["latency"]
    stages = latency["stages_ms"]
    failure = quality["failure_decomposition_at_max_k"]
    lines = [
        "# KFCore Relation Qualification",
        "",
        "Context SHA-256: " + report["context_sha256"],
        "Backend/device: "
        + context["backend"]
        + " / "
        + context["device"],
        "Relation model: " + context["relation_model_sha256"],
        "Detector: "
        + context["detector_id"]
        + " / "
        + context["detector_model_sha256"],
        "Vocabulary size: " + str(context["vocabulary_size"]),
        "Vocabulary SHA-256: " + context["vocabulary_sha256"],
        "",
        "## Detector / Relation Quality",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Object recoverability ceiling | "
        + f"{quality['object_recoverability_ceiling']:.6f} |",
        "| Directed pair recoverability ceiling | "
        + f"{quality['directed_pair_recoverability_ceiling']:.6f} |",
        "| Sampler recall conditional on recoverable pairs | "
        + f"{quality['sampler_recall_conditional_on_recoverable_pairs']:.6f} |",
        "| Sampler pair recall end-to-end | "
        + f"{quality['sampler_pair_recall_end_to_end']:.6f} |",
        "",
        "Failure decomposition: "
        + ", ".join(
            key + "=" + str(failure[key])
            for key in (
                "detector_miss",
                "sampler_miss",
                "predicate_miss",
                "recovered",
            )
        ),
        "",
        "## Latency",
        "",
        "Samples: " + str(latency["samples"]),
        "",
        "| Stage | p50 ms | p90 ms | p95 ms | p99 ms | mean ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in (
        "detector",
        "tracker",
        "region_prepare",
        "relation",
        "assembly",
        "temporal",
        "total",
    ):
        value = stages[stage]
        lines.append(
            "| "
            + stage
            + " | "
            + f"{value['p50']:.3f} | {value['p90']:.3f} | "
            + f"{value['p95']:.3f} | {value['p99']:.3f} | "
            + f"{value['mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--latency", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()
    report = make_report(
        load_json(args.context),
        load_json(args.quality),
        load_json(args.latency),
    )
    Path(args.out_json).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(args.out_md).write_text(
        render_markdown(report),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
