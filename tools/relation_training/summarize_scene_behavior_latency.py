from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from make_relation_qualification_report import (
    LATENCY_SCHEMA,
    context_digest,
    load_json,
    validate_context,
)


SAMPLE_SCHEMA = "kfcore.scene-behavior-timing-sample/1"


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid timing JSON at line {line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"timing sample at line {line_number} must be an object"
            )
        samples.append(validate_sample(value, line_number=line_number))
    if not samples:
        raise ValueError("timing sample file is empty")
    return samples


def validate_sample(
    payload: dict[str, Any],
    *,
    line_number: int = 0,
) -> dict[str, Any]:
    prefix = (
        f"timing sample at line {line_number}"
        if line_number
        else "timing sample"
    )
    if payload.get("schema") != SAMPLE_SCHEMA:
        raise ValueError(f"{prefix} has unsupported schema")

    result = dict(payload)
    timing_keys = (
        "detector_ms",
        "tracker_ms",
        "region_prepare_ms",
        "relation_ms",
        "assembly_ms",
        "scene_graph_total_ms",
        "temporal_ms",
        "total_ms",
    )
    for key in timing_keys:
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"{prefix} {key} must be finite and non-negative"
            )
        result[key] = float(value)

    if result["scene_graph_total_ms"] > result["total_ms"] + 1.0e-9:
        raise ValueError(
            f"{prefix} scene_graph_total_ms exceeds total_ms"
        )
    if result["temporal_ms"] > result["total_ms"] + 1.0e-9:
        raise ValueError(
            f"{prefix} temporal_ms exceeds total_ms"
        )

    for key in (
        "detection_count",
        "tracked_object_count",
        "relation_edge_count",
        "event_count",
        "pair_state_count",
    ):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"{prefix} {key} must be a non-negative integer"
            )
        result[key] = value
    return result


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be within (0,1]")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def stage_stats(values: list[float]) -> dict[str, float]:
    return {
        "p50": nearest_rank(values, 0.50),
        "p90": nearest_rank(values, 0.90),
        "p95": nearest_rank(values, 0.95),
        "p99": nearest_rank(values, 0.99),
        "mean": float(fmean(values)),
    }


def summarize(
    context: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    context = validate_context(context)
    if not samples:
        raise ValueError("latency summary requires timing samples")
    samples = [validate_sample(value) for value in samples]

    stage_map = {
        "detector": "detector_ms",
        "tracker": "tracker_ms",
        "region_prepare": "region_prepare_ms",
        "relation": "relation_ms",
        "assembly": "assembly_ms",
        "temporal": "temporal_ms",
        "total": "total_ms",
    }
    stages = {
        stage: stage_stats(
            [sample[source] for sample in samples]
        )
        for stage, source in stage_map.items()
    }

    def mean_count(key: str) -> float:
        return float(
            fmean(float(sample[key]) for sample in samples)
        )

    return {
        "schema": LATENCY_SCHEMA,
        "context_sha256": context_digest(context),
        "samples": len(samples),
        "percentile_method": "nearest-rank",
        "stages_ms": stages,
        "cardinality": {
            "detections_mean": mean_count("detection_count"),
            "tracked_objects_mean": mean_count(
                "tracked_object_count"
            ),
            "relation_edges_mean": mean_count(
                "relation_edge_count"
            ),
            "events_mean": mean_count("event_count"),
            "pair_states_mean": mean_count(
                "pair_state_count"
            ),
        },
        "scene_graph_total_ms": stage_stats(
            [sample["scene_graph_total_ms"] for sample in samples]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--samples-jsonl", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = summarize(
        load_json(args.context),
        load_samples(args.samples_jsonl),
    )
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
