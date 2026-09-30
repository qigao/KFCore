from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import unittest

from make_relation_qualification_report import (
    CONTEXT_SCHEMA,
    LATENCY_SCHEMA,
    QUALITY_SCHEMA,
    context_digest,
    make_report,
    render_markdown,
)


def h(char: str) -> str:
    return char * 64


def context() -> dict:
    return {
        "schema": CONTEXT_SCHEMA,
        "relation_model_sha256": h("1"),
        "vocabulary_sha256": h("2"),
        "relation_config_sha256": h("3"),
        "detector_model_sha256": h("4"),
        "detector_config_sha256": h("5"),
        "backend": "onnxruntime-cpu",
        "device": "cpu",
        "relation_model_type": "relation.open-vocabulary",
        "detector_id": "synthetic-yolo",
        "max_boxes": 32,
        "vocabulary_size": 17,
    }


def quality() -> dict:
    return {
        "schema": QUALITY_SCHEMA,
        "examples": 4,
        "annotations_sha256": h("6"),
        "detector_predictions_sha256": h("7"),
        "detector": {
            "id": "synthetic-yolo",
            "model_sha256": h("4"),
            "config_sha256": h("5"),
            "iou_threshold": 0.5,
            "boxes_before_cap": 20,
            "boxes_used": 16,
        },
        "ground_truth_objects": 12,
        "recoverable_ground_truth_objects": 10,
        "object_recoverability_ceiling": 10.0 / 12.0,
        "ground_truth_directed_pairs": 8,
        "recoverable_directed_pairs": 6,
        "directed_pair_recoverability_ceiling": 0.75,
        "sampled_recoverable_pairs": 5,
        "sampler_recall_conditional_on_recoverable_pairs": 5.0 / 6.0,
        "sampler_pair_recall_end_to_end": 5.0 / 8.0,
        "ground_truth_triplets": 10,
        "recoverable_ground_truth_triplets": 8,
        "sampled_recoverable_triplets": 7,
        "conditional_recall_at_k": {"20": 0.5, "50": 0.75, "100": 0.875},
        "end_to_end_recall_at_k": {"20": 0.4, "50": 0.6, "100": 0.7},
        "conditional_mean_recall_at_k": {"20": 0.45, "50": 0.7, "100": 0.8},
        "end_to_end_mean_recall_at_k": {"20": 0.35, "50": 0.55, "100": 0.65},
        "failure_decomposition_at_max_k": {
            "max_k": 100,
            "detector_miss": 2,
            "sampler_miss": 1,
            "predicate_miss": 1,
            "recovered": 6,
        },
    }


def latency(ctx: dict) -> dict:
    def stage(base: float) -> dict:
        return {
            "p50": base,
            "p90": base + 1.0,
            "p95": base + 2.0,
            "p99": base + 3.0,
            "mean": base + 0.5,
        }

    return {
        "schema": LATENCY_SCHEMA,
        "context_sha256": context_digest(ctx),
        "samples": 100,
        "stages_ms": {
            "detector": stage(3.0),
            "tracker": stage(0.2),
            "region_prepare": stage(0.1),
            "relation": stage(6.0),
            "assembly": stage(0.1),
            "temporal": stage(0.2),
            "total": stage(10.0),
        },
        "cardinality": {
            "detections_mean": 4.5,
            "tracked_objects_mean": 3.8,
            "relation_edges_mean": 5.2,
            "events_mean": 0.4,
            "pair_states_mean": 3.0,
        },
    }


class RelationQualificationReportTest(unittest.TestCase):
    def test_matching_context_builds_report_and_markdown(self):
        ctx = context()
        report = make_report(
            ctx,
            quality(),
            latency(ctx),
        )
        self.assertEqual(
            report["schema"],
            "kfcore.relation-qualification-report/1",
        )
        self.assertEqual(
            report["context_sha256"],
            context_digest(ctx),
        )
        markdown = render_markdown(report)
        self.assertIn(
            "Directed pair recoverability ceiling",
            markdown,
        )
        self.assertIn("| relation |", markdown)

    def test_latency_context_mismatch_is_rejected(self):
        ctx = context()
        bad_latency = latency(ctx)
        bad_latency["context_sha256"] = h("a")
        with self.assertRaisesRegex(
            ValueError,
            "different qualification context",
        ):
            make_report(
                ctx,
                quality(),
                bad_latency,
            )

    def test_detector_provenance_mismatch_is_rejected(self):
        ctx = context()
        bad_quality = quality()
        bad_quality["detector"]["model_sha256"] = h("b")
        with self.assertRaisesRegex(
            ValueError,
            "detector model",
        ):
            make_report(
                ctx,
                bad_quality,
                latency(ctx),
            )

    def test_invalid_latency_percentiles_are_rejected(self):
        ctx = context()
        bad_latency = latency(ctx)
        bad_latency["stages_ms"]["relation"]["p95"] = 1.0
        with self.assertRaisesRegex(
            ValueError,
            "percentiles",
        ):
            make_report(
                ctx,
                quality(),
                bad_latency,
            )

    def test_failure_decomposition_must_sum_to_gt(self):
        ctx = context()
        bad_quality = quality()
        bad_quality[
            "failure_decomposition_at_max_k"
        ]["recovered"] = 5
        with self.assertRaisesRegex(
            ValueError,
            "does not sum",
        ):
            make_report(
                ctx,
                bad_quality,
                latency(ctx),
            )


if __name__ == "__main__":
    unittest.main()
