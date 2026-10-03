from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from make_relation_qualification_report import (
    CONTEXT_SCHEMA,
    context_digest,
)
from summarize_scene_behavior_latency import (
    SAMPLE_SCHEMA,
    load_samples,
    nearest_rank,
    summarize,
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
        "detector_id": "detector-v1",
        "max_boxes": 32,
        "vocabulary_size": 17,
    }


def sample(index: int) -> dict:
    base = float(index)
    return {
        "schema": SAMPLE_SCHEMA,
        "detector_ms": 1.0 + base,
        "tracker_ms": 0.1 + base,
        "region_prepare_ms": 0.2 + base,
        "relation_ms": 2.0 + base,
        "assembly_ms": 0.05 + base,
        "scene_graph_total_ms": 4.0 + 5.0 * base,
        "temporal_ms": 0.2 + base,
        "total_ms": 5.0 + 6.0 * base,
        "detection_count": 2 + index,
        "tracked_object_count": 1 + index,
        "relation_edge_count": 3 + index,
        "event_count": index,
        "pair_state_count": 1 + index,
    }


class SceneBehaviorLatencySummaryTest(unittest.TestCase):
    def test_nearest_rank_is_stable(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(nearest_rank(values, 0.50), 3.0)
        self.assertEqual(nearest_rank(values, 0.90), 5.0)
        self.assertEqual(nearest_rank(values, 0.99), 5.0)

    def test_summary_binds_context_and_cardinality(self):
        ctx = context()
        report = summarize(
            ctx,
            [sample(index) for index in range(5)],
        )
        self.assertEqual(
            report["schema"],
            "kfcore.scene-behavior-latency/1",
        )
        self.assertEqual(
            report["context_sha256"],
            context_digest(ctx),
        )
        self.assertEqual(report["samples"], 5)
        self.assertEqual(
            report["percentile_method"],
            "nearest-rank",
        )
        self.assertEqual(
            report["stages_ms"]["detector"]["p50"],
            3.0,
        )
        self.assertEqual(
            report["stages_ms"]["detector"]["p90"],
            5.0,
        )
        self.assertEqual(
            report["cardinality"]["detections_mean"],
            4.0,
        )
        self.assertIn("scene_graph_total_ms", report)

    def test_jsonl_loader_rejects_invalid_sample(self):
        bad = sample(0)
        bad["relation_ms"] = -1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.jsonl"
            path.write_text(
                json.dumps(bad) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "relation_ms",
            ):
                load_samples(path)

    def test_total_must_cover_scene_graph_and_temporal(self):
        bad = sample(0)
        bad["scene_graph_total_ms"] = 99.0
        with self.assertRaisesRegex(
            ValueError,
            "exceeds total_ms",
        ):
            summarize(context(), [bad])


if __name__ == "__main__":
    unittest.main()
