from __future__ import annotations

import copy
import unittest

from relation_backend_matrix import (
    MATRIX_SCHEMA,
    make_matrix,
    make_released_deployment_matrix,
    stable_json,
)
from make_relation_qualification_report import context_digest


def h(char: str) -> str:
    return char * 64


def export_metadata(*, encoder: bool = False) -> dict[str, object]:
    return {
        "schema": "kfcore.relation-onnx/2",
        "model_type": (
            "relation.open-vocabulary-encoder"
            if encoder
            else "relation.open-vocabulary"
        ),
        "output_kind": (
            "relation-queries"
            if encoder
            else "dynamic-vocabulary-logits"
        ),
        "checkpoint_sha256": h("1"),
        "onnx_sha256": h("3" if encoder else "2"),
        "backbone_model": "hf_hub:timm/dinov3",
        "image_size": 448,
        "max_boxes": 32,
        "final_budget": 128,
        "query_dim": 512,
        "score_logit_scale": 14.0,
        "score_logit_bias": -0.25,
        "score_contract": (
            "scale*((1-alpha)*cos(q_sem,W)+alpha*cos(q_spa,W))+bias"
        ),
    }


def quality_report() -> dict[str, object]:
    return {
        "schema": "kfcore.detector-box-relation-qualification/1",
        "relation": {
            "checkpoint_sha256": h("1"),
            "config_sha256": h("4"),
            "vocabulary_sha256": h("5"),
        },
        "evaluation": {
            "pair_weight": 1.0,
        },
        "quality": {
            "object_recoverability_ceiling": 0.9,
            "directed_pair_recoverability_ceiling": 0.8,
            "sampler_recall_conditional_on_recoverable_pairs": 0.75,
            "sampler_pair_recall_end_to_end": 0.6,
        },
    }


def policy() -> dict[str, object]:
    return {
        "logit_scale": 14.0,
        "logit_bias": -0.25,
        "pair_weight": 1.0,
        "calibration_a": 1.0,
        "calibration_b": 0.0,
        "threshold": 0.4,
        "top_k": 20,
        "weight_ranking_by_detector_score": False,
    }


def case(label: str, size: int) -> dict[str, object]:
    return {
        "label": label,
        "predicate_count": size,
        "vocabulary_version": 1,
        "region_count": 3,
        "valid_pair_count": 3,
        "edge_count": 3,
        "preprocess_ms": 0.1,
        "backend_ms": 2.0,
        "predicate_score_ms": 0.0,
        "backbone_context_ms": None,
        "predicate_scoring_ms": None,
        "predicate_scoring_in_backend": True,
        "runtime_ms": 2.0,
        "decode_ms": 0.2,
        "total_ms": 2.3,
        "pair_keys": [[0, 1], [1, 0], [0, 2]],
    }


def host_case(label: str, size: int) -> dict[str, object]:
    value = case(label, size)
    value.update({
        "backend_ms": 1.5,
        "predicate_score_ms": 0.5,
        "backbone_context_ms": 1.5,
        "predicate_scoring_ms": 0.5,
        "predicate_scoring_in_backend": False,
        "runtime_ms": 2.0,
    })
    return value


def ort_cpu_report() -> dict[str, object]:
    specs = (
        ("v1", 1),
        ("v3", 3),
        ("default", 4),
        ("large", 64),
        ("v1-repeat", 1),
    )
    return {
        "schema": "kfcore.relation-ort-cpu-cpp-qualification/1",
        "passed": True,
        "backend_scoring": True,
        "host_fallback_reference": True,
        "backend_host_pair_keyed_parity": True,
        "provider": "onnxruntime",
        "device": "cpu",
        "model_sha256": h("2"),
        "host_model_sha256": h("3"),
        "score_decode_policy": policy(),
        "cases": [case(label, size) for label, size in specs],
        "host_cases": [
            host_case(label, size) for label, size in specs
        ],
    }


def provenance() -> dict[str, object]:
    return {
        "schema": "kfcore.relation-backend-provenance/1",
        "backend": "onnxruntime-cpu",
        "hardware": {
            "machine": "x86_64",
            "cpu_model": "fixture-cpu",
            "logical_cpu_count": 8,
        },
        "software": {
            "os": "fixture-os",
            "python": "3.11.0",
            "onnxruntime": "1.22.0",
            "compiler": "fixture-c++",
        },
    }


def tensorrt_report() -> dict[str, object]:
    def trt_case(label: str, size: int) -> dict[str, object]:
        return {
            "label": label,
            "vocabulary_size": size,
            "pair_set_equal": True,
            "reference_valid_pairs": 3,
            "actual_valid_pairs": 3,
            "passed": True,
            "tensorrt_latency": {
                "mean_ms": 1.0,
                "median_ms": 0.9,
                "p95_ms": 1.2,
                "min_ms": 0.8,
                "max_ms": 1.3,
            },
        }

    return {
        "schema": "kfcore.tensorrt-dynamic-vocab-qualification/1",
        "hardware_executed": True,
        "same_engine_reused": True,
        "same_context_reused": True,
        "passed": True,
        "source_onnx_sha256": h("2"),
        "profile": {
            "W": {
                "min": [1, 512],
                "opt": [64, 512],
                "max": [1024, 512],
            }
        },
        "hardware": {
            "tensorrt_version": "10",
            "torch_version": "2",
            "cuda_version": "13",
            "gpu": "fixture-gpu",
            "compute_capability": [9, 0],
            "engine_load_count": 1,
            "context_create_count": 1,
        },
        "cases": [
            trt_case("v1", 1),
            trt_case("v3", 3),
            trt_case("opt", 64),
            trt_case("max", 1024),
        ],
    }


def released_context() -> dict[str, object]:
    return {
        "schema": "kfcore.relation-qualification-context/1",
        "relation_model_sha256": h("a"),
        "vocabulary_sha256": h("b"),
        "relation_config_sha256": h("c"),
        "detector_model_sha256": h("d"),
        "detector_config_sha256": h("e"),
        "backend": "onnxruntime",
        "device": "cpu",
        "relation_model_type": "relation.open-vocabulary",
        "detector_id": "yolox-tiny",
        "max_boxes": 32,
        "vocabulary_size": 14,
        "predicate_bank_sha256": h("f"),
        "detector_predictions_sha256": h("1"),
        "annotations_sha256": h("2"),
        "image_corpus_sha256": h("3"),
        "dataset": "HICO-DET test",
        "dataset_revision": "fixture-revision",
        "dataset_source_parquet_sha256": h("4"),
        "quality_input_contract": "detector-boxes",
        "quality_iou_threshold": 0.5,
        "pair_weight": 1.0,
        "top_ks": [20, 50, 100],
        "tracking_sample_policy": "independent-epoch-warm-then-measure",
        "runtime_provenance": {
            "hardware": {
                "machine": "x86_64",
                "cpu_model": "fixture-cpu",
                "logical_cpu_count": 4,
            },
            "software": {
                "os": "fixture-os",
                "python": "3.11.0",
                "onnxruntime_python": "1.22.0",
                "onnxruntime_sdk": "1.22.0",
                "compiler": "fixture-c++",
            },
        },
    }


def released_quality() -> dict[str, object]:
    return {
        "schema": "kfcore.detector-relation-ceiling/1",
        "examples": 32,
        "annotations_sha256": h("2"),
        "detector_predictions_sha256": h("1"),
        "detector": {
            "id": "yolox-tiny",
            "model_sha256": h("d"),
            "config_sha256": h("e"),
            "iou_threshold": 0.5,
            "boxes_before_cap": 303,
            "boxes_used": 303,
        },
        "ground_truth_objects": 74,
        "recoverable_ground_truth_objects": 63,
        "object_recoverability_ceiling": 0.85,
        "ground_truth_directed_pairs": 38,
        "recoverable_directed_pairs": 26,
        "directed_pair_recoverability_ceiling": 0.68,
        "sampled_recoverable_pairs": 26,
        "sampler_recall_conditional_on_recoverable_pairs": 1.0,
        "sampler_pair_recall_end_to_end": 0.68,
        "ground_truth_triplets": 53,
        "recoverable_ground_truth_triplets": 36,
        "sampled_recoverable_triplets": 36,
        "conditional_recall_at_k": {
            "20": 0.86,
            "50": 0.94,
            "100": 0.94,
        },
        "end_to_end_recall_at_k": {
            "20": 0.58,
            "50": 0.64,
            "100": 0.64,
        },
        "conditional_mean_recall_at_k": {
            "20": 0.83,
            "50": 0.91,
            "100": 0.91,
        },
        "end_to_end_mean_recall_at_k": {
            "20": 0.60,
            "50": 0.66,
            "100": 0.66,
        },
        "failure_decomposition_at_max_k": {
            "max_k": 100,
            "detector_miss": 17,
            "sampler_miss": 0,
            "predicate_miss": 2,
            "recovered": 34,
        },
    }


def released_latency(context_sha: str) -> dict[str, object]:
    stats = {
        "p50": 1.0,
        "p90": 2.0,
        "p95": 3.0,
        "p99": 4.0,
        "mean": 2.0,
    }
    return {
        "schema": "kfcore.scene-behavior-latency/1",
        "context_sha256": context_sha,
        "samples": 32,
        "percentile_method": "nearest-rank",
        "stages_ms": {
            name: dict(stats)
            for name in (
                "detector",
                "tracker",
                "region_prepare",
                "relation",
                "assembly",
                "temporal",
                "total",
            )
        },
        "cardinality": {
            "detections_mean": 9.0,
            "tracked_objects_mean": 9.0,
            "relation_edges_mean": 50.0,
            "events_mean": 0.0,
            "pair_states_mean": 7.0,
        },
        "scene_graph_total_ms": dict(stats),
    }


def released_report() -> dict[str, object]:
    context = released_context()
    digest = context_digest(context)
    return {
        "schema": "kfcore.relation-qualification-report/1",
        "context_sha256": digest,
        "context": context,
        "quality": released_quality(),
        "latency": released_latency(digest),
    }


class RelationBackendMatrixTest(unittest.TestCase):
    def build(self, *, trt=None, released=None):
        return make_matrix(
            quality_report=quality_report(),
            dynamic_export=export_metadata(),
            encoder_export=export_metadata(encoder=True),
            ort_cpu_report=ort_cpu_report(),
            ort_cpu_provenance=provenance(),
            tensorrt_report=trt,
            released_deployment_report=released,
        )

    def test_builds_deterministic_quality_latency_matrix(self):
        first = self.build()
        second = self.build()
        self.assertEqual(first["schema"], MATRIX_SCHEMA)
        self.assertEqual(stable_json(first), stable_json(second))
        self.assertEqual(
            [row["id"] for row in first["backends"]],
            [
                "host-query-scorer-reference",
                "onnxruntime-cpu",
            ],
        )
        self.assertTrue(
            first["acceptance"]["quality_and_latency_together"]
        )
        self.assertTrue(
            first["acceptance"]["score_decode_policy_shared"]
        )
        self.assertTrue(
            first["acceptance"]["pair_keyed_output_parity"]
        )
        self.assertEqual(
            [
                case["vocabulary_size"]
                for case in first["backends"][1]["cases"]
            ],
            [1, 3, 4, 64, 1],
        )
        host_large = first["backends"][0]["cases"][3]
        self.assertEqual(host_large["vocabulary_size"], 64)
        self.assertEqual(
            host_large["timing"]["predicate_scoring_ms"],
            0.5,
        )
        backend_large = first["backends"][1]["cases"][3]
        self.assertIsNone(
            backend_large["timing"]["predicate_scoring_ms"]
        )
        self.assertTrue(
            backend_large["predicate_scoring_in_backend"]
        )

    def test_rejects_checkpoint_lineage_drift(self):
        quality = quality_report()
        quality["relation"]["checkpoint_sha256"] = h("9")
        with self.assertRaisesRegex(
            ValueError,
            "different checkpoints",
        ):
            make_matrix(
                quality_report=quality,
                dynamic_export=export_metadata(),
                encoder_export=export_metadata(encoder=True),
                ort_cpu_report=ort_cpu_report(),
                ort_cpu_provenance=provenance(),
            )

    def test_rejects_score_decode_policy_drift(self):
        ort = ort_cpu_report()
        ort["score_decode_policy"]["pair_weight"] = 0.5
        with self.assertRaisesRegex(
            ValueError,
            "pair weight",
        ):
            make_matrix(
                quality_report=quality_report(),
                dynamic_export=export_metadata(),
                encoder_export=export_metadata(encoder=True),
                ort_cpu_report=ort,
                ort_cpu_provenance=provenance(),
            )

    def test_rejects_pair_key_drift(self):
        ort = ort_cpu_report()
        ort["host_cases"][2]["pair_keys"] = [
            [0, 1],
            [1, 0],
            [2, 0],
        ]
        with self.assertRaisesRegex(
            ValueError,
            "pair keys differ",
        ):
            make_matrix(
                quality_report=quality_report(),
                dynamic_export=export_metadata(),
                encoder_export=export_metadata(encoder=True),
                ort_cpu_report=ort,
                ort_cpu_provenance=provenance(),
            )

    def test_tensorrt_engine_proof_is_kept_separate_from_full_latency(self):
        matrix = self.build(trt=tensorrt_report())
        self.assertTrue(
            matrix["availability"]["tensorrt_engine_proof"]
        )
        self.assertFalse(
            matrix["availability"]["tensorrt_full_relation"]
        )
        evidence = matrix["optional_engine_qualifications"][0]
        self.assertEqual(evidence["id"], "tensorrt-engine")
        self.assertEqual(
            evidence["measurement_scope"],
            "engine-only",
        )
        self.assertFalse(evidence["full_relation_stage_latency"])

    def test_released_deployment_matrix_is_explicitly_separate(self):
        report = released_report()
        matrix = make_released_deployment_matrix(report)
        self.assertEqual(matrix["schema"], MATRIX_SCHEMA)
        self.assertEqual(
            matrix["mode"],
            "upstream-released-deployment",
        )
        self.assertEqual(
            [row["id"] for row in matrix["backends"]],
            ["upstream-released-onnxruntime-cpu"],
        )
        self.assertTrue(
            matrix["acceptance"]["quality_and_latency_together"]
        )
        self.assertTrue(
            matrix["acceptance"]["released_context_digest_valid"]
        )
        self.assertFalse(
            matrix["acceptance"][
                "baseline_pair_keyed_output_parity_applicable"
            ]
        )
        self.assertFalse(
            matrix["availability"]["host_query_scorer_reference"]
        )
        self.assertFalse(
            matrix["boundary"]["encoder_query_export_available"]
        )

    def test_released_deployment_can_attach_without_changing_baseline(self):
        matrix = self.build(released=released_report())
        self.assertEqual(
            [row["id"] for row in matrix["backends"]],
            [
                "host-query-scorer-reference",
                "onnxruntime-cpu",
            ],
        )
        self.assertEqual(
            len(matrix["measured_deployments"]),
            1,
        )
        self.assertEqual(
            matrix["measured_deployments"][0]["id"],
            "upstream-released-onnxruntime-cpu",
        )
        self.assertTrue(
            matrix["acceptance"]["pair_keyed_output_parity"]
        )

    def test_released_deployment_rejects_context_digest_drift(self):
        report = released_report()
        report["context_sha256"] = h("9")
        with self.assertRaisesRegex(
            ValueError,
            "context digest differs",
        ):
            make_released_deployment_matrix(report)

    def test_released_deployment_rejects_detector_prediction_drift(self):
        report = released_report()
        report["quality"]["detector_predictions_sha256"] = h("9")
        with self.assertRaisesRegex(
            ValueError,
            "detector predictions differ",
        ):
            make_released_deployment_matrix(report)

    def test_released_deployment_rejects_quality_iou_drift(self):
        report = released_report()
        report["quality"]["detector"]["iou_threshold"] = 0.6
        with self.assertRaisesRegex(
            ValueError,
            "IoU threshold differs",
        ):
            make_released_deployment_matrix(report)

    def test_tensorrt_must_use_same_dynamic_onnx(self):
        trt = tensorrt_report()
        trt["source_onnx_sha256"] = h("8")
        with self.assertRaisesRegex(
            ValueError,
            "source ONNX",
        ):
            self.build(trt=trt)


if __name__ == "__main__":
    unittest.main()
