from __future__ import annotations

import copy
import unittest

from relation_backend_matrix import (
    MATRIX_SCHEMA,
    make_matrix,
    make_released_matrix,
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
        "relation_model_sha256": h("1"),
        "vocabulary_sha256": h("2"),
        "relation_config_sha256": h("3"),
        "detector_model_sha256": h("4"),
        "detector_config_sha256": h("5"),
        "backend": "onnxruntime",
        "device": "cpu",
        "relation_model_type": "relation.open-vocabulary",
        "detector_id": "yolox-tiny-coco-0.1.1rc0",
        "max_boxes": 32,
        "vocabulary_size": 14,
        "predicate_bank_sha256": h("6"),
        "detector_predictions_sha256": h("7"),
        "annotations_sha256": h("8"),
        "image_corpus_sha256": h("9"),
        "pair_weight": 1.0,
        "top_ks": [20, 50, 100],
        "runtime_provenance": {
            "hardware": {
                "machine": "x86_64",
                "cpu_model": "fixture-cpu",
                "logical_cpu_count": 4,
            },
            "software": {
                "os": "fixture-os",
                "python": "3.11",
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
        "detector": {
            "id": "yolox-tiny-coco-0.1.1rc0",
            "model_sha256": h("4"),
            "config_sha256": h("5"),
        },
        "object_recoverability_ceiling": 0.85,
        "directed_pair_recoverability_ceiling": 0.68,
        "sampler_recall_conditional_on_recoverable_pairs": 1.0,
        "sampler_pair_recall_end_to_end": 0.68,
        "ground_truth_triplets": 53,
        "failure_decomposition_at_max_k": {
            "max_k": 100,
            "detector_miss": 17,
            "sampler_miss": 0,
            "predicate_miss": 2,
            "recovered": 34,
        },
    }


def released_latency(digest: str) -> dict[str, object]:
    stages = {}
    for name, base in (
        ("detector", 80.0),
        ("tracker", 0.04),
        ("region_prepare", 0.001),
        ("relation", 500.0),
        ("assembly", 0.001),
        ("temporal", 0.01),
        ("total", 590.0),
    ):
        stages[name] = {
            "p50": base,
            "p90": base + 1.0,
            "p95": base + 2.0,
            "p99": base + 3.0,
            "mean": base + 0.5,
        }
    return {
        "schema": "kfcore.scene-behavior-latency/1",
        "context_sha256": digest,
        "samples": 32,
        "stages_ms": stages,
        "cardinality": {
            "detections_mean": 9.4,
            "tracked_objects_mean": 9.4,
            "relation_edges_mean": 50.0,
            "events_mean": 0.0,
            "pair_states_mean": 7.0,
        },
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
    def build(self, *, trt=None):
        return make_matrix(
            quality_report=quality_report(),
            dynamic_export=export_metadata(),
            encoder_export=export_metadata(encoder=True),
            ort_cpu_report=ort_cpu_report(),
            ort_cpu_provenance=provenance(),
            tensorrt_report=trt,
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

    def test_tensorrt_must_use_same_dynamic_onnx(self):
        trt = tensorrt_report()
        trt["source_onnx_sha256"] = h("8")
        with self.assertRaisesRegex(
            ValueError,
            "source ONNX",
        ):
            self.build(trt=trt)



class ReleasedRelationBackendMatrixTest(unittest.TestCase):
    def test_released_matrix_keeps_missing_encoder_explicit(self):
        matrix = make_released_matrix(released_report())
        self.assertEqual(matrix["schema"], MATRIX_SCHEMA)
        self.assertEqual(
            matrix["lineage"]["kind"],
            "upstream-released-deployment",
        )
        self.assertFalse(
            matrix["availability"]["encoder_export_available"]
        )
        self.assertFalse(
            matrix["availability"]["host_query_scorer_reference"]
        )
        self.assertFalse(
            matrix["acceptance"]["pair_keyed_output_parity"]
        )
        self.assertTrue(
            matrix["acceptance"]["no_synthetic_encoder_lineage"]
        )
        self.assertEqual(
            matrix["backends"][0]["measurement_scope"],
            "full-scene-behavior",
        )
        self.assertTrue(
            matrix["acceptance"]["quality_and_latency_together"]
        )

    def test_released_matrix_rejects_context_digest_drift(self):
        report = released_report()
        report["context_sha256"] = h("a")
        with self.assertRaisesRegex(
            ValueError,
            "context digest differs",
        ):
            make_released_matrix(report)

    def test_released_matrix_rejects_detector_lineage_drift(self):
        report = released_report()
        report["quality"]["detector"]["model_sha256"] = h("b")
        with self.assertRaisesRegex(
            ValueError,
            "detector model differs",
        ):
            make_released_matrix(report)

    def test_released_matrix_is_deterministic(self):
        first = make_released_matrix(released_report())
        second = make_released_matrix(released_report())
        self.assertEqual(stable_json(first), stable_json(second))


if __name__ == "__main__":
    unittest.main()
