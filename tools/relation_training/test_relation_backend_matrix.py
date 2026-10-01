from __future__ import annotations

import copy
import unittest

from relation_backend_matrix import (
    MATRIX_SCHEMA,
    make_matrix,
    stable_json,
)


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


if __name__ == "__main__":
    unittest.main()
