from __future__ import annotations

import unittest

import numpy as np

from tensorrt_dynamic_vocab_qualification import (
    SCHEMA,
    compare_pair_keyed_outputs,
    deterministic_inputs,
    qualification_cases,
    validate_qualification_report,
)


def metadata() -> dict[str, object]:
    fixed = {
        "image": {
            "min": [1, 3, 448, 448],
            "opt": [1, 3, 448, 448],
            "max": [1, 3, 448, 448],
        },
        "boxes": {
            "min": [1, 40, 4],
            "opt": [1, 40, 4],
            "max": [1, 40, 4],
        },
        "box_counts": {
            "min": [1],
            "opt": [1],
            "max": [1],
        },
    }
    return {
        "model_type": "relation.open-vocabulary",
        "image_size": 448,
        "max_boxes": 40,
        "query_dim": 512,
        "onnx_sha256": "a" * 64,
        "tensorrt_engine_sha256": "b" * 64,
        "tensorrt": {
            "source_onnx_sha256": "a" * 64,
            "profiles": {
                **fixed,
                "W": {
                    "min": [1, 512],
                    "opt": [243, 512],
                    "max": [1024, 512],
                },
                "alpha": {
                    "min": [1],
                    "opt": [243],
                    "max": [1024],
                },
            },
        },
    }


def outputs(order: list[int], *, vocabulary_size: int = 3) -> dict[str, np.ndarray]:
    pairs = [(0, 1), (2, 3), (4, 5)]
    pred_rows = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [1.1, 1.2, 1.3],
            [2.1, 2.2, 2.3],
        ],
        dtype=np.float32,
    )[:, :vocabulary_size]
    pair_rows = np.asarray([0.5, 1.5, 2.5], dtype=np.float32)
    sub = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    obj = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    valid = np.asarray([True, True, True], dtype=np.bool_)
    return {
        "pred_logits": pred_rows[order][None, ...],
        "pair_logits": pair_rows[order][None, ...],
        "sub_idx": sub[order][None, ...],
        "obj_idx": obj[order][None, ...],
        "valid_mask": valid[order][None, ...],
    }


class TensorRtDynamicVocabularyQualificationTest(unittest.TestCase):
    def test_cases_cover_v1_v3_opt_max(self) -> None:
        self.assertEqual(
            qualification_cases(metadata()),
            [
                {"label": "v1", "vocabulary_size": 1},
                {"label": "v3", "vocabulary_size": 3},
                {"label": "opt", "vocabulary_size": 243},
                {"label": "max", "vocabulary_size": 1024},
            ],
        )

    def test_cases_reject_w_alpha_profile_drift(self) -> None:
        value = metadata()
        value["tensorrt"]["profiles"]["alpha"]["max"] = [1000]
        with self.assertRaisesRegex(ValueError, "W/alpha"):
            qualification_cases(value)

    def test_deterministic_inputs_are_repeatable_and_normalized(self) -> None:
        a = deterministic_inputs(metadata(), 3, seed=17)
        b = deterministic_inputs(metadata(), 3, seed=17)

        for name in ("image", "boxes", "box_counts", "W", "alpha"):
            self.assertTrue(np.array_equal(a[name], b[name]), name)

        self.assertEqual(a["image"].shape, (1, 3, 448, 448))
        self.assertEqual(a["boxes"].shape, (1, 40, 4))
        self.assertEqual(a["box_counts"].tolist(), [40])
        self.assertEqual(a["W"].shape, (3, 512))
        self.assertTrue(
            np.allclose(
                np.linalg.norm(a["W"], axis=1),
                np.ones(3, dtype=np.float32),
                atol=1.0e-6,
            )
        )
        self.assertEqual(a["alpha"].tolist(), [0.0, 0.5, 1.0])

    def test_pair_key_comparison_allows_row_reordering(self) -> None:
        reference = outputs([0, 1, 2])
        actual = outputs([2, 0, 1])
        report = compare_pair_keyed_outputs(reference, actual)
        self.assertTrue(report["passed"])
        self.assertTrue(report["pair_set_equal"])
        self.assertEqual(report["reference_valid_pairs"], 3)
        self.assertEqual(report["actual_valid_pairs"], 3)

    def test_pair_key_comparison_rejects_pair_set_drift(self) -> None:
        reference = outputs([0, 1, 2])
        actual = outputs([0, 1, 2])
        actual["obj_idx"][0, 2] = 6
        report = compare_pair_keyed_outputs(reference, actual)
        self.assertFalse(report["passed"])
        self.assertFalse(report["pair_set_equal"])
        self.assertEqual(report["missing_pairs"], [[4, 5]])
        self.assertEqual(report["extra_pairs"], [[4, 6]])

    def test_numeric_comparison_enforces_tolerance(self) -> None:
        reference = outputs([0, 1, 2])
        actual = outputs([0, 1, 2])
        actual["pred_logits"][0, 1, 1] += 0.01
        report = compare_pair_keyed_outputs(
            reference,
            actual,
            abs_tol=1.0e-5,
            rel_tol=1.0e-5,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["pred_logits"]["passed"])

    def test_report_requires_real_hardware_and_reuse(self) -> None:
        case = {
            "label": "v1",
            "passed": True,
            "pair_set_equal": True,
        }
        report = {
            "schema": SCHEMA,
            "hardware_executed": True,
            "same_engine_reused": True,
            "same_context_reused": True,
            "engine_sha256_before": "a" * 64,
            "engine_sha256_after": "a" * 64,
            "cases": [
                {**case, "label": "v1"},
                {**case, "label": "v3"},
                {**case, "label": "opt"},
                {**case, "label": "max"},
            ],
        }
        self.assertEqual(validate_qualification_report(report)["schema"], SCHEMA)

        invalid = dict(report)
        invalid["hardware_executed"] = False
        with self.assertRaisesRegex(ValueError, "real hardware"):
            validate_qualification_report(invalid)

        invalid = dict(report)
        invalid["same_context_reused"] = False
        with self.assertRaisesRegex(ValueError, "one-context"):
            validate_qualification_report(invalid)


if __name__ == "__main__":
    unittest.main()
