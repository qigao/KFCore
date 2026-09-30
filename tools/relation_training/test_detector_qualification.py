from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from benchmark import RelationVocabulary
from evaluate_detector_qualification import (
    QUALIFICATION_SCHEMA,
    build_qualification_report,
    resolve_device,
    sha256,
    validate_sha256,
)


class DetectorQualificationCliTest(unittest.TestCase):
    def test_validate_sha256_normalizes_case_and_rejects_bad_values(self):
        value = "AB" * 32
        self.assertEqual(
            validate_sha256(value, "artifact"),
            value.lower(),
        )
        for invalid in (
            "",
            "a" * 63,
            "g" * 64,
            "sha256:" + "a" * 64,
        ):
            with self.assertRaises(ValueError):
                validate_sha256(invalid, "artifact")

    def test_build_report_pins_relation_provenance(self):
        vocabulary = RelationVocabulary(
            predicates=("holding", "riding"),
            object_labels=("person", "horse"),
        )
        payload = {
            "backbone_model": "hf_hub:reference",
            "predicates": ["holding", "riding"],
            "config": {
                "image_size": 224,
                "max_boxes": 32,
                "pair_budget": 128,
            },
            "extra": {
                "training_schema": "kfcore.apache-reference/1",
                "objective": "apache",
            },
        }
        metrics = {
            "schema": "kfcore.detector-relation-ceiling/1",
            "directed_pair_recoverability_ceiling": 0.75,
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "relation.pt"
            checkpoint.write_bytes(b"fixed-checkpoint-bytes")
            report = build_qualification_report(
                checkpoint_path=checkpoint,
                payload=payload,
                vocabulary=vocabulary,
                metrics=metrics,
            )

        self.assertEqual(
            report["schema"],
            QUALIFICATION_SCHEMA,
        )
        relation = report["relation"]
        self.assertEqual(
            relation["checkpoint_sha256"],
            sha256_bytes(b"fixed-checkpoint-bytes"),
        )
        self.assertEqual(
            relation["backbone_model"],
            "hf_hub:reference",
        )
        self.assertEqual(
            relation["predicate_count"],
            2,
        )
        self.assertEqual(
            relation["vocabulary_sha256"],
            vocabulary.sha256(),
        )
        self.assertEqual(
            relation["checkpoint_extra_keys"],
            ["objective", "training_schema"],
        )
        self.assertEqual(report["metrics"], metrics)

    def test_build_report_rejects_predicate_order_drift(self):
        vocabulary = RelationVocabulary(
            predicates=("holding", "riding"),
        )
        payload = {
            "backbone_model": "reference",
            "predicates": ["riding", "holding"],
            "config": {},
            "extra": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "relation.pt"
            checkpoint.write_bytes(b"x")
            with self.assertRaises(ValueError):
                build_qualification_report(
                    checkpoint_path=checkpoint,
                    payload=payload,
                    vocabulary=vocabulary,
                    metrics={},
                )

    def test_backbone_override_is_recorded_explicitly(self):
        vocabulary = RelationVocabulary(
            predicates=("holding",),
        )
        payload = {
            "backbone_model": "remote/reference",
            "predicates": ["holding"],
            "config": {},
            "extra": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "relation.pt"
            checkpoint.write_bytes(b"x")
            report = build_qualification_report(
                checkpoint_path=checkpoint,
                payload=payload,
                vocabulary=vocabulary,
                metrics={},
                backbone_override="/local/dinov3",
            )
        self.assertEqual(
            report["relation"]["backbone_model"],
            "/local/dinov3",
        )

    def test_cpu_device_resolution_is_deterministic(self):
        self.assertEqual(
            resolve_device("cpu"),
            torch.device("cpu"),
        )


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    unittest.main()
