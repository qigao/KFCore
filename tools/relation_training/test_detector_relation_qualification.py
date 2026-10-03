from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch

from checkpoint import CHECKPOINT_SCHEMA
from model import RelationModelConfig
from run_detector_relation_qualification import (
    REPORT_SCHEMA,
    run_qualification,
    stable_json_bytes,
)


def h(char: str) -> str:
    return char * 64


def apache_config() -> RelationModelConfig:
    return RelationModelConfig(
        pair_evidence_contract="apache",
        pair_sampler_contract="apache",
        relation_context_contract="apache",
        predicate_head_contract="apache",
    )


def quality_report(
    *,
    annotations_sha256: str,
    predictions_sha256: str,
    detector_id: str,
    detector_model_sha256: str,
    detector_config_sha256: str,
    iou_threshold: float,
) -> dict[str, object]:
    return {
        "schema": "kfcore.detector-relation-ceiling/1",
        "examples": 1,
        "annotations_sha256": annotations_sha256,
        "detector_predictions_sha256": predictions_sha256,
        "detector": {
            "id": detector_id,
            "model_sha256": detector_model_sha256,
            "config_sha256": detector_config_sha256,
            "iou_threshold": iou_threshold,
            "boxes_before_cap": 2,
            "boxes_used": 2,
        },
        "object_recoverability_ceiling": 1.0,
        "directed_pair_recoverability_ceiling": 1.0,
        "sampler_recall_conditional_on_recoverable_pairs": 1.0,
        "sampler_pair_recall_end_to_end": 1.0,
        "conditional_recall_at_k": {"20": 1.0},
        "conditional_mean_recall_at_k": {"20": 1.0},
        "end_to_end_recall_at_k": {"20": 1.0},
        "end_to_end_mean_recall_at_k": {"20": 1.0},
        "failure_decomposition_at_max_k": {
            "max_k": 20,
            "detector_miss": 0,
            "sampler_miss": 0,
            "predicate_miss": 0,
            "recovered": 1,
        },
    }


class DetectorRelationQualificationTest(unittest.TestCase):
    def write_inputs(
        self,
        root: Path,
        *,
        checkpoint_predicates: list[str] | None = None,
        config: RelationModelConfig | None = None,
    ) -> dict[str, Path]:
        predicates = ["holding"]
        vocabulary = root / "vocabulary.json"
        vocabulary.write_text(
            json.dumps(
                {
                    "schema": "kfcore.relation-vocab/1",
                    "predicates": predicates,
                    "objects": ["person", "cup"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        annotations = root / "annotations.jsonl"
        annotations.write_text(
            json.dumps(
                {
                    "image": "sample.jpg",
                    "width": 100,
                    "height": 100,
                    "boxes_xyxy": [
                        [0, 0, 20, 20],
                        [40, 0, 60, 20],
                    ],
                    "object_labels": ["person", "cup"],
                    "relations": [[0, 0, 1]],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        image_root = root / "images"
        image_root.mkdir()
        (image_root / "sample.jpg").write_bytes(b"fixed-image-corpus")

        detector = root / "detector.jsonl"
        detector.write_text(
            json.dumps(
                {
                    "schema": "kfcore.detector-boxes/1",
                    "image": "sample.jpg",
                    "width": 100,
                    "height": 100,
                    "boxes_xyxy": [
                        [0, 0, 20, 20],
                        [40, 0, 60, 20],
                    ],
                    "scores": [0.9, 0.8],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        checkpoint = root / "relation.pt"
        model_config = config or apache_config()
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "backbone_model": (
                    "hf_hub:timm/"
                    "vit_small_patch16_dinov3.lvd1689m"
                ),
                "config": asdict(model_config),
                "predicates": (
                    checkpoint_predicates
                    if checkpoint_predicates is not None
                    else predicates
                ),
                "predicate_embeddings": torch.ones(1, 4),
                "state_dict": {},
                "extra": {},
            },
            checkpoint,
        )
        return {
            "checkpoint": checkpoint,
            "vocabulary": vocabulary,
            "annotations": annotations,
            "image_root": image_root,
            "detector": detector,
        }

    def test_executable_path_builds_one_deterministic_report(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(Path(directory))
            seen: dict[str, object] = {}

            def load_model(payload, *, device):
                seen["backbone"] = payload["backbone_model"]
                seen["device"] = device
                return object()

            def evaluate(
                model,
                manifest,
                detector_manifest,
                *,
                image_root,
                device,
                benchmark_config,
                detector_id,
                detector_model_sha256,
                detector_config_sha256,
            ):
                seen["model"] = model
                seen["pair_weight"] = benchmark_config.pair_weight
                seen["top_ks"] = benchmark_config.top_ks
                seen["iou_threshold"] = benchmark_config.iou_threshold
                return quality_report(
                    annotations_sha256=manifest.annotations_sha256,
                    predictions_sha256=(
                        detector_manifest.predictions_sha256
                    ),
                    detector_id=detector_id,
                    detector_model_sha256=detector_model_sha256,
                    detector_config_sha256=detector_config_sha256,
                    iou_threshold=benchmark_config.iou_threshold,
                )

            kwargs = {
                "checkpoint_path": paths["checkpoint"],
                "vocabulary_path": paths["vocabulary"],
                "annotations_path": paths["annotations"],
                "image_root": paths["image_root"],
                "detector_predictions_path": paths["detector"],
                "detector_id": "fixture-detector",
                "detector_model_sha256": h("a"),
                "detector_config_sha256": h("b"),
                "iou_threshold": 0.5,
                "top_ks": (20,),
                "pair_weight": 0.75,
                "model_loader": load_model,
                "evaluator": evaluate,
            }
            report_a = run_qualification(**kwargs)
            report_b = run_qualification(**kwargs)

        self.assertEqual(report_a["schema"], REPORT_SCHEMA)
        self.assertEqual(
            stable_json_bytes(report_a),
            stable_json_bytes(report_b),
        )
        self.assertEqual(report_a["evaluation"]["device"], "cpu")
        self.assertFalse(
            report_a["evaluation"][
                "detector_class_labels_enter_relation_inference"
            ]
        )
        self.assertEqual(report_a["evaluation"]["pair_weight"], 0.75)
        self.assertEqual(
            report_a["quality"][
                "directed_pair_recoverability_ceiling"
            ],
            1.0,
        )
        self.assertEqual(seen["device"], torch.device("cpu"))
        self.assertEqual(seen["top_ks"], (20,))
        self.assertEqual(seen["pair_weight"], 0.75)

    def test_predicate_order_mismatch_fails_before_model_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(
                Path(directory),
                checkpoint_predicates=["riding"],
            )
            called = False

            def load_model(payload, *, device):
                nonlocal called
                called = True
                return object()

            with self.assertRaisesRegex(
                ValueError,
                "predicate order",
            ):
                run_qualification(
                    checkpoint_path=paths["checkpoint"],
                    vocabulary_path=paths["vocabulary"],
                    annotations_path=paths["annotations"],
                    image_root=paths["image_root"],
                    detector_predictions_path=paths["detector"],
                    detector_id="fixture-detector",
                    detector_model_sha256=h("a"),
                    detector_config_sha256=h("b"),
                    top_ks=(20,),
                    model_loader=load_model,
                )
            self.assertFalse(called)

    def test_non_apache_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(
                Path(directory),
                config=RelationModelConfig(),
            )
            with self.assertRaisesRegex(
                ValueError,
                "must use Apache",
            ):
                run_qualification(
                    checkpoint_path=paths["checkpoint"],
                    vocabulary_path=paths["vocabulary"],
                    annotations_path=paths["annotations"],
                    image_root=paths["image_root"],
                    detector_predictions_path=paths["detector"],
                    detector_id="fixture-detector",
                    detector_model_sha256=h("a"),
                    detector_config_sha256=h("b"),
                    top_ks=(20,),
                    model_loader=lambda payload, device: object(),
                )

    def test_detector_provenance_requires_lowercase_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(Path(directory))
            with self.assertRaisesRegex(
                ValueError,
                "lowercase SHA-256",
            ):
                run_qualification(
                    checkpoint_path=paths["checkpoint"],
                    vocabulary_path=paths["vocabulary"],
                    annotations_path=paths["annotations"],
                    image_root=paths["image_root"],
                    detector_predictions_path=paths["detector"],
                    detector_id="fixture-detector",
                    detector_model_sha256="A" * 64,
                    detector_config_sha256=h("b"),
                    top_ks=(20,),
                    model_loader=lambda payload, device: object(),
                )


if __name__ == "__main__":
    unittest.main()
