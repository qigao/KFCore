from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from benchmark import (
    BenchmarkConfig,
    DatasetManifest,
    RelationBenchmark,
    RelationExample,
    RelationVocabulary,
    stable_report_json,
)


class RelationBenchmarkSchemaTest(unittest.TestCase):
    def test_manifest_validates_and_hashes_canonical_examples(self):
        vocabulary = RelationVocabulary(
            predicates=("beside", "holding", "riding"),
            object_labels=("person", "bicycle"),
        )
        payload = {
            "image": "images/frame-001.jpg",
            "width": 100,
            "height": 80,
            "boxes_xyxy": [
                [0, 0, 20, 40],
                [30, 10, 70, 60],
            ],
            "object_labels": ["person", "bicycle"],
            "relations": [
                [0, 2, 1],
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.jsonl"
            path.write_text(
                json.dumps(payload, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            manifest = DatasetManifest.load(path, vocabulary)

        self.assertEqual(len(manifest.examples), 1)
        self.assertEqual(
            manifest.examples[0].relations,
            ((0, 2, 1),),
        )
        self.assertEqual(len(manifest.annotations_sha256), 64)
        self.assertEqual(manifest.vocabulary_sha256, vocabulary.sha256())

    def test_schema_rejects_path_escape_duplicate_relation_and_bad_box(self):
        vocabulary = RelationVocabulary(predicates=("beside",))
        base = {
            "image": "frame.jpg",
            "width": 100,
            "height": 100,
            "boxes_xyxy": [
                [0, 0, 20, 20],
                [30, 30, 60, 60],
            ],
            "object_labels": ["person", "object"],
            "relations": [[0, 0, 1]],
        }

        escaped = dict(base)
        escaped["image"] = "../frame.jpg"
        with self.assertRaises(ValueError):
            RelationExample.from_payload(escaped, vocabulary)

        duplicate = dict(base)
        duplicate["relations"] = [[0, 0, 1], [0, 0, 1]]
        with self.assertRaises(ValueError):
            RelationExample.from_payload(duplicate, vocabulary)

        bad_box = dict(base)
        bad_box["boxes_xyxy"] = [
            [0, 0, 20, 20],
            [30, 30, 130, 60],
        ]
        with self.assertRaises(ValueError):
            RelationExample.from_payload(bad_box, vocabulary)


class RelationBenchmarkMetricTest(unittest.TestCase):
    @staticmethod
    def outputs_with_duplicate_positive():
        pred_logits = torch.tensor(
            [
                [
                    [5.0, 0.0, 0.0],
                    [4.5, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        pair_logits = torch.tensor([[2.0, 1.5, 1.0]])
        sub_idx = torch.tensor([[0, 0, 2]], dtype=torch.int64)
        obj_idx = torch.tensor([[1, 1, 1]], dtype=torch.int64)
        valid_mask = torch.tensor([[True, True, True]])
        return pred_logits, pair_logits, sub_idx, obj_idx, valid_mask

    def test_missed_pair_reduces_sampler_recall_pair_ap_and_triplet_recall(self):
        benchmark = RelationBenchmark(
            predicate_count=3,
            config=BenchmarkConfig(top_ks=(1, 3), pair_weight=1.0),
        )
        benchmark.add(
            self.outputs_with_duplicate_positive(),
            relations=[
                (0, 0, 1),
                (1, 2, 0),
            ],
        )
        report = benchmark.report(
            annotations_sha256="a" * 64,
            vocabulary_sha256="b" * 64,
        )

        self.assertAlmostEqual(report["sampler_recall"], 0.5)
        self.assertAlmostEqual(report["pair_ap"], 0.5)
        self.assertAlmostEqual(
            report["predicate_top1_accuracy_on_sampled_pairs"], 1.0
        )
        self.assertAlmostEqual(report["recall_at_k"]["1"], 0.5)
        self.assertAlmostEqual(report["recall_at_k"]["3"], 0.5)
        self.assertAlmostEqual(report["mean_recall_at_k"]["1"], 0.5)
        self.assertEqual(report["ground_truth_pairs"], 2)
        self.assertEqual(report["sampled_ground_truth_pairs"], 1)

    def test_duplicate_predictions_cannot_inflate_recall_or_pair_ap(self):
        duplicate = RelationBenchmark(
            predicate_count=3,
            config=BenchmarkConfig(top_ks=(1, 3)),
        )
        duplicate.add(
            self.outputs_with_duplicate_positive(),
            relations=[(0, 0, 1)],
        )

        single = RelationBenchmark(
            predicate_count=3,
            config=BenchmarkConfig(top_ks=(1, 3)),
        )
        pred_logits, pair_logits, sub_idx, obj_idx, valid_mask = (
            self.outputs_with_duplicate_positive()
        )
        single.add(
            (
                pred_logits[:, (0, 2), :],
                pair_logits[:, (0, 2)],
                sub_idx[:, (0, 2)],
                obj_idx[:, (0, 2)],
                valid_mask[:, (0, 2)],
            ),
            relations=[(0, 0, 1)],
        )

        duplicate_report = duplicate.report()
        single_report = single.report()
        self.assertEqual(
            duplicate_report["sampler_recall"],
            single_report["sampler_recall"],
        )
        self.assertEqual(
            duplicate_report["pair_ap"],
            single_report["pair_ap"],
        )
        self.assertEqual(
            duplicate_report["recall_at_k"],
            single_report["recall_at_k"],
        )

    def test_mean_recall_excludes_predicates_without_support(self):
        benchmark = RelationBenchmark(
            predicate_count=3,
            config=BenchmarkConfig(top_ks=(1,)),
        )
        outputs = (
            torch.tensor([[[3.0, 0.0, 0.0]]]),
            torch.tensor([[2.0]]),
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([[1]], dtype=torch.int64),
            torch.tensor([[True]]),
        )
        benchmark.add(outputs, relations=[(0, 0, 1)])
        report = benchmark.report()

        self.assertEqual(report["mean_recall_at_k"]["1"], 1.0)
        self.assertEqual(report["per_predicate"][0]["support"], 1)
        self.assertEqual(report["per_predicate"][1]["support"], 0)
        self.assertIsNone(report["per_predicate"][1]["recall"]["1"])
        self.assertIsNone(report["per_predicate"][2]["recall"]["1"])

    def test_seen_and_zero_shot_mean_recall_are_reported_separately(self):
        benchmark = RelationBenchmark(
            predicate_count=3,
            config=BenchmarkConfig(top_ks=(1, 3)),
        )
        outputs = (
            torch.tensor(
                [[
                    [5.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                ]]
            ),
            torch.tensor([[2.0, 1.0]]),
            torch.tensor([[0, 1]], dtype=torch.int64),
            torch.tensor([[1, 0]], dtype=torch.int64),
            torch.tensor([[True, True]]),
        )
        benchmark.add(
            outputs,
            relations=[
                (0, 0, 1),
                (1, 1, 0),
            ],
        )
        report = benchmark.report(
            train_predicate_support=(10, 0, 0),
        )

        self.assertEqual(
            report["mean_recall_at_k"]["1"],
            0.5,
        )
        groups = report["predicate_groups"]
        self.assertEqual(
            groups["seen"]["predicate_indices"],
            [0],
        )
        self.assertEqual(
            groups["train_zero_support"]["predicate_indices"],
            [1],
        )
        self.assertEqual(
            groups["seen"]["validation_triplet_support"],
            1,
        )
        self.assertEqual(
            groups["train_zero_support"]["validation_triplet_support"],
            1,
        )
        self.assertEqual(
            groups["seen"]["mean_recall_at_k"]["1"],
            1.0,
        )
        self.assertEqual(
            groups["train_zero_support"]["mean_recall_at_k"]["1"],
            0.0,
        )
        self.assertEqual(
            groups["train_zero_support"]["mean_recall_at_k"]["3"],
            1.0,
        )
        # Predicate 2 has no validation support and belongs to neither group.
        self.assertNotIn(
            2,
            groups["seen"]["predicate_indices"],
        )
        self.assertNotIn(
            2,
            groups["train_zero_support"]["predicate_indices"],
        )

    def test_predicate_group_with_no_validation_support_reports_null(self):
        benchmark = RelationBenchmark(
            predicate_count=2,
            config=BenchmarkConfig(top_ks=(1,)),
        )
        outputs = (
            torch.tensor([[[3.0, 0.0]]]),
            torch.tensor([[2.0]]),
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([[1]], dtype=torch.int64),
            torch.tensor([[True]]),
        )
        benchmark.add(outputs, relations=[(0, 0, 1)])
        report = benchmark.report(
            train_predicate_support=(5, 0),
        )
        zero = report["predicate_groups"]["train_zero_support"]
        self.assertEqual(zero["predicate_indices"], [])
        self.assertEqual(zero["predicate_count"], 0)
        self.assertEqual(zero["validation_triplet_support"], 0)
        self.assertIsNone(zero["mean_recall_at_k"]["1"])

    def test_invalid_train_predicate_support_fails_fast(self):
        benchmark = RelationBenchmark(
            predicate_count=2,
            config=BenchmarkConfig(top_ks=(1,)),
        )
        outputs = (
            torch.tensor([[[2.0, 0.0]]]),
            torch.tensor([[1.0]]),
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([[1]], dtype=torch.int64),
            torch.tensor([[True]]),
        )
        benchmark.add(outputs, relations=[(0, 0, 1)])

        with self.assertRaises(ValueError):
            benchmark.report(train_predicate_support=(1,))
        with self.assertRaises(ValueError):
            benchmark.report(train_predicate_support=(1, -1))
        with self.assertRaises(ValueError):
            benchmark.report(train_predicate_support=(1, True))

    def test_report_json_is_deterministic_and_strict(self):
        benchmark = RelationBenchmark(
            predicate_count=1,
            config=BenchmarkConfig(top_ks=(1,)),
        )
        outputs = (
            torch.tensor([[[2.0]]]),
            torch.tensor([[1.0]]),
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([[1]], dtype=torch.int64),
            torch.tensor([[True]]),
        )
        benchmark.add(outputs, relations=[(0, 0, 1)])
        report = benchmark.report(
            annotations_sha256="a" * 64,
            vocabulary_sha256="b" * 64,
        )
        first = stable_report_json(report)
        second = stable_report_json(report)

        self.assertEqual(first, second)
        self.assertTrue(first.endswith("\n"))
        self.assertEqual(json.loads(first)["schema"], "kfcore.relation-benchmark/1")


if __name__ == "__main__":
    unittest.main()
