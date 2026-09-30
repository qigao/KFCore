from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image
import torch

from benchmark import (
    DatasetManifest,
    RelationExample,
)
from detector_ceiling import (
    DetectorPredictionExample,
    DetectorPredictionManifest,
    DetectorRecoverabilityBenchmark,
    DetectorRecoverabilityConfig,
)
from training import (
    evaluate_detector_boxes,
    prepare_detector_boxes,
)


def gt_example() -> RelationExample:
    return RelationExample(
        image="sample.png",
        width=100,
        height=100,
        boxes_xyxy=(
            (0.0, 0.0, 20.0, 20.0),
            (40.0, 0.0, 60.0, 20.0),
            (80.0, 0.0, 100.0, 20.0),
        ),
        object_labels=("a", "b", "c"),
        relations=(
            (0, 0, 1),
            (1, 1, 2),
        ),
    )


def detector_two_boxes() -> DetectorPredictionExample:
    return DetectorPredictionExample(
        image="sample.png",
        width=100,
        height=100,
        boxes_xyxy=(
            (0.0, 0.0, 20.0, 20.0),
            (40.0, 0.0, 60.0, 20.0),
        ),
        scores=(0.9, 0.8),
    )


def runtime_hit() -> tuple[torch.Tensor, ...]:
    pred = torch.tensor(
        [[[8.0, -8.0], [-8.0, 8.0]]],
        dtype=torch.float32,
    )
    pair = torch.tensor(
        [[4.0, 1.0]],
        dtype=torch.float32,
    )
    sub = torch.tensor(
        [[0, 1]],
        dtype=torch.int64,
    )
    obj = torch.tensor(
        [[1, 0]],
        dtype=torch.int64,
    )
    valid = torch.tensor(
        [[True, True]],
    )
    return pred, pair, sub, obj, valid


class DetectorRecoverabilityBenchmarkTest(unittest.TestCase):
    def test_reports_detector_ceiling_and_conditional_relation_recall(self):
        benchmark = DetectorRecoverabilityBenchmark(
            2,
            DetectorRecoverabilityConfig(
                iou_threshold=0.5,
                top_ks=(1, 2),
            ),
        )
        benchmark.add(
            runtime_hit(),
            gt_example(),
            detector_two_boxes(),
        )
        report = benchmark.report(
            annotations_sha256="gt-sha",
            detector_predictions_sha256="det-sha",
            detector_id="stub-detector",
            detector_model_sha256="model-sha",
            detector_config_sha256="config-sha",
        )

        self.assertEqual(report["ground_truth_objects"], 3)
        self.assertEqual(
            report["recoverable_ground_truth_objects"],
            2,
        )
        self.assertAlmostEqual(
            report["object_recoverability_ceiling"],
            2.0 / 3.0,
        )

        self.assertEqual(
            report["ground_truth_directed_pairs"],
            2,
        )
        self.assertEqual(
            report["recoverable_directed_pairs"],
            1,
        )
        self.assertAlmostEqual(
            report["directed_pair_recoverability_ceiling"],
            0.5,
        )
        self.assertAlmostEqual(
            report[
                "sampler_recall_conditional_on_recoverable_pairs"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            report["sampler_pair_recall_end_to_end"],
            0.5,
        )

        self.assertEqual(
            report["recoverable_ground_truth_triplets"],
            1,
        )
        self.assertAlmostEqual(
            report["conditional_recall_at_k"]["1"],
            1.0,
        )
        self.assertAlmostEqual(
            report["end_to_end_recall_at_k"]["1"],
            0.5,
        )
        self.assertAlmostEqual(
            report["conditional_mean_recall_at_k"]["1"],
            1.0,
        )
        self.assertAlmostEqual(
            report["end_to_end_mean_recall_at_k"]["1"],
            0.5,
        )

        failure = report["failure_decomposition_at_max_k"]
        self.assertEqual(failure["detector_miss"], 1)
        self.assertEqual(failure["sampler_miss"], 0)
        self.assertEqual(failure["predicate_miss"], 0)
        self.assertEqual(failure["recovered"], 1)

    def test_one_detector_box_cannot_recover_both_pair_endpoints(self):
        gt = RelationExample(
            image="overlap.png",
            width=100,
            height=100,
            boxes_xyxy=(
                (10.0, 10.0, 50.0, 50.0),
                (10.0, 10.0, 50.0, 50.0),
            ),
            object_labels=("a", "b"),
            relations=((0, 0, 1),),
        )
        detector = DetectorPredictionExample(
            image="overlap.png",
            width=100,
            height=100,
            boxes_xyxy=((10.0, 10.0, 50.0, 50.0),),
            scores=(0.9,),
        )
        runtime = (
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1, dtype=torch.int64),
            torch.zeros(1, 1, dtype=torch.int64),
            torch.zeros(1, 1, dtype=torch.bool),
        )

        benchmark = DetectorRecoverabilityBenchmark(
            1,
            DetectorRecoverabilityConfig(
                top_ks=(1,),
            ),
        )
        benchmark.add(runtime, gt, detector)
        report = benchmark.report()

        self.assertEqual(
            report["recoverable_ground_truth_objects"],
            2,
        )
        self.assertEqual(
            report["recoverable_directed_pairs"],
            0,
        )
        self.assertEqual(
            report["recoverable_ground_truth_triplets"],
            0,
        )
        self.assertIsNone(
            report["conditional_recall_at_k"]["1"]
        )
        self.assertEqual(
            report["failure_decomposition_at_max_k"][
                "detector_miss"
            ],
            1,
        )

    def test_failure_decomposition_separates_sampler_and_predicate_miss(self):
        detector = DetectorPredictionExample(
            image="sample.png",
            width=100,
            height=100,
            boxes_xyxy=(
                (0.0, 0.0, 20.0, 20.0),
                (40.0, 0.0, 60.0, 20.0),
                (80.0, 0.0, 100.0, 20.0),
            ),
            scores=(0.9, 0.8, 0.7),
        )
        # Pair 0->1 is sampled but predicate 0 is ranked wrong.
        # Pair 1->2 is never sampled.
        runtime = (
            torch.tensor(
                [[[-8.0, 8.0], [8.0, -8.0]]]
            ),
            torch.tensor([[4.0, 1.0]]),
            torch.tensor([[0, 2]], dtype=torch.int64),
            torch.tensor([[1, 1]], dtype=torch.int64),
            torch.tensor([[True, True]]),
        )
        benchmark = DetectorRecoverabilityBenchmark(
            2,
            DetectorRecoverabilityConfig(
                top_ks=(1,),
            ),
        )
        benchmark.add(runtime, gt_example(), detector)
        report = benchmark.report()

        failure = report["failure_decomposition_at_max_k"]
        self.assertEqual(failure["detector_miss"], 0)
        self.assertEqual(failure["sampler_miss"], 1)
        self.assertEqual(failure["predicate_miss"], 1)
        self.assertEqual(failure["recovered"], 0)
        self.assertAlmostEqual(
            report[
                "sampler_recall_conditional_on_recoverable_pairs"
            ],
            0.5,
        )

    def test_detector_limit_is_score_stable(self):
        detector = DetectorPredictionExample(
            image="sample.png",
            width=100,
            height=100,
            boxes_xyxy=(
                (0.0, 0.0, 10.0, 10.0),
                (10.0, 0.0, 20.0, 10.0),
                (20.0, 0.0, 30.0, 10.0),
            ),
            scores=(0.2, 0.9, 0.9),
        )
        limited = detector.limit(2)
        self.assertEqual(
            limited.boxes_xyxy,
            (
                (10.0, 0.0, 20.0, 10.0),
                (20.0, 0.0, 30.0, 10.0),
            ),
        )

    def test_prepare_detector_boxes_uses_detector_geometry_only(self):
        boxes, count = prepare_detector_boxes(
            detector_two_boxes(),
            max_boxes=4,
        )
        self.assertEqual(int(count), 2)
        self.assertTrue(
            torch.allclose(
                boxes[0],
                torch.tensor([0.1, 0.1, 0.2, 0.2]),
            )
        )
        self.assertTrue(
            torch.allclose(
                boxes[1],
                torch.tensor([0.5, 0.1, 0.2, 0.2]),
            )
        )
        self.assertTrue(
            torch.equal(
                boxes[2:],
                torch.zeros_like(boxes[2:]),
            )
        )

    def test_evaluate_detector_boxes_runs_model_on_detector_boxes(self):
        class StubModel:
            def __init__(self) -> None:
                self.config = SimpleNamespace(
                    max_boxes=4,
                    image_size=8,
                )
                self.predicate_bank = torch.zeros(2, 4)
                self.seen_boxes: torch.Tensor | None = None

            def eval(self) -> "StubModel":
                return self

            def __call__(
                self,
                image: torch.Tensor,
                boxes: torch.Tensor,
                box_counts: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                self.seen_boxes = boxes.detach().cpu().clone()
                self.assert_image = image.detach().cpu().clone()
                self.assert_count = box_counts.detach().cpu().clone()
                return runtime_hit()

        gt = gt_example()
        manifest = DatasetManifest(
            examples=(gt,),
            annotations_sha256="gt-sha",
            vocabulary_sha256="vocab-sha",
        )
        detector_manifest = DetectorPredictionManifest(
            examples=(detector_two_boxes(),),
            predictions_sha256="det-sha",
        )
        model = StubModel()

        with tempfile.TemporaryDirectory() as directory:
            Image.new(
                "RGB",
                (100, 100),
                color=(10, 20, 30),
            ).save(Path(directory) / "sample.png")
            report = evaluate_detector_boxes(
                model,  # type: ignore[arg-type]
                manifest,
                detector_manifest,
                image_root=directory,
                device=torch.device("cpu"),
                benchmark_config=DetectorRecoverabilityConfig(
                    top_ks=(1, 2),
                ),
                detector_id="stub",
            )

        self.assertIsNotNone(model.seen_boxes)
        assert model.seen_boxes is not None
        self.assertTrue(
            torch.allclose(
                model.seen_boxes[0, 0],
                torch.tensor([0.1, 0.1, 0.2, 0.2]),
            )
        )
        self.assertTrue(
            torch.allclose(
                model.seen_boxes[0, 1],
                torch.tensor([0.5, 0.1, 0.2, 0.2]),
            )
        )
        self.assertEqual(
            report["detector"]["id"],
            "stub",
        )
        self.assertEqual(
            report["detector_predictions_sha256"],
            "det-sha",
        )


if __name__ == "__main__":
    unittest.main()
