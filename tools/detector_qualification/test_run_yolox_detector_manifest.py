from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_yolox_detector_manifest import (
    _decode,
    _letterbox_nchw,
    _load_annotation_images,
)


class YoloXDetectorManifestTest(unittest.TestCase):
    def test_letterbox_matches_top_left_rgb_normalization_contract(self):
        rgb = np.asarray(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )
        tensor, scale = _letterbox_nchw(rgb)
        self.assertEqual(tensor.shape, (1, 3, 416, 416))
        self.assertEqual(scale, 208.0)
        expected = (
            np.asarray([255.0, 0.0, 0.0], dtype=np.float32) / 255.0
            - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        np.testing.assert_allclose(
            tensor[0, :, 0, 0],
            expected,
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_decode_uses_objectness_class_score_stable_order_and_class_nms(self):
        output = np.zeros((1, 4, 7), dtype=np.float32)
        # Candidate 0: class 0, score .9 * .9 = .81.
        output[0, 0] = [50, 50, 40, 40, 0.9, 0.9, 0.1]
        # Candidate 1: same class/box neighborhood, lower score -> suppressed.
        output[0, 1] = [51, 51, 40, 40, 0.8, 0.9, 0.1]
        # Candidate 2: different class, overlapping -> retained.
        output[0, 2] = [50, 50, 40, 40, 0.9, 0.1, 0.9]
        # Candidate 3: below threshold.
        output[0, 3] = [150, 150, 20, 20, 0.2, 0.9, 0.1]

        boxes, scores = _decode(
            output,
            source_width=416,
            source_height=416,
            scale=1.0,
            score_threshold=0.25,
            iou_threshold=0.45,
            max_detections=300,
        )
        self.assertEqual(
            boxes,
            [
                [30.0, 30.0, 70.0, 70.0],
                [30.0, 30.0, 70.0, 70.0],
            ],
        )
        self.assertAlmostEqual(scores[0], 0.81, places=6)
        self.assertAlmostEqual(scores[1], 0.81, places=6)

    def test_decode_tie_order_is_source_stable(self):
        output = np.zeros((1, 2, 7), dtype=np.float32)
        output[0, 0] = [20, 20, 10, 10, 0.9, 0.9, 0.1]
        output[0, 1] = [80, 80, 10, 10, 0.9, 0.9, 0.1]
        boxes, _ = _decode(
            output,
            source_width=100,
            source_height=100,
            scale=1.0,
            score_threshold=0.25,
            iou_threshold=0.45,
            max_detections=300,
        )
        self.assertEqual(boxes[0], [15.0, 15.0, 25.0, 25.0])
        self.assertEqual(boxes[1], [75.0, 75.0, 85.0, 85.0])

    def test_annotations_are_unique_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.jsonl"
            path.write_text(
                json.dumps(
                    {"image": "a.jpg", "width": 10, "height": 20}
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _load_annotation_images(path),
                [("a.jpg", 10, 20)],
            )

            path.write_text(
                json.dumps(
                    {"image": "../a.jpg", "width": 10, "height": 20}
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inside image root"):
                _load_annotation_images(path)


if __name__ == "__main__":
    unittest.main()
