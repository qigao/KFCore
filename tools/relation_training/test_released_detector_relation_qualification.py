from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmark import RelationVocabulary
from detector_ceiling import (
    DetectorPredictionExample,
    DetectorPredictionManifest,
)
from run_released_detector_relation_qualification import (
    _load_bank,
    _prepare_boxes,
    _resize_rgb_like_kfcore,
    _validate_detector_evidence,
    stable_json_bytes,
)


class ReleasedDetectorRelationQualificationTest(unittest.TestCase):
    def test_resize_identity_matches_kfcore_uint8_contract(self):
        rgb = np.asarray(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        resized = _resize_rgb_like_kfcore(
            rgb,
            width=2,
            height=2,
        )
        np.testing.assert_array_equal(resized, rgb)

    def test_resize_upscale_clamps_pixel_centers(self):
        rgb = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
        resized = _resize_rgb_like_kfcore(
            rgb,
            width=3,
            height=2,
        )
        np.testing.assert_array_equal(
            resized,
            np.broadcast_to(rgb, (2, 3, 3)),
        )

    def test_prepare_boxes_uses_normalized_cxcywh_and_padding(self):
        boxes, count = _prepare_boxes(
            (
                (0.0, 0.0, 20.0, 40.0),
                (40.0, 20.0, 80.0, 60.0),
            ),
            100,
            80,
        )
        self.assertEqual(boxes.shape, (1, 32, 4))
        np.testing.assert_allclose(
            boxes[0, 0],
            [0.1, 0.25, 0.2, 0.5],
            rtol=0.0,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            boxes[0, 1],
            [0.6, 0.5, 0.4, 0.5],
            rtol=0.0,
            atol=1.0e-7,
        )
        np.testing.assert_array_equal(
            boxes[0, 2:],
            np.zeros((30, 4), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            count,
            np.asarray([2], dtype=np.int64),
        )

    def test_bank_is_sliced_in_qualification_vocabulary_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.npz"
            W = np.zeros((3, 512), dtype=np.float32)
            W[0, 0] = 1.0
            W[1, 1] = 1.0
            W[2, 2] = 1.0
            alpha = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
            np.savez(
                path,
                names=np.asarray(
                    ["above", "holding", "behind"],
                    dtype=object,
                ),
                W=W,
                alpha=alpha,
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            vocabulary = RelationVocabulary(
                predicates=("behind", "above"),
            )
            with patch(
                "run_released_detector_relation_qualification."
                "PREDICATE_BANK_SHA256",
                digest,
            ):
                selected_W, selected_alpha, report = _load_bank(
                    path,
                    vocabulary,
                )

        self.assertEqual(selected_W.shape, (2, 512))
        self.assertEqual(float(selected_W[0, 2]), 1.0)
        self.assertEqual(float(selected_W[1, 0]), 1.0)
        np.testing.assert_allclose(
            selected_alpha,
            [0.3, 0.1],
            rtol=0.0,
            atol=1.0e-7,
        )
        self.assertEqual(
            report["selected_names"],
            ["behind", "above"],
        )

    def test_detector_evidence_binds_manifest_and_config(self):
        manifest = DetectorPredictionManifest(
            examples=(
                DetectorPredictionExample(
                    image="a.jpg",
                    width=10,
                    height=10,
                    boxes_xyxy=((1.0, 1.0, 5.0, 5.0),),
                    scores=(0.9,),
                ),
            ),
            predictions_sha256="a" * 64,
        )
        evidence = {
            "schema": "kfcore.yolox-detector-box-manifest-evidence/1",
            "detector_id": "yolox",
            "detector_model_sha256": "b" * 64,
            "detector_config_sha256": "c" * 64,
            "manifest_sha256": "a" * 64,
            "class_labels_emitted": False,
        }
        self.assertEqual(
            _validate_detector_evidence(evidence, manifest),
            ("yolox", "b" * 64, "c" * 64),
        )

    def test_stable_json_rejects_nan(self):
        with self.assertRaises(ValueError):
            stable_json_bytes({"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
