from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from prepare_released_tensorrt_source import prepare


class ReleasedTensorRtSourceTest(unittest.TestCase):
    def test_prepares_dynamic_vocab_builder_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx = root / "relateanything.onnx"
            onnx.write_bytes(b"onnx-fixture")
            bank = root / "predicate_bank.npz"
            np.savez(
                bank,
                names=np.asarray(["holding", "behind", "above"], dtype=object),
                default=np.asarray(["holding", "behind"], dtype=object),
            )
            onnx_sha = hashlib.sha256(onnx.read_bytes()).hexdigest()
            bank_sha = hashlib.sha256(bank.read_bytes()).hexdigest()
            out = root / "source-metadata.json"

            with (
                patch(
                    "prepare_released_tensorrt_source.ONNX_SHA256",
                    onnx_sha,
                ),
                patch(
                    "prepare_released_tensorrt_source.PREDICATE_BANK_SHA256",
                    bank_sha,
                ),
            ):
                report = prepare(
                    onnx_path=onnx,
                    predicate_bank_path=bank,
                    out_path=out,
                )
            self.assertTrue(out.is_file())

        self.assertEqual(report["schema"], "kfcore.relation-onnx/2")
        self.assertEqual(report["model_type"], "relation.open-vocabulary")
        self.assertTrue(report["vocabulary_dynamic"])
        self.assertTrue(report["vocabulary_graph_input"])
        self.assertEqual(report["image_size"], 448)
        self.assertEqual(report["max_boxes"], 32)
        self.assertEqual(report["final_budget"], 128)
        self.assertEqual(report["query_dim"], 512)
        self.assertEqual(report["default_predicate_count"], 2)
        self.assertEqual(report["released_predicate_bank_size"], 3)
        self.assertEqual(
            report["source"]["kind"],
            "upstream-released-deployment",
        )

    def test_rejects_default_outside_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx = root / "relateanything.onnx"
            onnx.write_bytes(b"onnx-fixture")
            bank = root / "predicate_bank.npz"
            np.savez(
                bank,
                names=np.asarray(["holding"], dtype=object),
                default=np.asarray(["missing"], dtype=object),
            )
            with (
                patch(
                    "prepare_released_tensorrt_source.ONNX_SHA256",
                    hashlib.sha256(onnx.read_bytes()).hexdigest(),
                ),
                patch(
                    "prepare_released_tensorrt_source.PREDICATE_BANK_SHA256",
                    hashlib.sha256(bank.read_bytes()).hexdigest(),
                ),
                self.assertRaisesRegex(ValueError, "outside predicate bank"),
            ):
                prepare(
                    onnx_path=onnx,
                    predicate_bank_path=bank,
                    out_path=root / "metadata.json",
                )


if __name__ == "__main__":
    unittest.main()
