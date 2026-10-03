from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from prepare_yolox_tiny import (
    CLASS_COUNT,
    EVIDENCE_SCHEMA,
    INPUT_SIZE,
    STRIDES,
    append_yolox_decode,
    prepare,
    sha256_file,
)


def make_source(path: Path) -> None:
    candidate_count = sum(
        (INPUT_SIZE[0] // stride) * (INPUT_SIZE[1] // stride)
        for stride in STRIDES
    )
    channels = 5 + CLASS_COUNT
    inputs = [
        helper.make_tensor_value_info(
            "images",
            TensorProto.FLOAT,
            [1, 3, INPUT_SIZE[0], INPUT_SIZE[1]],
        )
    ]
    outputs = [
        helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT,
            [1, candidate_count, channels],
        )
    ]
    initializers = [
        numpy_helper.from_array(
            np.asarray([0.0], dtype=np.float32),
            name="zero",
        ),
        numpy_helper.from_array(
            np.asarray(
                [1, candidate_count, channels],
                dtype=np.int64,
            ),
            name="shape",
        ),
    ]
    nodes = [
        helper.make_node(
            "Expand",
            ["zero", "shape"],
            ["output"],
        )
    ]
    graph = helper.make_graph(
        nodes,
        "synthetic-yolox-tiny",
        inputs,
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
        producer_name="kfcore-test",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


class PrepareYoloXTinyTest(unittest.TestCase):
    def test_decode_graph_executes_official_grid_stride_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.onnx"
            make_source(source_path)
            source = onnx.load(source_path)
            decoded, evidence = append_yolox_decode(source)
            onnx.checker.check_model(decoded)

            decoded_path = Path(directory) / "decoded.onnx"
            onnx.save(decoded, decoded_path)
            session = ort.InferenceSession(
                str(decoded_path),
                providers=["CPUExecutionProvider"],
            )
            output = session.run(
                None,
                {
                    "images": np.zeros(
                        (1, 3, INPUT_SIZE[0], INPUT_SIZE[1]),
                        dtype=np.float32,
                    )
                },
            )[0]

        self.assertEqual(output.shape, (1, 3549, 85))
        np.testing.assert_allclose(
            output[0, 0, :5],
            np.asarray([0.0, 0.0, 8.0, 8.0, 0.0], dtype=np.float32),
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            output[0, 1, :4],
            np.asarray([8.0, 0.0, 8.0, 8.0], dtype=np.float32),
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertEqual(evidence["candidate_count"], 3549)
        self.assertEqual(evidence["class_count"], 80)
        self.assertEqual(evidence["strides"], [8, 16, 32])

    def test_prepare_is_byte_deterministic_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "yolox_tiny.onnx"
            make_source(source_path)
            source_sha = sha256_file(source_path)

            first = prepare(
                source_path,
                root / "first",
                expected_source_sha256=source_sha,
            )
            second = prepare(
                source_path,
                root / "second",
                expected_source_sha256=source_sha,
            )

            self.assertEqual(first["schema"], EVIDENCE_SCHEMA)
            self.assertEqual(
                first["upstream"]["commit"],
                "e1052df71842031413f6030723c3607b839c80ce",
            )
            self.assertEqual(
                first["upstream"]["source_sha256"],
                source_sha,
            )
            self.assertEqual(
                first["artifact"]["flavor"],
                "raw-yolox",
            )
            self.assertEqual(
                first["artifact"]["decoded_onnx_sha256"],
                second["artifact"]["decoded_onnx_sha256"],
            )
            self.assertEqual(
                first["artifact"]["package_sha256"],
                second["artifact"]["package_sha256"],
            )
            self.assertEqual(
                (root / "first" / "yolox_tiny_decoded.onnx").read_bytes(),
                (root / "second" / "yolox_tiny_decoded.onnx").read_bytes(),
            )
            self.assertEqual(
                (root / "first" / "model.json").read_bytes(),
                (root / "second" / "model.json").read_bytes(),
            )

    def test_source_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "yolox_tiny.onnx"
            make_source(source_path)

            with self.assertRaisesRegex(
                ValueError,
                "SHA-256 differs",
            ):
                prepare(
                    source_path,
                    root / "out",
                    expected_source_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
