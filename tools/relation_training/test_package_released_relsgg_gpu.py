from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from package_released_relsgg_gpu import package


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleasedRelSggGpuPackageTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        onnx = root / "relateanything.onnx"
        onnx.write_bytes(b"released-onnx-fixture")
        bank = root / "predicate_bank.npz"
        bank.write_bytes(b"predicate-bank-fixture")
        engine = root / "relation.engine"
        engine.write_bytes(b"tensorrt-engine-fixture")
        metadata = root / "relation.json"
        metadata.write_text(
            json.dumps(
                {
                    "schema": "kfcore.relation-onnx/2",
                    "model_type": "relation.open-vocabulary",
                    "onnx_sha256": sha256(onnx),
                    "tensorrt_engine_sha256": sha256(engine),
                    "tensorrt": {
                        "version": "10.13.3.9",
                        "precision": "fp32",
                        "gpu": "Fixture NVIDIA GPU",
                        "compute_capability": [9, 0],
                        "source_onnx_sha256": sha256(onnx),
                        "profiles": {
                            "W": {
                                "min": [1, 512],
                                "opt": [64, 512],
                                "max": [243, 512],
                            },
                            "alpha": {
                                "min": [1],
                                "opt": [64],
                                "max": [243],
                            },
                        },
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return onnx, bank, engine, metadata

    def test_packages_cpu_cuda_and_tensorrt_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx, bank, engine, metadata = self.make_fixture(root)
            out = root / "package"
            with (
                patch(
                    "package_released_relsgg_gpu.ONNX_SHA256",
                    sha256(onnx),
                ),
                patch(
                    "package_released_relsgg_gpu.PREDICATE_BANK_SHA256",
                    sha256(bank),
                ),
            ):
                evidence = package(
                    onnx_path=onnx,
                    predicate_bank_path=bank,
                    engine_path=engine,
                    engine_metadata_path=metadata,
                    out_dir=out,
                )

            manifest = json.loads((out / "model.json").read_text())
            self.assertEqual(
                [(row["backend"], row["device"]) for row in manifest["artifacts"]],
                [
                    ("onnxruntime", "cpu"),
                    ("onnxruntime", "cuda"),
                    ("tensorrt", "cuda"),
                ],
            )
            trt = manifest["artifacts"][2]
            self.assertEqual(trt["format"], "tensorrt-engine")
            self.assertEqual(trt["source_artifact"], "ort-cuda")
            self.assertEqual(trt["source_sha256"], sha256(onnx))
            self.assertEqual(trt["runtime_version"], "10.13.3.9")
            self.assertEqual(trt["device_name"], "Fixture NVIDIA GPU")
            self.assertEqual(trt["compute_capability"], "9.0")
            self.assertEqual(trt["precision"], "fp32")
            self.assertEqual(
                evidence["available_routes"],
                [
                    {"backend": "onnxruntime", "device": "cpu"},
                    {"backend": "onnxruntime", "device": "cuda"},
                    {"backend": "tensorrt", "device": "cuda"},
                ],
            )
            self.assertTrue((out / "relation.engine").is_file())
            self.assertTrue((out / "tensorrt.json").is_file())

    def test_rejects_engine_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx, bank, engine, metadata = self.make_fixture(root)
            payload = json.loads(metadata.read_text())
            payload["tensorrt_engine_sha256"] = "a" * 64
            metadata.write_text(json.dumps(payload), encoding="utf-8")
            with (
                patch(
                    "package_released_relsgg_gpu.ONNX_SHA256",
                    sha256(onnx),
                ),
                patch(
                    "package_released_relsgg_gpu.PREDICATE_BANK_SHA256",
                    sha256(bank),
                ),
                self.assertRaisesRegex(ValueError, "engine SHA differs"),
            ):
                package(
                    onnx_path=onnx,
                    predicate_bank_path=bank,
                    engine_path=engine,
                    engine_metadata_path=metadata,
                    out_dir=root / "package",
                )

    def test_same_compute_capability_requires_tensorrt_10_9(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx, bank, engine, metadata = self.make_fixture(root)
            payload = json.loads(metadata.read_text())
            payload["tensorrt"]["version"] = "10.8.0.0"
            metadata.write_text(json.dumps(payload), encoding="utf-8")
            with (
                patch(
                    "package_released_relsgg_gpu.ONNX_SHA256",
                    sha256(onnx),
                ),
                patch(
                    "package_released_relsgg_gpu.PREDICATE_BANK_SHA256",
                    sha256(bank),
                ),
                self.assertRaisesRegex(ValueError, "TensorRT >= 10.9"),
            ):
                package(
                    onnx_path=onnx,
                    predicate_bank_path=bank,
                    engine_path=engine,
                    engine_metadata_path=metadata,
                    out_dir=root / "package",
                    hardware_compatibility="same-compute-capability",
                )


if __name__ == "__main__":
    unittest.main()
