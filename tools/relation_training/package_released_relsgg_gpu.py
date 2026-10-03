from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path
from typing import Any

from prepare_released_relsgg import (
    HF_REPOSITORY,
    HF_REVISION,
    ONNX_SHA256,
    PREDICATE_BANK_SHA256,
    TRAINING_GIT_SHA,
    sha256_file,
)


PACKAGE_SCHEMA = "kfcore.model/1"
EVIDENCE_SCHEMA = "kfcore.released-relsgg-gpu-package/1"


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def current_platform() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        arch = "x86_64"
    elif machine in {"arm64", "aarch64"}:
        arch = "aarch64"
    else:
        raise ValueError(f"unsupported GPU package architecture: {machine}")
    if system == "linux":
        return f"linux-{arch}"
    if system == "windows":
        return f"windows-{arch}"
    raise ValueError(f"unsupported GPU package platform: {system}")


def validate_runtime_version(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("TensorRT version must be a string")
    parts = value.split(".")
    if len(parts) != 4 or any(not part.isdigit() for part in parts):
        raise ValueError(
            "TensorRT version must use major.minor.patch.build form"
        )
    if int(parts[0]) <= 0:
        raise ValueError("TensorRT major version must be positive")
    return value


def validate_compute_capability(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] <= 0
        or value[1] < 0
    ):
        raise ValueError("compute_capability must be [major, minor]")
    return int(value[0]), int(value[1])


def package(
    *,
    onnx_path: str | Path,
    predicate_bank_path: str | Path,
    engine_path: str | Path,
    engine_metadata_path: str | Path,
    out_dir: str | Path,
    hardware_compatibility: str = "exact-device",
) -> dict[str, object]:
    onnx_file = Path(onnx_path)
    bank_file = Path(predicate_bank_path)
    engine_file = Path(engine_path)
    metadata_file = Path(engine_metadata_path)
    for path in (onnx_file, bank_file, engine_file, metadata_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    if hardware_compatibility not in {
        "exact-device",
        "same-compute-capability",
    }:
        raise ValueError("unsupported hardware compatibility policy")

    onnx_sha = sha256_file(onnx_file)
    bank_sha = sha256_file(bank_file)
    engine_sha = sha256_file(engine_file)
    sidecar_sha = sha256_file(metadata_file)
    if onnx_sha != ONNX_SHA256:
        raise ValueError("released ONNX SHA-256 differs from pinned artifact")
    if bank_sha != PREDICATE_BANK_SHA256:
        raise ValueError("released predicate bank SHA-256 differs from pinned artifact")

    metadata = load_json(metadata_file)
    if metadata.get("schema") != "kfcore.relation-onnx/2":
        raise ValueError("TensorRT engine metadata uses unsupported source schema")
    if metadata.get("model_type") != "relation.open-vocabulary":
        raise ValueError("TensorRT engine metadata has wrong model_type")
    if metadata.get("onnx_sha256") != onnx_sha:
        raise ValueError("TensorRT source metadata ONNX SHA differs")

    tensorrt = metadata.get("tensorrt")
    if not isinstance(tensorrt, dict):
        raise ValueError("TensorRT engine metadata is missing tensorrt block")
    if tensorrt.get("source_onnx_sha256") != onnx_sha:
        raise ValueError("TensorRT sidecar source ONNX SHA differs")
    if metadata.get("tensorrt_engine_sha256") != engine_sha:
        raise ValueError("TensorRT engine SHA differs from sidecar")

    runtime_version = validate_runtime_version(tensorrt.get("version"))
    cc_major, cc_minor = validate_compute_capability(
        tensorrt.get("compute_capability")
    )
    gpu = tensorrt.get("gpu")
    precision = tensorrt.get("precision")
    profiles = tensorrt.get("profiles")
    if not isinstance(gpu, str) or not gpu:
        raise ValueError("TensorRT GPU name is missing")
    if not isinstance(precision, str) or not precision:
        raise ValueError("TensorRT precision is missing")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("TensorRT optimization profiles are missing")

    version_parts = [int(value) for value in runtime_version.split(".")]
    if hardware_compatibility == "same-compute-capability":
        if version_parts[0] < 10 or (
            version_parts[0] == 10 and version_parts[1] < 9
        ):
            raise ValueError(
                "same-compute-capability requires TensorRT >= 10.9"
            )

    root = Path(out_dir)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    shutil.copyfile(onnx_file, root / "relateanything.onnx")
    shutil.copyfile(bank_file, root / "predicate_bank.npz")
    shutil.copyfile(engine_file, root / "relation.engine")
    shutil.copyfile(metadata_file, root / "tensorrt.json")

    profile_text = json.dumps(
        profiles,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    package_payload = {
        "schema": PACKAGE_SCHEMA,
        "id": "upstream-relsgg-vits16plus-gpu",
        "version": HF_REVISION,
        "model_type": "relation.open-vocabulary",
        "artifacts": [
            {
                "id": "ort-cpu",
                "format": "onnx",
                "path": "relateanything.onnx",
                "sha256": onnx_sha,
                "backend": "onnxruntime",
                "device": "cpu",
            },
            {
                "id": "ort-cuda",
                "format": "onnx",
                "path": "relateanything.onnx",
                "sha256": onnx_sha,
                "backend": "onnxruntime",
                "device": "cuda",
            },
            {
                "id": "tensorrt-cuda",
                "format": "tensorrt-engine",
                "path": "relation.engine",
                "sha256": engine_sha,
                "backend": "tensorrt",
                "device": "cuda",
                "source_artifact": "ort-cuda",
                "source_sha256": onnx_sha,
                "runtime_version": runtime_version,
                "platform": current_platform(),
                "hardware_compatibility": hardware_compatibility,
                "device_name": gpu,
                "compute_capability": f"{cc_major}.{cc_minor}",
                "precision": precision,
                "profile": profile_text,
            },
        ],
    }
    (root / "model.json").write_text(
        json.dumps(package_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "source": {
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "training_git_sha": TRAINING_GIT_SHA,
            "weight_license": "DINOv3 license",
        },
        "artifacts": {
            "onnx_sha256": onnx_sha,
            "predicate_bank_sha256": bank_sha,
            "engine_sha256": engine_sha,
            "engine_sidecar_sha256": sidecar_sha,
            "model_package_sha256": sha256_file(root / "model.json"),
        },
        "runtime": {
            "tensorrt_version": runtime_version,
            "platform": current_platform(),
            "gpu": gpu,
            "compute_capability": f"{cc_major}.{cc_minor}",
            "precision": precision,
            "hardware_compatibility": hardware_compatibility,
            "profiles": profiles,
        },
        "available_routes": [
            {"backend": "onnxruntime", "device": "cpu"},
            {"backend": "onnxruntime", "device": "cuda"},
            {"backend": "tensorrt", "device": "cuda"},
        ],
        "training_reproduction_complete": False,
    }
    (root / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Package the pinned released RelateAnything ONNX plus a real "
            "TensorRT engine as one production KFCore ModelPackage."
        )
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--predicate-bank", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--engine-metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--hardware-compatibility",
        choices=("exact-device", "same-compute-capability"),
        default="exact-device",
    )
    args = parser.parse_args()
    report = package(
        onnx_path=args.onnx,
        predicate_bank_path=args.predicate_bank,
        engine_path=args.engine,
        engine_metadata_path=args.engine_metadata,
        out_dir=args.out_dir,
        hardware_compatibility=args.hardware_compatibility,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
