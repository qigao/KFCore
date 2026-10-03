from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


HF_REPOSITORY = "maelic/relsgg-vits16plus"
HF_REVISION = "2db90096be5217bdc7a9003c042950f45723d105"
TRAINING_GIT_SHA = "e9ea42aed60f766f12ad19d51709129c50110a3b"
ONNX_SHA256 = "b8b6a047c5e0771a897a5015c2ffb09d8fe5e3ffa0651e1e61af0e8436a3617a"
PREDICATE_BANK_SHA256 = (
    "708f812d6579eab1be85b3378b79e376"
    "453c2d4f65b4fad9bcebb9f5bf05da06"
)
EVIDENCE_SCHEMA = "kfcore.released-relsgg-vits16plus/1"
PACKAGE_SCHEMA = "kfcore.model/1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value_info: onnx.ValueInfoProto) -> list[int]:
    result: list[int] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        else:
            result.append(-1)
    return result


def _validate_metadata(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("relateanything.json must be an object")
    expected = {
        "img_size": 448,
        "max_boxes": 32,
        "final_budget": 128,
        "vocab_mode": "input",
        "text_dim": 512,
        "output_kind": "logits",
        "score_contract": "sigmoid(a * (pred_logit + w * pair_logit) + b)",
        "run_name": "relsgg-vits16plus",
        "git_sha": TRAINING_GIT_SHA,
        "epoch": 12,
        "opset": 17,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"released metadata differs for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    outputs = payload.get("outputs")
    if outputs != [
        "pred_logits",
        "pair_logits",
        "sub_idx",
        "obj_idx",
        "valid_mask",
    ]:
        raise ValueError("released metadata output contract drifted")
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("released metadata calibration is missing")
    if calibration.get("a") != 0.5651 or calibration.get("b") != -1.9623:
        raise ValueError("released metadata calibration drifted")
    return dict(payload)


def _validate_onnx(path: Path) -> dict[str, object]:
    model = onnx.load(path, load_external_data=False)
    if not model.opset_import or int(model.opset_import[0].version) != 17:
        raise ValueError("released ONNX opset drifted")
    inputs = {
        item.name: (
            item.type.tensor_type.elem_type,
            _shape(item),
        )
        for item in model.graph.input
    }
    outputs = {
        item.name: (
            item.type.tensor_type.elem_type,
            _shape(item),
        )
        for item in model.graph.output
    }

    expected_inputs = {
        "image": (TensorProto.FLOAT, [-1, 3, 448, 448]),
        "boxes": (TensorProto.FLOAT, [-1, -1, 4]),
        "box_counts": (TensorProto.INT64, [-1]),
        "W": (TensorProto.FLOAT, [-1, 512]),
        "alpha": (TensorProto.FLOAT, [-1]),
    }
    expected_outputs = {
        "pred_logits": (TensorProto.FLOAT, [-1, -1, -1]),
        "pair_logits": (TensorProto.FLOAT, [-1, -1]),
        "sub_idx": (TensorProto.INT64, [-1, -1]),
        "obj_idx": (TensorProto.INT64, [-1, -1]),
        "valid_mask": (TensorProto.BOOL, [-1, -1]),
    }
    if inputs != expected_inputs:
        raise ValueError(
            f"released ONNX input contract drifted: {inputs}"
        )
    if outputs != expected_outputs:
        raise ValueError(
            f"released ONNX output contract drifted: {outputs}"
        )
    return {
        "inputs": {
            key: shape
            for key, (_, shape) in sorted(inputs.items())
        },
        "outputs": {
            key: shape
            for key, (_, shape) in sorted(outputs.items())
        },
        "opset": 17,
    }


def prepare(
    *,
    onnx_path: str | Path,
    predicate_bank_path: str | Path,
    metadata_path: str | Path,
    out_dir: str | Path,
) -> dict[str, object]:
    onnx_file = Path(onnx_path)
    bank_file = Path(predicate_bank_path)
    metadata_file = Path(metadata_path)
    for path in (onnx_file, bank_file, metadata_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    onnx_sha = sha256_file(onnx_file)
    bank_sha = sha256_file(bank_file)
    if onnx_sha != ONNX_SHA256:
        raise ValueError("released ONNX SHA-256 differs from pinned artifact")
    if bank_sha != PREDICATE_BANK_SHA256:
        raise ValueError(
            "released predicate bank SHA-256 differs from pinned artifact"
        )

    metadata = _validate_metadata(
        json.loads(metadata_file.read_text(encoding="utf-8"))
    )
    contract = _validate_onnx(onnx_file)

    root = Path(out_dir)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(onnx_file, root / "relateanything.onnx")
    shutil.copyfile(bank_file, root / "predicate_bank.npz")
    shutil.copyfile(metadata_file, root / "relateanything.json")

    package = {
        "schema": PACKAGE_SCHEMA,
        "id": "upstream-relsgg-vits16plus",
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
        ],
    }
    (root / "model.json").write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema": EVIDENCE_SCHEMA,
        "source": {
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "training_git_sha": TRAINING_GIT_SHA,
            "weight_license": "DINOv3 license",
            "claim": "upstream-released-deployment-artifact",
        },
        "artifacts": {
            "onnx_sha256": onnx_sha,
            "predicate_bank_sha256": bank_sha,
            "metadata_sha256": sha256_file(metadata_file),
            "package_sha256": sha256_file(root / "model.json"),
        },
        "deployment": {
            "backends": {
                "onnxruntime_cpu": True,
                "onnxruntime_cuda": True,
                "tensorrt": False,
            },
            "image_size": metadata["img_size"],
            "max_boxes": metadata["max_boxes"],
            "final_budget": metadata["final_budget"],
            "text_dim": metadata["text_dim"],
            "calibration": metadata["calibration"],
            "score_contract": metadata["score_contract"],
            "outputs": metadata["outputs"],
            "onnx_contract": contract,
        },
        "training_reproduction_complete": False,
    }
    (root / "evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--predicate-bank", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    report = prepare(
        onnx_path=args.onnx,
        predicate_bank_path=args.predicate_bank,
        metadata_path=args.metadata,
        out_dir=args.out_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
