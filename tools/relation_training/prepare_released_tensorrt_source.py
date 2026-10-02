from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_released_relsgg import (
    HF_REPOSITORY,
    HF_REVISION,
    ONNX_SHA256,
    PREDICATE_BANK_SHA256,
    TRAINING_GIT_SHA,
    sha256_file,
)


SCHEMA = "kfcore.relation-onnx/2"


def prepare(
    *,
    onnx_path: str | Path,
    predicate_bank_path: str | Path,
    out_path: str | Path,
) -> dict[str, object]:
    onnx_file = Path(onnx_path)
    bank_file = Path(predicate_bank_path)
    output = Path(out_path)
    if not onnx_file.is_file():
        raise FileNotFoundError(onnx_file)
    if not bank_file.is_file():
        raise FileNotFoundError(bank_file)
    if output.exists():
        raise FileExistsError(output)

    if sha256_file(onnx_file) != ONNX_SHA256:
        raise ValueError("released ONNX SHA-256 differs from pinned artifact")
    if sha256_file(bank_file) != PREDICATE_BANK_SHA256:
        raise ValueError("released predicate bank SHA-256 differs from pinned artifact")

    with np.load(bank_file, allow_pickle=True) as bank:
        if "names" not in bank.files or "default" not in bank.files:
            raise ValueError("released predicate bank lacks names/default")
        names = [str(value) for value in bank["names"]]
        default = [str(value) for value in bank["default"]]
    if not names or not default:
        raise ValueError("released predicate bank is empty")
    if any(name not in set(names) for name in default):
        raise ValueError("released default vocabulary is outside predicate bank")

    payload = {
        "schema": SCHEMA,
        "model_type": "relation.open-vocabulary",
        "output_kind": "dynamic-vocabulary-logits",
        "onnx_sha256": ONNX_SHA256,
        "vocabulary_dynamic": True,
        "vocabulary_graph_input": True,
        "image_size": 448,
        "max_boxes": 32,
        "final_budget": 128,
        "query_dim": 512,
        "default_predicate_count": len(default),
        "released_predicate_bank_size": len(names),
        "source": {
            "kind": "upstream-released-deployment",
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "training_git_sha": TRAINING_GIT_SHA,
            "predicate_bank_sha256": PREDICATE_BANK_SHA256,
            "weight_license": "DINOv3 license",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a KFCore TensorRT-builder metadata sidecar for the pinned "
            "upstream released relsgg-vits16plus dynamic W/alpha ONNX."
        )
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--predicate-bank", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = prepare(
        onnx_path=args.onnx,
        predicate_bank_path=args.predicate_bank,
        out_path=args.out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
