from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from apache_text_student import (
    PredicateTextStudent,
    load_apache_checkpoint,
)


INPUT_NAMES = ["input_ids", "padding_mask"]
OUTPUT_NAMES = ["predicate_embedding"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_text_student(
    model: PredicateTextStudent,
    output_path: str | Path,
    input_ids: Tensor,
    padding_mask: Tensor,
    *,
    opset: int = 18,
) -> Tensor:
    if input_ids.ndim != 2 or padding_mask.shape != input_ids.shape:
        raise ValueError(
            "text-student export inputs must be matching [N,L] tensors"
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        reference = model(input_ids, padding_mask)

    torch.onnx.export(
        model,
        (input_ids, padding_mask),
        str(path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "input_ids": {0: "num_predicates"},
            "padding_mask": {0: "num_predicates"},
            "predicate_embedding": {0: "num_predicates"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return reference


def check_text_student_parity(
    output_path: str | Path,
    input_ids: Tensor,
    padding_mask: Tensor,
    reference: Tensor,
    *,
    tolerance: float = 1.0e-4,
) -> float:
    import onnx
    import onnxruntime as ort

    path = Path(output_path)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)

    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    if [item.name for item in session.get_inputs()] != INPUT_NAMES:
        raise RuntimeError("text-student ONNX input names drifted")
    if [item.name for item in session.get_outputs()] != OUTPUT_NAMES:
        raise RuntimeError("text-student ONNX output names drifted")

    actual = session.run(
        OUTPUT_NAMES,
        {
            "input_ids": input_ids.detach().cpu().numpy(),
            "padding_mask": padding_mask.detach().cpu().numpy(),
        },
    )[0]
    expected = reference.detach().cpu().numpy()
    delta = float(np.max(np.abs(expected - actual)))
    if delta > tolerance:
        raise RuntimeError(
            f"text-student ONNX parity failed: max abs delta {delta}"
        )
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export an Apache-reference PredicateTextStudent checkpoint "
            "to a KFCore runtime ONNX contract."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.out)
    model = load_apache_checkpoint(str(checkpoint))

    length = model.config.max_length
    input_ids = torch.zeros(
        (2, length),
        dtype=torch.int64,
    )
    input_ids[0, :4] = torch.tensor([1, 2, 3, 4])
    input_ids[1, :3] = torch.tensor([5, 6, 7])
    padding_mask = input_ids == model.PAD_ID

    reference = export_text_student(
        model,
        output,
        input_ids,
        padding_mask,
        opset=args.opset,
    )
    metadata = {
        "schema": "kfcore.predicate-text-student-onnx/1",
        "model_type": "relation.predicate-text-encoder",
        "checkpoint_sha256": sha256(checkpoint),
        "onnx_sha256": sha256(output),
        "vocab_size": model.config.vocab_size,
        "token_dim": model.config.token_dim,
        "model_dim": model.config.model_dim,
        "depth": model.config.depth,
        "heads": model.config.heads,
        "ffn_dim": model.config.ffn_dim,
        "output_dim": model.config.output_dim,
        "max_length": model.config.max_length,
        "output_normalized": True,
        "opset": args.opset,
    }
    if args.check:
        metadata["check_max_abs_delta"] = (
            check_text_student_parity(
                output,
                input_ids,
                padding_mask,
                reference,
            )
        )

    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
