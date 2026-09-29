from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from checkpoint import config_from_payload, load_payload
from model import HFDinoV3Backbone, KFRelationModel


INPUT_NAMES = ["image", "boxes", "box_counts"]
OUTPUT_NAMES = [
    "pred_logits",
    "pair_logits",
    "sub_idx",
    "obj_idx",
    "valid_mask",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dummy_boxes(max_boxes: int) -> torch.Tensor:
    boxes = torch.zeros((1, max_boxes, 4), dtype=torch.float32)
    columns = max(1, int(max_boxes**0.5))
    rows = max(1, (max_boxes + columns - 1) // columns)
    for index in range(max_boxes):
        x = (index % columns + 0.5) / columns
        y = (index // columns + 0.5) / rows
        boxes[0, index] = torch.tensor((x, y, 0.2, 0.2))
    return boxes.clamp(0.01, 0.99)


def export_graph(
    model: nn.Module,
    output_path: str | Path,
    image: Tensor,
    boxes: Tensor,
    box_counts: Tensor,
    *,
    opset: int = 18,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Export the exact native batch-1 relation contract.

    Native kfcore::relation::RelateAnything executes one frame per call.
    Keeping batch=1 static makes the ONNX/TensorRT contract smaller and avoids
    advertising a dynamic-batch capability the C++ API does not consume.
    """
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("ONNX export image must be [1,3,H,W]")
    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[2] != 4:
        raise ValueError("ONNX export boxes must be [1,N,4]")
    if box_counts.shape != (1,):
        raise ValueError("ONNX export box_counts must be [1]")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        reference = model(image, boxes, box_counts)

    torch.onnx.export(
        model,
        (image, boxes, box_counts),
        str(path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return reference


def check_onnx_parity(
    output_path: str | Path,
    image: Tensor,
    boxes: Tensor,
    box_counts: Tensor,
    reference: Sequence[Tensor],
    *,
    tolerance: float = 1.0e-3,
) -> float:
    import onnx
    import onnxruntime as ort

    path = Path(output_path)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)

    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if [item.name for item in inputs] != INPUT_NAMES:
        raise RuntimeError("ONNX input names do not match native runtime contract")
    if [item.name for item in outputs] != OUTPUT_NAMES:
        raise RuntimeError("ONNX output names do not match native runtime contract")

    expected_input_shapes = [
        list(image.shape),
        list(boxes.shape),
        list(box_counts.shape),
    ]
    expected_input_types = [
        "tensor(float)",
        "tensor(float)",
        "tensor(int64)",
    ]
    for item, shape, dtype in zip(
        inputs, expected_input_shapes, expected_input_types
    ):
        if list(item.shape) != shape:
            raise RuntimeError(
                f"ONNX input {item.name} shape {item.shape} != {shape}"
            )
        if item.type != dtype:
            raise RuntimeError(
                f"ONNX input {item.name} type {item.type} != {dtype}"
            )

    expected_output_shapes = [
        list(value.shape) for value in reference
    ]
    expected_output_types = [
        "tensor(float)",
        "tensor(float)",
        "tensor(int64)",
        "tensor(int64)",
        "tensor(bool)",
    ]
    for item, shape, dtype in zip(
        outputs, expected_output_shapes, expected_output_types
    ):
        if list(item.shape) != shape:
            raise RuntimeError(
                f"ONNX output {item.name} shape {item.shape} != {shape}"
            )
        if item.type != dtype:
            raise RuntimeError(
                f"ONNX output {item.name} type {item.type} != {dtype}"
            )

    feed = {
        "image": image.detach().cpu().numpy(),
        "boxes": boxes.detach().cpu().numpy(),
        "box_counts": box_counts.detach().cpu().numpy(),
    }
    actual = session.run(OUTPUT_NAMES, feed)
    worst = 0.0
    for name, expected, got in zip(OUTPUT_NAMES, reference, actual):
        expected_np = expected.detach().cpu().numpy()
        if expected_np.dtype == np.bool_ or np.issubdtype(
            expected_np.dtype, np.integer
        ):
            if not np.array_equal(expected_np, got):
                raise RuntimeError(f"ONNX parity failed for {name}")
        else:
            delta = float(np.max(np.abs(expected_np - got)))
            worst = max(worst, delta)
            if delta > tolerance:
                raise RuntimeError(
                    f"ONNX parity failed for {name}: max abs delta {delta}"
                )
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a KFCore-owned DINOv3 relation checkpoint to ONNX."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--backbone",
        default="",
        help="Hugging Face model ID or local model directory. Defaults to checkpoint provenance.",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.out)
    payload = load_payload(checkpoint_path)
    config = config_from_payload(payload)
    predicates = list(payload["predicates"])
    backbone_name = args.backbone or str(payload["backbone_model"])

    backbone = HFDinoV3Backbone.from_pretrained(
        backbone_name, train_backbone=False
    )
    model = KFRelationModel(
        backbone,
        payload["predicate_embeddings"].float(),
        config,
    )
    model.load_state_dict(payload["state_dict"], strict=True)

    image = torch.rand(
        1, 3, config.image_size, config.image_size, dtype=torch.float32
    )
    boxes = dummy_boxes(config.max_boxes)
    box_counts = torch.tensor([config.max_boxes], dtype=torch.int64)

    reference = export_graph(
        model,
        output_path,
        image,
        boxes,
        box_counts,
        opset=args.opset,
    )

    metadata = {
        "schema": "kfcore.relation-onnx/1",
        "model_type": "relation.relate-anything",
        "implementation": "kfcore-relation-v1",
        "output_kind": "logits",
        "batch_size": 1,
        "score_contract": "sigmoid(a * (pred + w * pair) + b)",
        "image_size": config.image_size,
        "max_boxes": config.max_boxes,
        "final_budget": config.pair_budget,
        "predicates": predicates,
        "predicate_count": len(predicates),
        "backbone_model": backbone_name,
        "backbone_patch_size": backbone.patch_size,
        "backbone_hidden_size": backbone.hidden_size,
        "tap_indices": list(config.tap_indices),
        "checkpoint_sha256": sha256(checkpoint_path),
        "onnx_sha256": sha256(output_path),
        "opset": args.opset,
    }

    if args.check:
        metadata["check_max_abs_delta"] = check_onnx_parity(
            output_path,
            image,
            boxes,
            box_counts,
            reference,
        )

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
