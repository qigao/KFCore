from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from checkpoint import config_from_payload, load_payload
from model import HFDinoV3Backbone, KFRelationModel


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
    for index in range(max_boxes):
        x = (index % columns + 0.5) / columns
        y = (index // columns + 0.5) / max(1, (max_boxes + columns - 1) // columns)
        boxes[0, index] = torch.tensor((x, y, 0.2, 0.2))
    return boxes.clamp(0.01, 0.99)


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
    model.eval()

    image = torch.rand(
        1, 3, config.image_size, config.image_size, dtype=torch.float32
    )
    boxes = dummy_boxes(config.max_boxes)
    box_counts = torch.tensor([config.max_boxes], dtype=torch.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference = model(image, boxes, box_counts)

    torch.onnx.export(
        model,
        (image, boxes, box_counts),
        str(output_path),
        input_names=["image", "boxes", "box_counts"],
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "image": {0: "batch"},
            "boxes": {0: "batch"},
            "box_counts": {0: "batch"},
            "pred_logits": {0: "batch"},
            "pair_logits": {0: "batch"},
            "sub_idx": {0: "batch"},
            "obj_idx": {0: "batch"},
            "valid_mask": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    metadata = {
        "schema": "kfcore.relation-onnx/1",
        "model_type": "relation.relate-anything",
        "implementation": "kfcore-relation-v1",
        "output_kind": "logits",
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
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(output_path), providers=["CPUExecutionProvider"]
        )
        feed = {
            "image": image.numpy(),
            "boxes": boxes.numpy(),
            "box_counts": box_counts.numpy(),
        }
        actual = session.run(OUTPUT_NAMES, feed)
        worst = 0.0
        for name, expected, got in zip(OUTPUT_NAMES, reference, actual):
            expected_np = expected.detach().cpu().numpy()
            if expected_np.dtype == np.bool_ or np.issubdtype(expected_np.dtype, np.integer):
                if not np.array_equal(expected_np, got):
                    raise RuntimeError(f"ONNX parity failed for {name}")
            else:
                delta = float(np.max(np.abs(expected_np - got)))
                worst = max(worst, delta)
                if delta > 1.0e-3:
                    raise RuntimeError(
                        f"ONNX parity failed for {name}: max abs delta {delta}"
                    )
        metadata["check_max_abs_delta"] = worst

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
