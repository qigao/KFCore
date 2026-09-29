from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from export_onnx import check_onnx_parity, export_graph
from model import HFDinoV3Backbone, KFRelationModel, RelationModelConfig


DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
# Qualification intentionally uses real pretrained backbone weights but a random
# relation head; accuracy is not part of this smoke.


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real DINOv3 ViT-S/16 -> KFRelationModel -> ONNX smoke."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    if args.image_size <= 0:
        raise ValueError("image-size must be positive")

    torch.manual_seed(20260929)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "real-dinov3-relation-smoke.onnx"
    report_path = output_dir / "real-dinov3-smoke.json"

    backbone = HFDinoV3Backbone.from_pretrained(
        args.model,
        train_backbone=False,
    )
    if args.image_size % backbone.patch_size != 0:
        raise ValueError("smoke image size must be divisible by DINOv3 patch size")

    # Keep the relation head intentionally small: this qualification targets
    # the real DINOv3 adapter/export path, not relation-model accuracy.
    config = RelationModelConfig(
        image_size=args.image_size,
        max_boxes=4,
        pair_budget=6,
        hidden_dim=64,
        geometry_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        tap_indices=(-6, -3, -1),
    )
    predicate_names = ["beside", "holding", "riding"]
    predicate_embeddings = torch.randn(len(predicate_names), 32)

    model = KFRelationModel(
        backbone,
        predicate_embeddings,
        config,
    )
    model.eval()

    image = torch.rand(1, 3, args.image_size, args.image_size)
    boxes = torch.tensor(
        [
            [
                [0.20, 0.20, 0.20, 0.20],
                [0.50, 0.20, 0.20, 0.20],
                [0.25, 0.60, 0.25, 0.25],
                [0.70, 0.65, 0.20, 0.30],
            ]
        ],
        dtype=torch.float32,
    )
    box_counts = torch.tensor([4], dtype=torch.int64)

    with torch.inference_mode():
        outputs = model(image, boxes, box_counts)
    pred_logits, pair_logits, sub_idx, obj_idx, valid_mask = outputs

    expected_shapes = [
        (1, config.pair_budget, len(predicate_names)),
        (1, config.pair_budget),
        (1, config.pair_budget),
        (1, config.pair_budget),
        (1, config.pair_budget),
    ]
    for value, shape in zip(outputs, expected_shapes):
        if tuple(value.shape) != shape:
            raise RuntimeError(
                f"real DINO smoke output shape {tuple(value.shape)} != {shape}"
            )
    if not torch.isfinite(pred_logits).all() or not torch.isfinite(pair_logits).all():
        raise RuntimeError("real DINO smoke produced non-finite logits")
    for slot in range(config.pair_budget):
        if bool(valid_mask[0, slot]):
            subject = int(sub_idx[0, slot])
            object_ = int(obj_idx[0, slot])
            if subject >= 4 or object_ >= 4 or subject == object_:
                raise RuntimeError("real DINO smoke emitted an invalid pair index")

    reference = export_graph(
        model,
        onnx_path,
        image,
        boxes,
        box_counts,
        opset=args.opset,
    )
    max_abs_delta = check_onnx_parity(
        onnx_path,
        image,
        boxes,
        box_counts,
        reference,
    )

    report = {
        "schema": "kfcore.real-dinov3-relation-smoke/1",
        "model": args.model,
        "image_size": args.image_size,
        "patch_size": backbone.patch_size,
        "hidden_size": backbone.hidden_size,
        "tap_indices": list(config.tap_indices),
        "max_boxes": config.max_boxes,
        "pair_budget": config.pair_budget,
        "predicate_names": predicate_names,
        "onnx_opset": args.opset,
        "onnx_sha256": file_sha256(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "ort_max_abs_delta": max_abs_delta,
        "valid_pair_count": int(valid_mask.sum().item()),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
