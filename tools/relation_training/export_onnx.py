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
DYNAMIC_VOCAB_INPUT_NAMES = [
    "image",
    "boxes",
    "box_counts",
    "W",
    "alpha",
]
OUTPUT_NAMES = [
    "pred_logits",
    "pair_logits",
    "sub_idx",
    "obj_idx",
    "valid_mask",
]
ENCODER_OUTPUT_NAMES = [
    "semantic_query",
    "spatial_query",
    "pair_logits",
    "sub_idx",
    "obj_idx",
    "valid_mask",
]

_OUTPUT_TYPES = {
    "pred_logits": "tensor(float)",
    "semantic_query": "tensor(float)",
    "spatial_query": "tensor(float)",
    "pair_logits": "tensor(float)",
    "sub_idx": "tensor(int64)",
    "obj_idx": "tensor(int64)",
    "valid_mask": "tensor(bool)",
}


class RelationDynamicVocabularyExport(nn.Module):
    def __init__(self, model: KFRelationModel) -> None:
        super().__init__()
        if model.config.predicate_head_contract != "apache":
            raise ValueError(
                "dynamic-vocabulary graph requires the Apache predicate head"
            )
        if model.apache_vocab_head is None:
            raise ValueError(
                "dynamic-vocabulary graph requires ApacheVocabHead"
            )
        self.model = model

    def forward(
        self,
        image: Tensor,
        boxes: Tensor,
        box_counts: Tensor,
        W: Tensor,
        alpha: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        (
            semantic_query,
            spatial_query,
            pair_logits,
            sub_idx,
            obj_idx,
            valid_mask,
        ) = self.model.forward_encoder(
            image,
            boxes,
            box_counts,
        )
        assert self.model.apache_vocab_head is not None
        pred_logits = self.model.apache_vocab_head.score_query_dual(
            semantic_query,
            spatial_query,
            W,
            alpha=alpha,
        )
        return (
            pred_logits,
            pair_logits,
            sub_idx,
            obj_idx,
            valid_mask,
        )


class RelationEncoderExport(nn.Module):
    def __init__(self, model: KFRelationModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, image: Tensor, boxes: Tensor, box_counts: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.model.forward_encoder(image, boxes, box_counts)


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
    output_names: Sequence[str] = OUTPUT_NAMES,
) -> tuple[Tensor, ...]:
    """Export an exact batch-1 KFCore relation graph contract."""
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("ONNX export image must be [1,3,H,W]")
    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[2] != 4:
        raise ValueError("ONNX export boxes must be [1,N,4]")
    if box_counts.shape != (1,):
        raise ValueError("ONNX export box_counts must be [1]")

    names = list(output_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("ONNX output names must be non-empty and unique")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        reference = tuple(model(image, boxes, box_counts))
    if len(reference) != len(names):
        raise RuntimeError(
            "model output count does not match requested ONNX names"
        )

    torch.onnx.export(
        model,
        (image, boxes, box_counts),
        str(path),
        input_names=INPUT_NAMES,
        output_names=names,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return reference


def export_dynamic_vocabulary_graph(
    model: KFRelationModel,
    output_path: str | Path,
    image: Tensor,
    boxes: Tensor,
    box_counts: Tensor,
    W: Tensor,
    alpha: Tensor,
    *,
    opset: int = 18,
) -> tuple[Tensor, ...]:
    if W.ndim != 2 or W.shape[1] != model.predicate_dim:
        raise ValueError(
            "dynamic predicate bank must be [V,predicate_dim]"
        )
    if alpha.shape != (W.shape[0],):
        raise ValueError("dynamic predicate alpha must be [V]")
    wrapper = RelationDynamicVocabularyExport(model).eval()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference = tuple(
            wrapper(
                image,
                boxes,
                box_counts,
                W,
                alpha,
            )
        )
    torch.onnx.export(
        wrapper,
        (image, boxes, box_counts, W, alpha),
        str(path),
        input_names=DYNAMIC_VOCAB_INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "W": {0: "num_predicates"},
            "alpha": {0: "num_predicates"},
            "pred_logits": {2: "num_predicates"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return reference


def check_dynamic_vocabulary_parity(
    output_path: str | Path,
    model: KFRelationModel,
    image: Tensor,
    boxes: Tensor,
    box_counts: Tensor,
    vocabularies: Sequence[tuple[Tensor, Tensor]],
    *,
    tolerance: float = 1.0e-3,
) -> float:
    import onnx
    import onnxruntime as ort

    if not vocabularies:
        raise ValueError(
            "dynamic-vocabulary parity requires at least one vocabulary"
        )
    path = Path(output_path)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    if [item.name for item in session.get_inputs()] != DYNAMIC_VOCAB_INPUT_NAMES:
        raise RuntimeError(
            "dynamic-vocabulary ONNX input names drifted"
        )
    if [item.name for item in session.get_outputs()] != OUTPUT_NAMES:
        raise RuntimeError(
            "dynamic-vocabulary ONNX output names drifted"
        )

    wrapper = RelationDynamicVocabularyExport(model).eval()
    worst = 0.0
    for W, alpha in vocabularies:
        if W.ndim != 2 or W.shape[1] != model.predicate_dim:
            raise ValueError(
                "dynamic parity W must be [V,predicate_dim]"
            )
        if alpha.shape != (W.shape[0],):
            raise ValueError("dynamic parity alpha must be [V]")
        with torch.inference_mode():
            expected = tuple(
                wrapper(
                    image,
                    boxes,
                    box_counts,
                    W,
                    alpha,
                )
            )
        actual = session.run(
            OUTPUT_NAMES,
            {
                "image": image.detach().cpu().numpy(),
                "boxes": boxes.detach().cpu().numpy(),
                "box_counts": box_counts.detach().cpu().numpy(),
                "W": W.detach().cpu().numpy(),
                "alpha": alpha.detach().cpu().numpy(),
            },
        )
        if actual[0].shape != (
            1,
            model.config.pair_budget,
            int(W.shape[0]),
        ):
            raise RuntimeError(
                "dynamic predicate-logit shape does not follow runtime V"
            )
        for name, expected_value, got in zip(
            OUTPUT_NAMES,
            expected,
            actual,
        ):
            expected_np = expected_value.detach().cpu().numpy()
            if expected_np.dtype == np.bool_ or np.issubdtype(
                expected_np.dtype,
                np.integer,
            ):
                if not np.array_equal(expected_np, got):
                    raise RuntimeError(
                        f"dynamic ONNX parity failed for {name}"
                    )
            else:
                delta = float(
                    np.max(np.abs(expected_np - got))
                )
                worst = max(worst, delta)
                if delta > tolerance:
                    raise RuntimeError(
                        f"dynamic ONNX parity failed for {name}: "
                        f"max abs delta {delta}"
                    )
    return worst


def export_encoder_graph(
    model: KFRelationModel,
    output_path: str | Path,
    image: Tensor,
    boxes: Tensor,
    box_counts: Tensor,
    *,
    opset: int = 18,
) -> tuple[Tensor, ...]:
    return export_graph(
        RelationEncoderExport(model),
        output_path,
        image,
        boxes,
        box_counts,
        opset=opset,
        output_names=ENCODER_OUTPUT_NAMES,
    )


def check_onnx_parity(
    output_path: str | Path,
    image: Tensor,
    boxes: Tensor,
    box_counts: Tensor,
    reference: Sequence[Tensor],
    *,
    tolerance: float = 1.0e-3,
    output_names: Sequence[str] = OUTPUT_NAMES,
) -> float:
    import onnx
    import onnxruntime as ort

    names = list(output_names)
    path = Path(output_path)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)

    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if [item.name for item in inputs] != INPUT_NAMES:
        raise RuntimeError(
            "ONNX input names do not match native runtime contract"
        )
    if [item.name for item in outputs] != names:
        raise RuntimeError(
            "ONNX output names do not match requested runtime contract"
        )

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

    if len(reference) != len(names):
        raise RuntimeError(
            "reference output count does not match requested contract"
        )
    expected_output_shapes = [list(value.shape) for value in reference]
    expected_output_types = [_OUTPUT_TYPES[name] for name in names]
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
    actual = session.run(names, feed)
    worst = 0.0
    for name, expected, got in zip(names, reference, actual):
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
                    f"ONNX parity failed for {name}: "
                    f"max abs delta {delta}"
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
        help=(
            "Hugging Face model ID or local model directory. "
            "Defaults to checkpoint provenance."
        ),
    )
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help=(
            "Export relation.open-vocabulary-encoder with fixed [K,D] "
            "queries instead of baked [K,V] predicate logits."
        ),
    )
    parser.add_argument(
        "--dynamic-vocabulary",
        action="store_true",
        help=(
            "Export relation.open-vocabulary with runtime W[V,D] and "
            "alpha[V] inputs and dynamic pred_logits vocabulary axis."
        ),
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.encoder_only and args.dynamic_vocabulary:
        raise ValueError(
            "--encoder-only and --dynamic-vocabulary are mutually exclusive"
        )

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
    box_counts = torch.tensor(
        [config.max_boxes], dtype=torch.int64
    )

    dynamic_W = None
    dynamic_alpha = None
    if args.dynamic_vocabulary:
        if model.config.predicate_head_contract != "apache":
            raise ValueError(
                "--dynamic-vocabulary requires predicate_head_contract=apache"
            )
        dynamic_W = torch.nn.functional.normalize(
            payload["predicate_embeddings"].float(),
            dim=-1,
        )
        assert model.apache_vocab_head is not None
        with torch.inference_mode():
            dynamic_alpha = model.apache_vocab_head.routing_alpha(
                dynamic_W
            )
        reference = export_dynamic_vocabulary_graph(
            model,
            output_path,
            image,
            boxes,
            box_counts,
            dynamic_W,
            dynamic_alpha,
            opset=args.opset,
        )
        output_names = OUTPUT_NAMES
        model_type = "relation.open-vocabulary"
        output_kind = "dynamic-vocabulary-logits"
    elif args.encoder_only:
        reference = export_encoder_graph(
            model,
            output_path,
            image,
            boxes,
            box_counts,
            opset=args.opset,
        )
        output_names = ENCODER_OUTPUT_NAMES
        model_type = "relation.open-vocabulary-encoder"
        output_kind = "relation-queries"
    else:
        reference = export_graph(
            model,
            output_path,
            image,
            boxes,
            box_counts,
            opset=args.opset,
        )
        output_names = OUTPUT_NAMES
        model_type = "relation.relate-anything"
        output_kind = "logits"

    metadata = {
        "schema": "kfcore.relation-onnx/2",
        "model_type": model_type,
        "implementation": "kfcore-relation-v1",
        "output_kind": output_kind,
        "batch_size": 1,
        "image_size": config.image_size,
        "max_boxes": config.max_boxes,
        "final_budget": config.pair_budget,
        "backbone_model": backbone_name,
        "backbone_patch_size": backbone.patch_size,
        "backbone_hidden_size": backbone.hidden_size,
        "tap_indices": list(config.tap_indices),
        "checkpoint_sha256": sha256(checkpoint_path),
        "onnx_sha256": sha256(output_path),
        "opset": args.opset,
    }
    if args.dynamic_vocabulary:
        assert model.apache_vocab_head is not None
        metadata.update({
            "vocabulary_dynamic": True,
            "vocabulary_graph_input": True,
            "query_dim": model.predicate_dim,
            "predicate_axis": "dynamic",
            "score_logit_scale": float(
                model.apache_vocab_head.logit_scale.exp()
                .clamp(max=100.0)
                .detach()
                .cpu()
            ),
            "score_logit_bias": float(
                model.apache_vocab_head.logit_bias.detach().cpu()
            ),
            "score_contract": (
                "scale*((1-alpha)*cos(q_sem,W)+alpha*cos(q_spa,W))+bias"
            ),
        })
    elif args.encoder_only:
        metadata.update({
            "vocabulary_dynamic": True,
            "query_dim": model.predicate_dim,
            "semantic_query": True,
            "spatial_query": True,
            "score_logit_scale": float(
                (
                    model.apache_vocab_head.logit_scale.exp()
                    if model.apache_vocab_head is not None
                    else model.logit_scale.exp()
                )
                .clamp(max=100.0)
                .detach()
                .cpu()
            ),
            "score_logit_bias": float(
                (
                    model.apache_vocab_head.logit_bias
                    if model.apache_vocab_head is not None
                    else torch.zeros(())
                )
                .detach()
                .cpu()
            ),
            "score_contract": (
                "scale*((1-alpha)*dot(q_sem,w)+alpha*dot(q_spa,w))+bias"
            ),
        })
    else:
        metadata.update({
            "vocabulary_dynamic": False,
            "score_contract": "sigmoid(a * (pred + w * pair) + b)",
            "predicates": predicates,
            "predicate_count": len(predicates),
        })

    if args.check:
        if args.dynamic_vocabulary:
            assert dynamic_W is not None
            assert dynamic_alpha is not None
            one_W = dynamic_W[:1]
            one_alpha = dynamic_alpha[:1]
            random_W = torch.nn.functional.normalize(
                torch.randn(3, model.predicate_dim),
                dim=-1,
            )
            random_alpha = torch.tensor(
                [0.0, 0.5, 1.0],
                dtype=torch.float32,
            )
            metadata["check_max_abs_delta"] = (
                check_dynamic_vocabulary_parity(
                    output_path,
                    model,
                    image,
                    boxes,
                    box_counts,
                    [
                        (one_W, one_alpha),
                        (random_W, random_alpha),
                        (dynamic_W, dynamic_alpha),
                    ],
                )
            )
        else:
            metadata["check_max_abs_delta"] = check_onnx_parity(
                output_path,
                image,
                boxes,
                box_counts,
                reference,
                output_names=output_names,
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
