from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from export_onnx import check_onnx_parity, export_graph
from model import KFRelationModel, MetaDinoV3Backbone, RelationModelConfig


DEFAULT_MODEL = "dinov3_vits16"
DEFAULT_REPO_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
DEFAULT_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16/"
    "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo_dir: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run official Meta DINOv3 ViT-S/16 -> KFRelationModel -> ONNX smoke."
        )
    )
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-repo-commit", default=DEFAULT_REPO_COMMIT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    if args.image_size <= 0:
        raise ValueError("image-size must be positive")

    repo_dir = Path(args.repo_dir).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    if not repo_dir.is_dir():
        raise ValueError("repo-dir does not exist")
    if not weights_path.is_file():
        raise ValueError("weights file does not exist")

    actual_repo_commit = git_head(repo_dir)
    if actual_repo_commit != args.expected_repo_commit:
        raise RuntimeError(
            f"DINOv3 source commit {actual_repo_commit} != "
            f"{args.expected_repo_commit}"
        )

    # The official ViT-S/16 filename contains the first 8 chars of its SHA-256,
    # and the official loader is called with check_hash=True below.
    weights_sha256 = file_sha256(weights_path)
    if not weights_sha256.startswith("08c60483"):
        raise RuntimeError(
            "downloaded DINOv3 ViT-S/16 weights do not match official hash prefix"
        )

    torch.manual_seed(20260929)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "real-dinov3-relation-smoke.onnx"
    report_path = output_dir / "real-dinov3-smoke.json"

    backbone = MetaDinoV3Backbone.from_official_repo(
        str(repo_dir),
        model_name=args.model,
        weights=str(weights_path),
        train_backbone=False,
        check_hash=True,
    )
    if args.image_size % backbone.patch_size != 0:
        raise ValueError(
            "smoke image size must be divisible by DINOv3 patch size"
        )
    if backbone.patch_size != 16 or backbone.hidden_size != 384:
        raise RuntimeError(
            "official DINOv3 ViT-S/16 architecture contract changed"
        )

    # Keep the relation head intentionally small: this qualification targets
    # the real DINOv3 backbone/adapter/export path, not relation accuracy.
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
    if not torch.isfinite(pred_logits).all() or not torch.isfinite(
        pair_logits
    ).all():
        raise RuntimeError("real DINO smoke produced non-finite logits")
    for slot in range(config.pair_budget):
        if bool(valid_mask[0, slot]):
            subject = int(sub_idx[0, slot])
            object_ = int(obj_idx[0, slot])
            if subject >= 4 or object_ >= 4 or subject == object_:
                raise RuntimeError(
                    "real DINO smoke emitted an invalid pair index"
                )

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
        "schema": "kfcore.real-dinov3-relation-smoke/2",
        "source_repository": "facebookresearch/dinov3",
        "source_commit": actual_repo_commit,
        "model": args.model,
        "weights_filename": weights_path.name,
        "weights_sha256": weights_sha256,
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
