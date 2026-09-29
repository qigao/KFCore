from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmark import (
    BenchmarkConfig,
    DatasetManifest,
    RelationVocabulary,
    stable_report_json,
)
from checkpoint import save_checkpoint
from losses import RelationLossConfig
from model import KFRelationModel, RelationModelConfig, TimmDinoV3Backbone
from training import (
    FrozenBaselineConfig,
    RelationTrainingDataset,
    build_predicate_weighting,
    config_payload,
    evaluate_gt_boxes,
    freeze_backbone,
    make_training_loader,
    seed_everything,
    train_epoch,
    trainable_parameters,
    validate_disjoint_splits,
)


DEFAULT_BACKBONE = "hf_hub:timm/vit_small_patch16_dinov3.lvd1689m"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predicate_embeddings(
    path: str | Path,
    predicate_count: int,
) -> torch.Tensor:
    value = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(value, torch.Tensor):
        raise ValueError("predicate embedding file must contain one tensor")
    if value.ndim != 2 or value.shape[0] != predicate_count:
        raise ValueError(
            "predicate embeddings must be [predicate_count,embedding_dim]"
        )
    if value.shape[1] <= 0 or not torch.isfinite(value).all():
        raise ValueError("predicate embeddings must be finite and non-empty")
    return value.float()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the first frozen-DINOv3 KFCore relation baseline "
            "on canonical GT-box annotations."
        )
    )
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--validation-annotations", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--predicate-embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-boxes", type=int, default=32)
    parser.add_argument("--pair-budget", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--geometry-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--sampler-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-loss-weight", type=float, default=1.0)
    parser.add_argument("--predicate-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--predicate-positive-weight-mode",
        choices=("none", "sqrt-balanced", "balanced"),
        default="none",
    )
    parser.add_argument(
        "--predicate-positive-weight-cap",
        type=float,
        default=20.0,
    )
    parser.add_argument("--negative-pair-weight", type=float, default=0.25)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}"
        )

    vocabulary = RelationVocabulary.load(args.vocabulary)
    train_manifest = DatasetManifest.load(
        args.train_annotations, vocabulary
    )
    validation_manifest = DatasetManifest.load(
        args.validation_annotations, vocabulary
    )
    validate_disjoint_splits(
        train_manifest, validation_manifest
    )

    baseline_config = FrozenBaselineConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    model_config = RelationModelConfig(
        image_size=args.image_size,
        max_boxes=args.max_boxes,
        pair_budget=args.pair_budget,
        hidden_dim=args.hidden_dim,
        geometry_dim=args.geometry_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.0,
        tap_indices=(-6, -3, -1),
    )
    loss_config = RelationLossConfig(
        sampler_loss_weight=args.sampler_loss_weight,
        pair_loss_weight=args.pair_loss_weight,
        predicate_loss_weight=args.predicate_loss_weight,
        negative_pair_weight=args.negative_pair_weight,
    )
    benchmark_config = BenchmarkConfig(
        top_ks=(20, 50, 100),
        pair_weight=args.pair_weight,
    )
    predicate_weighting = build_predicate_weighting(
        train_manifest,
        predicate_count=len(vocabulary.predicates),
        mode=args.predicate_positive_weight_mode,
        cap=args.predicate_positive_weight_cap,
    )
    predicate_positive_weights = predicate_weighting.tensor()

    seed_everything(baseline_config.seed)
    predicate_embeddings = load_predicate_embeddings(
        args.predicate_embeddings,
        len(vocabulary.predicates),
    )
    backbone = TimmDinoV3Backbone.from_pretrained(
        args.backbone,
        train_backbone=False,
    )
    model = KFRelationModel(
        backbone,
        predicate_embeddings,
        model_config,
    )
    freeze_backbone(model)

    device = resolve_device(args.device)
    model.to(device)

    train_dataset = RelationTrainingDataset(
        train_manifest,
        image_root=args.image_root,
        image_size=model.config.image_size,
        max_boxes=model.config.max_boxes,
        predicate_count=len(vocabulary.predicates),
    )
    loader = make_training_loader(
        train_dataset, baseline_config
    )
    parameters = trainable_parameters(model)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in parameters
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=baseline_config.learning_rate,
        weight_decay=baseline_config.weight_decay,
    )

    history: list[dict[str, object]] = []
    for epoch in range(1, baseline_config.epochs + 1):
        losses = train_epoch(
            model,
            loader,
            optimizer,
            device=device,
            loss_config=loss_config,
            predicate_positive_weights=predicate_positive_weights,
        )
        history.append({"epoch": epoch, **losses})
        print(json.dumps(history[-1], sort_keys=True))

    benchmark_report = evaluate_gt_boxes(
        model,
        validation_manifest,
        image_root=args.image_root,
        device=device,
        benchmark_config=benchmark_config,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "relation-v1.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        backbone_model=args.backbone,
        predicates=list(vocabulary.predicates),
        extra={
            "training_schema": "kfcore.relation-frozen-baseline/1",
            "frozen_backbone": True,
            "train_annotations_sha256": train_manifest.annotations_sha256,
            "validation_annotations_sha256": (
                validation_manifest.annotations_sha256
            ),
            "vocabulary_sha256": vocabulary.sha256(),
            "baseline_config": config_payload(baseline_config),
            "loss_config": {
                "sampler_loss_weight": loss_config.sampler_loss_weight,
                "pair_loss_weight": loss_config.pair_loss_weight,
                "predicate_loss_weight": loss_config.predicate_loss_weight,
                "negative_pair_weight": loss_config.negative_pair_weight,
            },
            "predicate_weighting": {
                "mode": predicate_weighting.mode,
                "cap": predicate_weighting.cap,
                "positive_pair_count": predicate_weighting.positive_pair_count,
                "predicate_positive_counts": list(
                    predicate_weighting.predicate_positive_counts
                ),
                "zero_support_predicate_indices": list(
                    predicate_weighting.zero_support_predicate_indices
                ),
                "positive_weights": list(
                    predicate_weighting.positive_weights
                ),
            },
            "history": history,
        },
    )

    training_report = {
        "schema": "kfcore.relation-training-run/1",
        "backbone": args.backbone,
        "frozen_backbone": True,
        "device": str(device),
        "train_examples": len(train_manifest.examples),
        "validation_examples": len(validation_manifest.examples),
        "train_annotations_sha256": train_manifest.annotations_sha256,
        "validation_annotations_sha256": (
            validation_manifest.annotations_sha256
        ),
        "vocabulary_sha256": vocabulary.sha256(),
        "predicate_embeddings_sha256": sha256(
            Path(args.predicate_embeddings)
        ),
        "predicate_embedding_shape": list(
            predicate_embeddings.shape
        ),
        "trainable_parameter_count": trainable_parameter_count,
        "checkpoint_sha256": sha256(checkpoint_path),
        "model_config": {
            "image_size": model.config.image_size,
            "max_boxes": model.config.max_boxes,
            "pair_budget": model.config.pair_budget,
            "hidden_dim": model.config.hidden_dim,
            "geometry_dim": model.config.geometry_dim,
            "num_heads": model.config.num_heads,
            "num_layers": model.config.num_layers,
            "tap_indices": list(model.config.tap_indices),
        },
        "baseline_config": config_payload(baseline_config),
        "loss_config": {
            "sampler_loss_weight": loss_config.sampler_loss_weight,
            "pair_loss_weight": loss_config.pair_loss_weight,
            "predicate_loss_weight": loss_config.predicate_loss_weight,
            "negative_pair_weight": loss_config.negative_pair_weight,
        },
        "benchmark_config": {
            "pair_weight": benchmark_config.pair_weight,
            "top_ks": list(benchmark_config.top_ks),
        },
        "predicate_weighting": {
            "mode": predicate_weighting.mode,
            "cap": predicate_weighting.cap,
            "positive_pair_count": predicate_weighting.positive_pair_count,
            "predicate_positive_counts": list(
                predicate_weighting.predicate_positive_counts
            ),
            "zero_support_predicate_indices": list(
                predicate_weighting.zero_support_predicate_indices
            ),
            "positive_weights": list(
                predicate_weighting.positive_weights
            ),
        },
        "history": history,
    }

    (output_dir / "training.json").write_text(
        json.dumps(
            training_report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "benchmark.json").write_text(
        stable_report_json(benchmark_report),
        encoding="utf-8",
    )

    print(f"wrote {checkpoint_path}")
    print(f"wrote {output_dir / 'training.json'}")
    print(f"wrote {output_dir / 'benchmark.json'}")


if __name__ == "__main__":
    main()
