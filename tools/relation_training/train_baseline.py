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
from make_predicate_embeddings import gram_diagnostics, tensor_sha256
from model import KFRelationModel, RelationModelConfig, TimmDinoV3Backbone
from training import (
    FrozenBaselineConfig,
    RelationTrainingDataset,
    build_predicate_weighting,
    build_zero_support_negative_weights,
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
    parser.add_argument(
        "--predicate-adapter-rank",
        type=int,
        default=0,
        help="Shared low-rank residual adapter rank; 0 disables it.",
    )
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
    parser.add_argument(
        "--mask-zero-support-predicates",
        action="store_true",
        help=(
            "Exclude train-zero-support predicate dimensions from predicate BCE."
        ),
    )
    parser.add_argument(
        "--holdout-predicate",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Repeat to explicitly exclude a predicate from predicate BCE "
            "while retaining pair-existence supervision."
        ),
    )
    parser.add_argument(
        "--zero-support-negative-weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier for negative BCE terms on train-zero-support "
            "predicate dimensions; 1 is legacy exhaustive BCE, 0 is "
            "equivalent to masking those negative-only dimensions."
        ),
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
        predicate_adapter_rank=args.predicate_adapter_rank,
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
    validation_weighting = build_predicate_weighting(
        validation_manifest,
        predicate_count=len(vocabulary.predicates),
        mode="none",
        cap=args.predicate_positive_weight_cap,
    )
    predicate_positive_weights = predicate_weighting.tensor()

    holdout_names = tuple(args.holdout_predicate)
    if len(set(holdout_names)) != len(holdout_names):
        raise ValueError("--holdout-predicate values must be unique")
    if holdout_names and args.mask_zero_support_predicates:
        raise ValueError(
            "--holdout-predicate cannot be combined with "
            "--mask-zero-support-predicates"
        )

    predicate_index = {
        name: index
        for index, name in enumerate(vocabulary.predicates)
    }
    unknown_holdouts = [
        name for name in holdout_names
        if name not in predicate_index
    ]
    if unknown_holdouts:
        raise ValueError(
            f"unknown holdout predicate: {unknown_holdouts[0]}"
        )
    holdout_indices = tuple(
        sorted(predicate_index[name] for name in holdout_names)
    )
    for index in holdout_indices:
        if predicate_weighting.predicate_positive_counts[index] <= 0:
            raise ValueError(
                "explicit holdout predicate must have positive train support"
            )
        if validation_weighting.predicate_positive_counts[index] <= 0:
            raise ValueError(
                "explicit holdout predicate must have positive validation support"
            )

    effective_train_predicate_support = list(
        predicate_weighting.predicate_positive_counts
    )
    for index in holdout_indices:
        effective_train_predicate_support[index] = 0
    if (
        args.mask_zero_support_predicates
        and args.zero_support_negative_weight != 1.0
    ):
        raise ValueError(
            "--mask-zero-support-predicates cannot be combined with "
            "--zero-support-negative-weight != 1"
        )
    predicate_negative_weights = build_zero_support_negative_weights(
        predicate_count=len(vocabulary.predicates),
        zero_support_predicate_indices=(
            predicate_weighting.zero_support_predicate_indices
        ),
        zero_support_negative_weight=args.zero_support_negative_weight,
    )
    allow_masked_positive_targets = False
    if holdout_indices:
        predicate_supervision_mask = torch.ones(
            len(vocabulary.predicates),
            dtype=torch.bool,
        )
        predicate_supervision_mask[
            list(holdout_indices)
        ] = False
        predicate_supervision_mode = "explicit-holdout"
        allow_masked_positive_targets = True
    elif args.mask_zero_support_predicates:
        predicate_supervision_mask = torch.tensor(
            [
                count > 0
                for count in predicate_weighting.predicate_positive_counts
            ],
            dtype=torch.bool,
        )
        predicate_supervision_mode = "train-supported-only"
    else:
        predicate_supervision_mask = None
        predicate_supervision_mode = "all"

    supervised_predicate_indices = [
        index
        for index in range(len(vocabulary.predicates))
        if (
            predicate_supervision_mask is None
            or bool(predicate_supervision_mask[index])
        )
    ]
    masked_predicate_indices = [
        index
        for index in range(len(vocabulary.predicates))
        if index not in supervised_predicate_indices
    ]

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
            predicate_supervision_mask=predicate_supervision_mask,
            predicate_negative_weights=predicate_negative_weights,
            allow_masked_positive_targets=allow_masked_positive_targets,
        )
        history.append({"epoch": epoch, **losses})
        print(json.dumps(history[-1], sort_keys=True))

    benchmark_report = evaluate_gt_boxes(
        model,
        validation_manifest,
        image_root=args.image_root,
        device=device,
        benchmark_config=benchmark_config,
        train_predicate_support=(
            predicate_weighting.predicate_positive_counts
        ),
        explicit_holdout_predicate_indices=(
            holdout_indices if holdout_indices else None
        ),
    )

    source_bank = model.predicate_bank.detach().cpu()
    effective_bank = model.effective_predicate_bank().detach().cpu()
    source_effective_cosine = torch.nn.functional.cosine_similarity(
        source_bank,
        effective_bank,
        dim=-1,
    )
    predicate_adapter_report = {
        "rank": model.config.predicate_adapter_rank,
        "parameter_count": model.predicate_adapter_parameter_count(),
        "source_tensor_sha256": tensor_sha256(source_bank),
        "effective_tensor_sha256": tensor_sha256(effective_bank),
        "source_gram": gram_diagnostics(source_bank),
        "effective_gram": gram_diagnostics(effective_bank),
        "mean_row_cosine_to_source": float(
            source_effective_cosine.mean().item()
        ),
        "min_row_cosine_to_source": float(
            source_effective_cosine.min().item()
        ),
    }

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
            "predicate_supervision": {
                "mode": predicate_supervision_mode,
                "supervised_predicate_indices": supervised_predicate_indices,
                "masked_predicate_indices": masked_predicate_indices,
                "held_out_predicate_indices": list(holdout_indices),
                "held_out_predicate_names": [
                    vocabulary.predicates[index]
                    for index in holdout_indices
                ],
                "original_train_predicate_support": list(
                    predicate_weighting.predicate_positive_counts
                ),
                "effective_train_predicate_support": list(
                    effective_train_predicate_support
                ),
            },
            "predicate_negative_weighting": {
                "zero_support_negative_weight": (
                    args.zero_support_negative_weight
                ),
                "negative_weights": (
                    predicate_negative_weights.tolist()
                ),
            },
            "predicate_adapter": predicate_adapter_report,
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
            "predicate_adapter_rank": model.config.predicate_adapter_rank,
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
        "predicate_supervision": {
            "mode": predicate_supervision_mode,
            "supervised_predicate_indices": supervised_predicate_indices,
            "masked_predicate_indices": masked_predicate_indices,
            "held_out_predicate_indices": list(holdout_indices),
            "held_out_predicate_names": [
                vocabulary.predicates[index]
                for index in holdout_indices
            ],
            "original_train_predicate_support": list(
                predicate_weighting.predicate_positive_counts
            ),
            "effective_train_predicate_support": list(
                effective_train_predicate_support
            ),
        },
        "predicate_negative_weighting": {
            "zero_support_negative_weight": (
                args.zero_support_negative_weight
            ),
            "negative_weights": (
                predicate_negative_weights.tolist()
            ),
        },
        "predicate_adapter": predicate_adapter_report,
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
