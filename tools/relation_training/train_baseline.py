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
from apache_mixture import (
    ApacheReleasedBatchSampler,
    RELEASED_SEED,
    RELEASED_WORLD_SIZE,
    DistributedWeightedSampler,
    load_relation_mixture,
    realized_source_draws,
    sample_weights_from_fractions,
    validate_mixture_disjoint_validation,
)
from apache_multiscale import scale_ladder
from apache_pair_sampler import PairOpportunityTable
from apache_training_recipe import (
    ApacheTrainingRecipeConfig,
    ModelEMA,
    RELEASED_EMA_DECAY,
    RELEASED_PHOTOMETRIC_AUGMENT,
    backbone_provenance,
    build_reference_optimizer,
    build_reference_scheduler,
    module_state_sha256,
    resolve_training_augment,
    resolve_training_epochs,
    resolve_training_hidden_dim,
    resolve_training_max_boxes,
    select_artifact_model,
    validate_training_text_dim,
)
from apache_objective import (
    ApacheObjectiveConfig,
    ApacheReferenceObjective,
    PredicateOntology,
    load_source_column_allow,
)
from apache_vocab_head import (
    balanced_spatial_probe_targets,
    load_predicate_spatial_flags,
)
from checkpoint import save_checkpoint
from losses import RelationLossConfig
from make_predicate_embeddings import gram_diagnostics, tensor_sha256
from model import KFRelationModel, RelationModelConfig, TimmDinoV3Backbone
from training import (
    FrozenBaselineConfig,
    RelationMixtureTrainingDataset,
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
    parser.add_argument(
        "--train-annotations",
        default="",
        help=(
            "Single-source training JSONL. Mutually exclusive with "
            "--train-mixture."
        ),
    )
    parser.add_argument(
        "--train-mixture",
        default="",
        help=(
            "JSON mixture contract for multi-source training. Sources carry "
            "name/annotations/image_root/fraction and are assigned source_id "
            "by mixture order."
        ),
    )
    parser.add_argument("--validation-annotations", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--predicate-embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument(
        "--augment",
        type=float,
        default=None,
        help=(
            "Training-only photometric jitter strength. Defaults to 0.3 for "
            "apache-reference and 0 for legacy. apache-reference rejects "
            "any value other than 0.3."
        ),
    )
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=None,
        help=(
            "Maximum relation boxes. Defaults to 40 for apache-reference "
            "and 32 for legacy. apache-reference rejects any value other than 40."
        ),
    )
    parser.add_argument("--pair-budget", type=int, default=128)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help=(
            "Relation head width. Defaults to 512 for apache-reference "
            "and 256 for legacy. apache-reference rejects any value other than 512."
        ),
    )
    parser.add_argument("--geometry-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--predicate-adapter-rank",
        type=int,
        default=0,
        help="Shared low-rank residual adapter rank; 0 disables it.",
    )
    parser.add_argument(
        "--pair-visual-evidence",
        choices=("endpoint", "union", "contact", "union-contact"),
        default="endpoint",
        help=(
            "Visual evidence injected into the relation pair token. "
            "endpoint preserves the historical model; union, contact, "
            "and union-contact add zero-initialized residual projections."
        ),
    )
    parser.add_argument(
        "--pair-geometry-evidence",
        choices=("basic", "rich"),
        default="basic",
        help=(
            "Pair geometry feature contract. rich adds a zero-initialized "
            "residual over normalized offsets, overlap fractions, gaps "
            "and direction without changing the historical basic path."
        ),
    )
    parser.add_argument(
        "--pair-evidence-contract",
        choices=("legacy", "apache"),
        default="legacy",
        help=(
            "Pair evidence implementation. apache enables the Apache-2.0 "
            "RelateAnything reference SoftSpatialPool, BoxPromptEncoder, "
            "19-D geometry and [sub,obj,union,contact,geo] pair fusion."
        ),
    )
    parser.add_argument(
        "--pair-sampler-contract",
        choices=("legacy", "apache"),
        default="legacy",
        help=(
            "Pair sampler implementation. apache enables the two-stage "
            "19-D geometry -> relatedness sampler and requires "
            "--pair-evidence-contract apache."
        ),
    )
    parser.add_argument(
        "--relation-context-contract",
        choices=("legacy", "apache"),
        default="legacy",
        help=(
            "Relation context implementation. apache enables the reference "
            "RelationTransformer -> DeformableRelRead -> "
            "RelationInteractionBlock stack."
        ),
    )
    parser.add_argument(
        "--predicate-head-contract",
        choices=("legacy", "apache"),
        default="legacy",
        help=(
            "Predicate head implementation. apache enables independent "
            "semantic/spatial queries and text-conditioned routing."
        ),
    )
    parser.add_argument(
        "--predicate-spatial-flags",
        default="",
        help=(
            "JSON sidecar for Apache routing warm start. Required when "
            "--predicate-head-contract apache. The predicate list/order must "
            "exactly match --vocabulary."
        ),
    )
    parser.add_argument(
        "--apache-context-dropout",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--apache-box-token-dropout",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--training-recipe",
        choices=("legacy", "apache-reference"),
        default="legacy",
        help=(
            "Optimizer/backbone recipe. apache-reference enables full "
            "DINOv3 fine-tuning with separate head/backbone learning rates, "
            "reference weight-decay grouping, warmup/cosine scheduling and "
            "gradient clipping."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Training epochs. Defaults to 5 for legacy recipe and 12 for "
            "apache-reference."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Micro-batch size. Defaults to 4 for legacy and 32 for the "
            "Apache reference recipe."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--apache-head-lr", type=float, default=4.0e-4)
    parser.add_argument("--apache-backbone-lr", type=float, default=5.0e-5)
    parser.add_argument("--apache-warmup-steps", type=int, default=500)
    parser.add_argument("--apache-min-lr-factor", type=float, default=0.01)
    parser.add_argument("--apache-clip-grad", type=float, default=1.0)
    parser.add_argument(
        "--apache-grad-accum",
        type=int,
        default=4,
        help=(
            "Micro-batches per optimizer step for the single-process Apache "
            "recipe. Default 4 with micro-batch 32 reproduces global batch 128."
        ),
    )
    parser.add_argument(
        "--apache-multi-scale",
        default="0.5,1.5",
        help=(
            "Apache per-batch square scale range relative to --image-size; "
            "empty string disables multi-scale."
        ),
    )
    parser.add_argument(
        "--apache-multi-scale-n",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--apache-cfa-prob",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--apache-cfa-alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument("--sampler-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-loss-weight", type=float, default=1.0)
    parser.add_argument("--predicate-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--predicate-objective",
        choices=("bce", "batch-local-infonce", "apache-reference"),
        default="bce",
        help=(
            "Predicate training objective. bce preserves the historical "
            "exhaustive baseline; batch-local-infonce preserves the earlier "
            "KFCore diagnostic; apache-reference enables the exact Apache "
            "ontology/source-aware objective."
        ),
    )
    parser.add_argument(
        "--predicate-contrastive-temperature",
        type=float,
        default=0.07,
    )
    parser.add_argument(
        "--predicate-contrastive-hard-negative-count",
        type=int,
        default=0,
        help=(
            "For batch-local InfoNCE, add up to this many hardest "
            "train-supported supervised predicate directions to each "
            "batch contrast set. Explicit holdouts and train-zero-support "
            "predicates are excluded from the candidate pool."
        ),
    )
    parser.add_argument(
        "--predicate-calibration-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the source-aware sigmoid calibration auxiliary "
            "on top of batch-local InfoNCE. Only rows with at least one "
            "visible positive predicate are calibrated."
        ),
    )
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
        "--holdout-row-policy",
        choices=("dimension-only", "skip-holdout-only"),
        default="dimension-only",
        help=(
            "Predicate-loss handling for explicit holdouts. "
            "dimension-only masks only held-out dimensions; "
            "skip-holdout-only also skips predicate BCE for positive "
            "pairs whose positive predicate labels are entirely held out."
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
    parser.add_argument(
        "--apache-ontology-meta",
        default="",
        help="Apache soft-supervision metadata JSON; required for apache-reference.",
    )
    parser.add_argument(
        "--apache-ontology-npz",
        default="",
        help="Apache soft_supervision.npz tables; required for apache-reference.",
    )
    parser.add_argument(
        "--apache-source-column-allow",
        default="",
        help="Source-aware predicate column allow JSON; required for apache-reference.",
    )
    parser.add_argument(
        "--apache-object-embeddings",
        default="",
        help="Object-category text embedding tensor [O,D]; required for lambda_obj > 0.",
    )
    parser.add_argument(
        "--apache-neg-rate-table",
        default="",
        help=(
            "Apache pair_opportunity.npz interaction-rate table; required "
            "for apache-reference relatedness/background PU weighting."
        ),
    )
    parser.add_argument("--apache-n-neg", type=int, default=512)
    parser.add_argument("--apache-hard-frac", type=float, default=0.5)
    parser.add_argument("--apache-lambda-obj", type=float, default=0.10)
    parser.add_argument("--apache-lambda-swap", type=float, default=0.50)
    parser.add_argument("--apache-lambda-sigmoid", type=float, default=0.25)
    parser.add_argument("--apache-lambda-bg", type=float, default=0.05)
    parser.add_argument("--apache-lambda-geo", type=float, default=1.0)
    parser.add_argument("--apache-lambda-rel", type=float, default=1.0)
    parser.add_argument("--apache-bg-topk", type=int, default=5)
    parser.add_argument("--apache-swap-margin", type=float, default=0.05)
    parser.add_argument("--apache-pair-negative-floor", type=float, default=0.30)
    parser.add_argument("--negative-pair-weight", type=float, default=0.25)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Global training seed. Defaults to 42 for apache-reference and "
            "20260929 for the legacy recipe."
        ),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}"
        )

    vocabulary = RelationVocabulary.load(args.vocabulary)
    if bool(args.train_annotations) == bool(args.train_mixture):
        raise ValueError(
            "exactly one of --train-annotations or --train-mixture is required"
        )

    training_mixture = None
    if args.train_mixture:
        training_mixture = load_relation_mixture(
            args.train_mixture,
            vocabulary,
        )
        train_manifest = training_mixture.combined_manifest
    else:
        train_manifest = DatasetManifest.load(
            args.train_annotations,
            vocabulary,
        )

    validation_manifest = DatasetManifest.load(
        args.validation_annotations,
        vocabulary,
    )
    if training_mixture is not None:
        validate_mixture_disjoint_validation(
            training_mixture,
            validation_manifest,
            validation_image_root=args.image_root,
        )
    else:
        validate_disjoint_splits(
            train_manifest,
            validation_manifest,
        )

    apache_mode = (
        args.predicate_objective == "apache-reference"
    )
    reference_training = (
        args.training_recipe == "apache-reference"
    )
    resolved_seed = (
        int(args.seed)
        if args.seed is not None
        else (RELEASED_SEED if reference_training else 20260929)
    )
    resolved_epochs = resolve_training_epochs(
        args.epochs,
        recipe=args.training_recipe,
    )
    resolved_augment = resolve_training_augment(
        args.augment,
        recipe=args.training_recipe,
    )
    resolved_batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else (32 if reference_training else 4)
    )
    resolved_max_boxes = resolve_training_max_boxes(
        args.max_boxes,
        recipe=args.training_recipe,
    )
    resolved_hidden_dim = resolve_training_hidden_dim(
        args.hidden_dim,
        recipe=args.training_recipe,
    )
    if resolved_batch_size <= 0:
        raise ValueError("batch size must be positive")
    if (
        isinstance(args.apache_grad_accum, bool)
        or args.apache_grad_accum <= 0
    ):
        raise ValueError(
            "--apache-grad-accum must be a positive integer"
        )
    if (
        not torch.isfinite(torch.tensor(args.apache_cfa_prob))
        or args.apache_cfa_prob < 0.0
        or args.apache_cfa_prob > 1.0
    ):
        raise ValueError(
            "--apache-cfa-prob must be within [0,1]"
        )
    if (
        not torch.isfinite(torch.tensor(args.apache_cfa_alpha))
        or args.apache_cfa_alpha <= 0.0
    ):
        raise ValueError(
            "--apache-cfa-alpha must be positive"
        )
    if reference_training and not apache_mode:
        raise ValueError(
            "--training-recipe apache-reference requires "
            "--predicate-objective apache-reference"
        )
    if apache_mode:
        if (
            args.pair_evidence_contract != "apache"
            or args.pair_sampler_contract != "apache"
            or args.relation_context_contract != "apache"
            or args.predicate_head_contract != "apache"
        ):
            raise ValueError(
                "apache-reference objective requires the full Apache "
                "pair-evidence/sampler/context/predicate-head stack"
            )
        required_assets = {
            "--apache-ontology-meta": args.apache_ontology_meta,
            "--apache-ontology-npz": args.apache_ontology_npz,
            "--apache-source-column-allow": (
                args.apache_source_column_allow
            ),
            "--apache-neg-rate-table": args.apache_neg_rate_table,
        }
        for flag, value in required_assets.items():
            if not value:
                raise ValueError(
                    f"{flag} is required for apache-reference"
                )
        if not vocabulary.object_labels:
            raise ValueError(
                "Apache reference sampler requires vocabulary objects "
                "for pair-opportunity PU weighting"
            )
        if args.apache_lambda_obj > 0.0:
            if not args.apache_object_embeddings:
                raise ValueError(
                    "--apache-object-embeddings is required when "
                    "--apache-lambda-obj > 0"
                )
        if (
            args.holdout_predicate
            or args.mask_zero_support_predicates
            or args.zero_support_negative_weight != 1.0
        ):
            raise ValueError(
                "legacy holdout/zero-support knobs are not part of the "
                "Apache reference objective"
            )
        if (
            args.predicate_positive_weight_mode != "none"
            or args.predicate_contrastive_hard_negative_count != 0
            or args.predicate_calibration_loss_weight != 0.0
        ):
            raise ValueError(
                "legacy predicate weighting/contrast/calibration knobs "
                "cannot be combined with apache-reference"
            )

    baseline_config = FrozenBaselineConfig(
        epochs=resolved_epochs,
        batch_size=resolved_batch_size,
        learning_rate=(
            args.apache_head_lr
            if reference_training
            else args.learning_rate
        ),
        weight_decay=args.weight_decay,
        seed=resolved_seed,
    )
    model_config = RelationModelConfig(
        image_size=args.image_size,
        max_boxes=resolved_max_boxes,
        pair_budget=args.pair_budget,
        hidden_dim=resolved_hidden_dim,
        geometry_dim=args.geometry_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.0,
        tap_indices=(-6, -3, -1),
        predicate_adapter_rank=args.predicate_adapter_rank,
        pair_visual_evidence=args.pair_visual_evidence,
        pair_geometry_evidence=args.pair_geometry_evidence,
        pair_evidence_contract=args.pair_evidence_contract,
        pair_sampler_contract=args.pair_sampler_contract,
        relation_context_contract=args.relation_context_contract,
        predicate_head_contract=args.predicate_head_contract,
        apache_context_dropout=args.apache_context_dropout,
        apache_box_token_dropout=args.apache_box_token_dropout,
        apache_pair_negative_floor=args.apache_pair_negative_floor,
        apache_cfa_prob=(
            args.apache_cfa_prob
            if reference_training
            else 0.0
        ),
        apache_cfa_alpha=args.apache_cfa_alpha,
        allow_training_multiscale=(
            reference_training
            and bool(args.apache_multi_scale)
        ),
    )
    loss_config = RelationLossConfig(
        sampler_loss_weight=args.sampler_loss_weight,
        pair_loss_weight=args.pair_loss_weight,
        predicate_loss_weight=args.predicate_loss_weight,
        negative_pair_weight=args.negative_pair_weight,
        predicate_objective=(
            "bce"
            if apache_mode
            else args.predicate_objective
        ),
        predicate_contrastive_temperature=(
            args.predicate_contrastive_temperature
        ),
        predicate_contrastive_hard_negative_count=(
            0
            if apache_mode
            else args.predicate_contrastive_hard_negative_count
        ),
        predicate_calibration_loss_weight=(
            0.0
            if apache_mode
            else args.predicate_calibration_loss_weight
        ),
    )
    apache_objective_config = (
        ApacheObjectiveConfig(
            infonce_temp=args.predicate_contrastive_temperature,
            n_neg=args.apache_n_neg,
            hard_frac=args.apache_hard_frac,
            lambda_obj=args.apache_lambda_obj,
            lambda_swap=args.apache_lambda_swap,
            lambda_sigmoid=args.apache_lambda_sigmoid,
            lambda_bg=args.apache_lambda_bg,
            lambda_geo=args.apache_lambda_geo,
            lambda_rel=args.apache_lambda_rel,
            bg_topk=args.apache_bg_topk,
            swap_margin=args.apache_swap_margin,
            pair_negative_floor=args.apache_pair_negative_floor,
        )
        if apache_mode
        else None
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
    if (
        args.holdout_row_policy != "dimension-only"
        and not holdout_names
    ):
        raise ValueError(
            "--holdout-row-policy skip-holdout-only requires "
            "--holdout-predicate"
        )
    if (
        args.predicate_objective == "batch-local-infonce"
        and args.zero_support_negative_weight != 1.0
    ):
        raise ValueError(
            "--zero-support-negative-weight is BCE-only; "
            "batch-local-infonce does not consume BCE negative weights"
        )
    if (
        args.predicate_objective != "batch-local-infonce"
        and args.predicate_contrastive_hard_negative_count != 0
    ):
        raise ValueError(
            "--predicate-contrastive-hard-negative-count is "
            "batch-local-infonce-only"
        )
    if (
        args.predicate_objective != "batch-local-infonce"
        and args.predicate_calibration_loss_weight != 0.0
    ):
        raise ValueError(
            "--predicate-calibration-loss-weight is "
            "batch-local-infonce-only"
        )
    if (
        args.predicate_head_contract == "apache"
        and not args.predicate_spatial_flags
    ):
        raise ValueError(
            "--predicate-spatial-flags is required for the Apache "
            "predicate-head routing warm start"
        )
    if (
        args.predicate_head_contract != "apache"
        and args.predicate_spatial_flags
    ):
        raise ValueError(
            "--predicate-spatial-flags requires "
            "--predicate-head-contract apache"
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
    explicit_holdout_mask = None
    if holdout_indices:
        predicate_supervision_mask = torch.ones(
            len(vocabulary.predicates),
            dtype=torch.bool,
        )
        predicate_supervision_mask[
            list(holdout_indices)
        ] = False
        explicit_holdout_mask = ~predicate_supervision_mask
        predicate_supervision_mode = "explicit-holdout"
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
    predicate_contrastive_negative_mask = torch.tensor(
        [
            count > 0
            and index in supervised_predicate_indices
            for index, count in enumerate(
                predicate_weighting.predicate_positive_counts
            )
        ],
        dtype=torch.bool,
    )
    if (
        args.predicate_objective == "batch-local-infonce"
        and args.predicate_contrastive_hard_negative_count > 0
        and not predicate_contrastive_negative_mask.any()
    ):
        raise ValueError(
            "hard-negative InfoNCE requires at least one "
            "train-supported supervised predicate"
        )

    seed_everything(baseline_config.seed)
    predicate_embeddings = load_predicate_embeddings(
        args.predicate_embeddings,
        len(vocabulary.predicates),
    )
    validate_training_text_dim(
        int(predicate_embeddings.shape[1]),
        recipe=args.training_recipe,
    )

    apache_ontology = None
    apache_source_names: tuple[str, ...] = ()
    apache_source_allow = None
    apache_object_embeddings = None
    apache_pair_opportunity = None
    apache_asset_report = None
    if apache_mode:
        apache_ontology = PredicateOntology.from_soft_supervision(
            args.apache_ontology_meta,
            args.apache_ontology_npz,
        )
        if apache_ontology.predicates != vocabulary.predicates:
            raise ValueError(
                "Apache ontology predicate order must match vocabulary"
            )
        (
            apache_source_names,
            apache_source_allow,
        ) = load_source_column_allow(
            args.apache_source_column_allow,
            vocabulary.predicates,
        )
        if (
            training_mixture is not None
            and tuple(apache_source_names)
            != training_mixture.config.source_names
        ):
            raise ValueError(
                "Apache source-column table order must exactly match "
                "training mixture source order"
            )
        max_source_id = max(
            example.source_id
            for example in train_manifest.examples
        )
        if max_source_id >= len(apache_source_names):
            raise ValueError(
                "training source_id exceeds Apache source-column table"
            )
        apache_pair_opportunity = PairOpportunityTable.load(
            args.apache_neg_rate_table
        )
        if (
            apache_pair_opportunity.num_cats
            != len(vocabulary.object_labels)
        ):
            raise ValueError(
                "Apache pair-opportunity num_cats must match "
                "vocabulary object-label count"
            )
        if args.apache_lambda_obj > 0.0:
            apache_object_embeddings = load_predicate_embeddings(
                args.apache_object_embeddings,
                len(vocabulary.object_labels),
            )
            if (
                apache_object_embeddings.shape[1]
                != predicate_embeddings.shape[1]
            ):
                raise ValueError(
                    "object/predicate embeddings must share text dimension"
                )
        apache_asset_report = {
            "ontology_meta_sha256": sha256(
                Path(args.apache_ontology_meta)
            ),
            "ontology_npz_sha256": sha256(
                Path(args.apache_ontology_npz)
            ),
            "source_column_allow_sha256": sha256(
                Path(args.apache_source_column_allow)
            ),
            "source_names": list(apache_source_names),
            "ontology": apache_ontology.stats(),
            "neg_rate_table_sha256": sha256(
                Path(args.apache_neg_rate_table)
            ),
            "pair_opportunity": (
                apache_pair_opportunity.stats(
                    negative_floor=args.apache_pair_negative_floor
                )
                if apache_pair_opportunity is not None
                else None
            ),
            "object_label_order": list(vocabulary.object_labels),
            "object_embeddings_sha256": (
                sha256(Path(args.apache_object_embeddings))
                if args.apache_object_embeddings
                else None
            ),
        }

    backbone = TimmDinoV3Backbone.from_pretrained(
        args.backbone,
        train_backbone=False,
    )
    model = KFRelationModel(
        backbone,
        predicate_embeddings,
        model_config,
    )
    if not reference_training:
        freeze_backbone(model)

    device = resolve_device(args.device)
    model.to(device)

    backbone_provenance_initial = backbone_provenance(
        model,
        model_name=args.backbone,
        mode=("full" if reference_training else "frozen"),
    )

    if apache_mode:
        if (
            model.apache_pair_sampler is None
            or apache_pair_opportunity is None
        ):
            raise RuntimeError(
                "Apache reference sampler/table is unavailable"
            )
        model.apache_pair_sampler.set_negative_rates(
            apache_pair_opportunity.rate.to(device),
            apache_pair_opportunity.trusted.to(device),
            apache_pair_opportunity.num_cats,
        )

    routing_warm_start = None
    if args.predicate_head_contract == "apache":
        spatial_flags = load_predicate_spatial_flags(
            args.predicate_spatial_flags,
            vocabulary.predicates,
        ).to(device)
        assert model.apache_vocab_head is not None
        bank = model.predicate_bank.detach()
        with torch.no_grad():
            alpha_before = model.apache_vocab_head.routing_alpha(
                bank
            )
        target_alpha = balanced_spatial_probe_targets(
            bank,
            spatial_flags,
            steps=1000,
            learning_rate=5.0e-2,
        )
        with torch.no_grad():
            mse_before = torch.nn.functional.mse_loss(
                alpha_before,
                target_alpha,
            ).item()
        final_mse = model.apache_vocab_head.warm_start_gate(
            bank,
            target_alpha,
            steps=300,
            learning_rate=1.0e-2,
        )
        with torch.no_grad():
            alpha_after = model.apache_vocab_head.routing_alpha(
                bank
            )
            mse_after = torch.nn.functional.mse_loss(
                alpha_after,
                target_alpha,
            ).item()
        routing_warm_start = {
            "schema": "kfcore.apache-routing-warm-start/1",
            "spatial_flags_sha256": sha256(
                Path(args.predicate_spatial_flags)
            ),
            "spatial_count": int(spatial_flags.sum().item()),
            "semantic_count": int((~spatial_flags).sum().item()),
            "target_alpha_mean": float(
                target_alpha.mean().detach().cpu()
            ),
            "target_alpha_spatial_mean": float(
                target_alpha[spatial_flags].mean().detach().cpu()
            ),
            "target_alpha_semantic_mean": float(
                target_alpha[~spatial_flags].mean().detach().cpu()
            ),
            "mse_before": float(mse_before),
            "mse_after": float(mse_after),
            "reported_final_mse": float(final_mse),
        }
        if not mse_after < mse_before:
            raise RuntimeError(
                "Apache predicate routing warm start did not improve MSE"
            )

    apache_objective = None
    if apache_mode:
        assert apache_ontology is not None
        assert apache_source_allow is not None
        assert apache_objective_config is not None
        apache_objective = ApacheReferenceObjective(
            apache_ontology,
            config=apache_objective_config,
            source_column_allow=apache_source_allow,
            object_text_bank=apache_object_embeddings,
        )

    ema = (
        ModelEMA(
            model,
            decay=RELEASED_EMA_DECAY,
        )
        if reference_training
        else None
    )

    if training_mixture is not None:
        train_dataset = RelationMixtureTrainingDataset(
            training_mixture,
            image_size=model.config.image_size,
            max_boxes=model.config.max_boxes,
            predicate_count=len(vocabulary.predicates),
            object_labels=(
                vocabulary.object_labels
                if apache_mode
                else ()
            ),
            augment=resolved_augment,
        )
    else:
        train_dataset = RelationTrainingDataset(
            train_manifest,
            image_root=args.image_root,
            image_size=model.config.image_size,
            max_boxes=model.config.max_boxes,
            predicate_count=len(vocabulary.predicates),
            object_labels=(
                vocabulary.object_labels
                if apache_mode
                else ()
            ),
            augment=resolved_augment,
        )
    multi_scale_resolutions = None
    if reference_training and args.apache_multi_scale:
        parts = [
            value.strip()
            for value in args.apache_multi_scale.split(",")
        ]
        if len(parts) != 2:
            raise ValueError(
                "--apache-multi-scale must be 'lo,hi' or empty"
            )
        lo, hi = (float(value) for value in parts)
        multi_scale_resolutions = scale_ladder(
            model.config.image_size,
            lo,
            hi,
            args.apache_multi_scale_n,
            patch=int(model.backbone.patch_size),
        )

    mixture_sampler = None
    mixture_batch_sampler = None
    mixture_draw_sampler = None
    mixture_report = None
    if training_mixture is not None:
        mixture_weights = sample_weights_from_fractions(
            training_mixture.source_of_index,
            training_mixture.config.fractions,
        )
        if reference_training:
            mixture_batch_sampler = ApacheReleasedBatchSampler(
                mixture_weights,
                resolutions=(
                    multi_scale_resolutions
                    if multi_scale_resolutions is not None
                    else ()
                ),
                num_samples=training_mixture.draws_per_epoch,
                batch_size=resolved_batch_size,
                world_size=RELEASED_WORLD_SIZE,
                seed=training_mixture.config.seed,
            )
            mixture_draw_sampler = (
                mixture_batch_sampler.global_sampler
            )
        else:
            mixture_sampler = DistributedWeightedSampler(
                mixture_weights,
                num_replicas=1,
                rank=0,
                num_samples=training_mixture.draws_per_epoch,
                seed=training_mixture.config.seed,
            )
            mixture_draw_sampler = mixture_sampler

        mixture_report = training_mixture.report()
        if mixture_batch_sampler is not None:
            runtime_matches = (
                mixture_batch_sampler.matches_released_topology()
                and args.apache_grad_accum == RELEASED_WORLD_SIZE
                and resolved_seed == RELEASED_SEED
            )
            mixture_report["runtime_topology"] = {
                "mode": "single-process-logical-ddp",
                "world_size": RELEASED_WORLD_SIZE,
                "micro_batch_size": resolved_batch_size,
                "grad_accum": args.apache_grad_accum,
                "global_training_seed": resolved_seed,
                "sampler_seed": training_mixture.config.seed,
                "optimizer_steps_per_epoch": (
                    mixture_batch_sampler.optimizer_steps_per_epoch
                ),
                "matches_released_topology": runtime_matches,
            }
            mixture_report["matches_released_sampling_stream"] = (
                bool(mixture_report["matches_released_mixture"])
                and runtime_matches
            )

    loader = make_training_loader(
        train_dataset,
        baseline_config,
        resolutions=(
            None
            if mixture_batch_sampler is not None
            else multi_scale_resolutions
        ),
        drop_last=reference_training,
        sampler=mixture_sampler,
        batch_sampler=mixture_batch_sampler,
    )
    reference_recipe_config = None
    optimizer_report = None
    scheduler_report = None
    scheduler = None
    if reference_training:
        reference_recipe_config = ApacheTrainingRecipeConfig(
            head_lr=args.apache_head_lr,
            backbone_lr=args.apache_backbone_lr,
            weight_decay=args.weight_decay,
            epochs=resolved_epochs,
            warmup_steps=args.apache_warmup_steps,
            min_lr_factor=args.apache_min_lr_factor,
            clip_grad=args.apache_clip_grad,
            backbone_mode="full",
            micro_batch_size=resolved_batch_size,
            grad_accum=args.apache_grad_accum,
            multi_scale=(
                args.apache_multi_scale
                if args.apache_multi_scale
                else ""
            ),
            multi_scale_n=args.apache_multi_scale_n,
            cfa_prob=args.apache_cfa_prob,
            cfa_alpha=args.apache_cfa_alpha,
            augment=resolved_augment,
            text_dim=int(predicate_embeddings.shape[1]),
        )
        optimizer, optimizer_report = build_reference_optimizer(
            model,
            reference_recipe_config,
        )
        scheduler, scheduler_report = build_reference_scheduler(
            optimizer,
            reference_recipe_config,
            steps_per_epoch=(
                len(loader)
                + args.apache_grad_accum
                - 1
            ) // args.apache_grad_accum,
        )
    else:
        parameters = trainable_parameters(model)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=baseline_config.learning_rate,
            weight_decay=baseline_config.weight_decay,
        )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    backbone_trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    )

    augmentation_report = {
        "kind": "brightness-contrast-saturation",
        "strength": resolved_augment,
        "horizontal_flip": False,
        "geometry_transform": False,
        "rng_source": "ambient-torch-rng",
        "rng_equivalence": (
            "stochastic-distribution"
            if reference_training
            else "disabled"
        ),
        "worker_trajectory_equivalence": False,
    }

    history: list[dict[str, object]] = []
    for epoch in range(1, baseline_config.epochs + 1):
        batch_sampler = getattr(
            loader,
            "batch_sampler",
            None,
        )
        loader_sampler = getattr(
            loader,
            "sampler",
            None,
        )
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch - 1)
        elif hasattr(loader_sampler, "set_epoch"):
            loader_sampler.set_epoch(epoch - 1)

        losses = train_epoch(
            model,
            loader,
            optimizer,
            device=device,
            loss_config=loss_config,
            predicate_positive_weights=predicate_positive_weights,
            predicate_supervision_mask=predicate_supervision_mask,
            predicate_negative_weights=predicate_negative_weights,
            explicit_holdout_mask=explicit_holdout_mask,
            explicit_holdout_row_policy=args.holdout_row_policy,
            predicate_contrastive_negative_mask=(
                predicate_contrastive_negative_mask
            ),
            apache_objective=apache_objective,
            backbone_training=reference_training,
            scheduler=scheduler,
            clip_grad=(
                reference_recipe_config.clip_grad
                if reference_recipe_config is not None
                else None
            ),
            record_gradient_health=reference_training,
            grad_accum=(
                args.apache_grad_accum
                if reference_training
                else 1
            ),
            ema=ema,
        )
        epoch_report: dict[str, object] = {
            "epoch": epoch,
            **losses,
        }
        if (
            mixture_draw_sampler is not None
            and training_mixture is not None
        ):
            epoch_report["source_mixture_draw"] = (
                realized_source_draws(
                    mixture_draw_sampler,
                    training_mixture.source_of_index,
                    len(training_mixture.config.sources),
                )
            )
        history.append(epoch_report)
        print(json.dumps(history[-1], sort_keys=True))

    backbone_provenance_final_raw = backbone_provenance(
        model,
        model_name=args.backbone,
        mode=("full" if reference_training else "frozen"),
    )
    artifact_model, artifact_weight_source = select_artifact_model(
        model,
        ema,
        recipe=args.training_recipe,
    )
    backbone_provenance_final = backbone_provenance(
        artifact_model,
        model_name=args.backbone,
        mode=(
            "full-ema"
            if reference_training
            else "frozen"
        ),
    )
    ema_report = (
        ema.report()
        if ema is not None
        else {
            "enabled": False,
            "weights_source": "raw",
        }
    )
    if ema is not None:
        ema_report["raw_state_sha256"] = (
            module_state_sha256(model)
        )

    benchmark_report = evaluate_gt_boxes(
        artifact_model,
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

    source_bank = artifact_model.predicate_bank.detach().cpu()
    effective_bank = artifact_model.effective_predicate_bank().detach().cpu()
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
        artifact_model,
        backbone_model=args.backbone,
        predicates=list(vocabulary.predicates),
        extra={
            "training_schema": (
                "kfcore.relation-apache-reference-training/1"
                if reference_training
                else "kfcore.relation-frozen-baseline/1"
            ),
            "frozen_backbone": not reference_training,
            "training_recipe": {
                "name": args.training_recipe,
                "released_epoch_target": 12,
                "matches_released_epoch_count": resolved_epochs == 12,
                "config": (
                    reference_recipe_config.__dict__
                    if reference_recipe_config is not None
                    else None
                ),
                "effective_batch_size": (
                    reference_recipe_config.effective_batch_size
                    if reference_recipe_config is not None
                    else None
                ),
                "multi_scale_resolutions": (
                    list(multi_scale_resolutions)
                    if multi_scale_resolutions is not None
                    else None
                ),
                "optimizer": optimizer_report,
                "scheduler": scheduler_report,
                "backbone_trainable_parameter_count": (
                    backbone_trainable_parameter_count
                ),
                "backbone_initial": backbone_provenance_initial,
                "backbone_final_raw": backbone_provenance_final_raw,
                "backbone_final": backbone_provenance_final,
                "weight_source": artifact_weight_source,
                "ema": ema_report,
                "augmentation": augmentation_report,
                "source_mixture": mixture_report,
            },
            "train_annotations_sha256": train_manifest.annotations_sha256,
            "train_mixture": mixture_report,
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
                "holdout_row_policy": args.holdout_row_policy,
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
            "routing_warm_start": routing_warm_start,
            "apache_reference_objective": (
                {
                    "config": (
                        apache_objective_config.__dict__
                        if apache_objective_config is not None
                        else None
                    ),
                    "assets": apache_asset_report,
                }
                if apache_mode
                else None
            ),
            "history": history,
        },
    )

    training_report = {
        "schema": "kfcore.relation-training-run/1",
        "backbone": args.backbone,
        "frozen_backbone": not reference_training,
        "training_recipe": {
            "name": args.training_recipe,
            "released_epoch_target": 12,
            "matches_released_epoch_count": resolved_epochs == 12,
            "config": (
                reference_recipe_config.__dict__
                if reference_recipe_config is not None
                else None
            ),
            "effective_batch_size": (
                reference_recipe_config.effective_batch_size
                if reference_recipe_config is not None
                else None
            ),
            "multi_scale_resolutions": (
                list(multi_scale_resolutions)
                if multi_scale_resolutions is not None
                else None
            ),
            "optimizer": optimizer_report,
            "scheduler": scheduler_report,
            "backbone_trainable_parameter_count": (
                backbone_trainable_parameter_count
            ),
            "backbone_final_raw": backbone_provenance_final_raw,
            "backbone_final": backbone_provenance_final,
            "weight_source": artifact_weight_source,
            "ema": ema_report,
            "augmentation": augmentation_report,
            "source_mixture": mixture_report,
        },
        "device": str(device),
        "train_examples": len(train_manifest.examples),
        "validation_examples": len(validation_manifest.examples),
        "train_annotations_sha256": train_manifest.annotations_sha256,
        "train_mixture": mixture_report,
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
            "pair_visual_evidence": model.config.pair_visual_evidence,
            "pair_geometry_evidence": model.config.pair_geometry_evidence,
            "pair_evidence_contract": model.config.pair_evidence_contract,
            "pair_sampler_contract": model.config.pair_sampler_contract,
            "relation_context_contract": (
                model.config.relation_context_contract
            ),
            "predicate_head_contract": model.config.predicate_head_contract,
            "apache_context_dropout": model.config.apache_context_dropout,
            "apache_box_token_dropout": (
                model.config.apache_box_token_dropout
            ),
            "apache_pair_negative_floor": (
                model.config.apache_pair_negative_floor
            ),
            "apache_cfa_prob": model.config.apache_cfa_prob,
            "apache_cfa_alpha": model.config.apache_cfa_alpha,
            "allow_training_multiscale": (
                model.config.allow_training_multiscale
            ),
        },
        "baseline_config": config_payload(baseline_config),
        "loss_config": {
            "sampler_loss_weight": loss_config.sampler_loss_weight,
            "pair_loss_weight": loss_config.pair_loss_weight,
            "predicate_loss_weight": loss_config.predicate_loss_weight,
            "negative_pair_weight": loss_config.negative_pair_weight,
            "predicate_objective": loss_config.predicate_objective,
            "predicate_contrastive_temperature": (
                loss_config.predicate_contrastive_temperature
            ),
            "predicate_contrastive_hard_negative_count": (
                loss_config.predicate_contrastive_hard_negative_count
            ),
            "predicate_calibration_loss_weight": (
                loss_config.predicate_calibration_loss_weight
            ),
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
            "holdout_row_policy": args.holdout_row_policy,
            "contrastive_negative_candidate_indices": [
                index
                for index, enabled in enumerate(
                    predicate_contrastive_negative_mask.tolist()
                )
                if enabled
            ],
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
        "routing_warm_start": routing_warm_start,
        "apache_reference_objective": (
            {
                "config": (
                    apache_objective_config.__dict__
                    if apache_objective_config is not None
                    else None
                ),
                "assets": apache_asset_report,
            }
            if apache_mode
            else None
        ),
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
