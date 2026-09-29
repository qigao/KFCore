from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from model import RelationTrainingOutputs


@dataclass(frozen=True)
class RelationLossConfig:
    sampler_loss_weight: float = 1.0
    pair_loss_weight: float = 1.0
    predicate_loss_weight: float = 1.0
    negative_pair_weight: float = 0.25
    predicate_objective: str = "bce"
    predicate_contrastive_temperature: float = 0.07
    predicate_contrastive_hard_negative_count: int = 0

    def __post_init__(self) -> None:
        if (
            self.sampler_loss_weight < 0.0
            or self.pair_loss_weight < 0.0
            or self.predicate_loss_weight < 0.0
        ):
            raise ValueError("loss weights must be non-negative")
        if not 0.0 <= self.negative_pair_weight <= 1.0:
            raise ValueError("negative_pair_weight must be within [0,1]")
        if self.predicate_objective not in {
            "bce",
            "batch-local-infonce",
        }:
            raise ValueError(
                "predicate_objective must be bce or batch-local-infonce"
            )
        if (
            not torch.isfinite(
                torch.tensor(self.predicate_contrastive_temperature)
            )
            or self.predicate_contrastive_temperature <= 0.0
        ):
            raise ValueError(
                "predicate_contrastive_temperature must be finite and positive"
            )
        if (
            isinstance(self.predicate_contrastive_hard_negative_count, bool)
            or not isinstance(
                self.predicate_contrastive_hard_negative_count, int
            )
            or self.predicate_contrastive_hard_negative_count < 0
        ):
            raise ValueError(
                "predicate_contrastive_hard_negative_count must be "
                "a non-negative integer"
            )


def _weighted_pair_bce(
    logits: Tensor,
    targets: Tensor,
    valid: Tensor,
    *,
    negative_pair_weight: float,
) -> Tensor:
    valid = valid.to(torch.bool)
    if not valid.any():
        return logits.sum() * 0.0

    selected_targets = targets[valid].to(logits.dtype)
    raw = F.binary_cross_entropy_with_logits(
        logits[valid],
        selected_targets,
        reduction="none",
    )
    weights = torch.where(
        selected_targets > 0.5,
        torch.ones_like(raw),
        torch.full_like(raw, negative_pair_weight),
    )
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def _validate_targets(
    pair_targets: Tensor,
    predicate_targets: Tensor,
    pred_logits: Tensor,
) -> None:
    if pair_targets.ndim != 3:
        raise ValueError("pair_targets must be [B,N,N]")
    if pair_targets.shape[1] != pair_targets.shape[2]:
        raise ValueError("relation target object dimensions must be square")
    if predicate_targets.ndim != 4:
        raise ValueError("predicate_targets must be [B,N,N,V]")
    if predicate_targets.shape[:3] != pair_targets.shape:
        raise ValueError(
            "predicate target batch/object dimensions must match pair targets"
        )
    if predicate_targets.shape[3] != pred_logits.shape[2]:
        raise ValueError(
            "predicate target vocabulary width does not match pred_logits"
        )
    if pair_targets.shape[0] != pred_logits.shape[0]:
        raise ValueError("target and output batch sizes must match")
    if not torch.isfinite(pair_targets).all():
        raise ValueError("pair_targets must be finite")
    if not torch.isfinite(predicate_targets).all():
        raise ValueError("predicate_targets must be finite")
    if ((pair_targets < 0) | (pair_targets > 1)).any():
        raise ValueError("pair_targets must be within [0,1]")
    if ((predicate_targets < 0) | (predicate_targets > 1)).any():
        raise ValueError("predicate_targets must be within [0,1]")

    positive_pairs = pair_targets > 0.5
    predicate_positive = predicate_targets > 0.5
    predicate_count = predicate_positive.sum(dim=-1)
    if (positive_pairs & (predicate_count == 0)).any():
        raise ValueError(
            "every positive pair must have at least one predicate label"
        )
    if ((~positive_pairs) & (predicate_count > 0)).any():
        raise ValueError(
            "negative pairs must not carry positive predicate labels"
        )


def _batch_local_predicate_infonce(
    query: Tensor,
    raw_query: Tensor,
    predicate_bank: Tensor,
    positive_targets: Tensor,
    *,
    supervision_mask: Tensor | None,
    positive_weights: Tensor | None,
    negative_candidate_mask: Tensor | None,
    semantic_positive_weights: Tensor | None,
    hard_negative_count: int,
    temperature: float,
) -> dict[str, Tensor]:
    if query.ndim != 2 or raw_query.shape != query.shape:
        raise ValueError("predicate query tensors must be [M,D]")
    if predicate_bank.ndim != 2 or predicate_bank.shape[1] != query.shape[1]:
        raise ValueError("predicate bank must be [V,D] matching query width")
    if positive_targets.ndim != 2:
        raise ValueError("contrastive predicate targets must be [M,V]")
    if positive_targets.shape != (
        query.shape[0],
        predicate_bank.shape[0],
    ):
        raise ValueError(
            "contrastive predicate target shape does not match query/bank"
        )

    visible = positive_targets > 0.5
    if supervision_mask is not None:
        visible = visible & supervision_mask.view(1, -1)

    keep_rows = visible.any(dim=1)
    skipped = (~keep_rows).sum().to(dtype=query.dtype)
    if not keep_rows.any():
        zero = query.sum() * 0.0
        return {
            "loss": zero,
            "rows_skipped": skipped,
            "contrast_set_size": zero,
            "positive_cosine": zero,
            "hard_negative_margin": zero,
            "query_raw_norm": zero,
            "unobserved_column_fraction": query.new_tensor(1.0),
            "hard_negative_count": zero,
            "soft_positive_count": zero,
        }

    query = query[keep_rows]
    raw_query = raw_query[keep_rows]
    visible = visible[keep_rows]

    direct_weight = visible.to(dtype=query.dtype)
    if positive_weights is not None:
        direct_weight = direct_weight * positive_weights.to(
            device=query.device,
            dtype=query.dtype,
        ).view(1, -1)

    positive_strength = direct_weight
    soft_positive_count = query.new_zeros(())
    if semantic_positive_weights is not None:
        if (
            semantic_positive_weights.ndim != 2
            or semantic_positive_weights.shape
            != (
                predicate_bank.shape[0],
                predicate_bank.shape[0],
            )
        ):
            raise ValueError(
                "predicate_soft_positive_weights must be [V,V]"
            )
        if not torch.isfinite(semantic_positive_weights).all():
            raise ValueError(
                "predicate_soft_positive_weights must be finite"
            )
        if (
            (semantic_positive_weights < 0).any()
            or (semantic_positive_weights > 1).any()
        ):
            raise ValueError(
                "predicate_soft_positive_weights must be within [0,1]"
            )
        semantic = semantic_positive_weights.to(
            device=query.device,
            dtype=query.dtype,
        )
        # Propagate supervision mass from visible seed labels. This is
        # deliberately seed-weighted: an indirectly reached holdout column
        # never reads that holdout predicate's own train-frequency weight.
        expanded_weight = torch.matmul(
            direct_weight,
            semantic,
        )
        # Semantic propagation only creates new soft-positive columns.
        # Already observed direct positives keep their original class weight
        # and are not inflated by semantic edges from other direct labels.
        expanded_weight = expanded_weight * (~visible).to(
            dtype=query.dtype
        )
        positive_strength = direct_weight + expanded_weight
        semantic_only = (
            positive_strength > 0
        ) & ~visible
        soft_positive_count = semantic_only.sum(
            dim=1
        ).to(dtype=query.dtype).mean()

    contrast_mask = (positive_strength > 0).any(dim=0)
    positive_indices = torch.nonzero(
        contrast_mask, as_tuple=False
    ).flatten()
    if positive_indices.numel() <= 0:
        raise RuntimeError("contrastive predicate set unexpectedly empty")

    hard_negative_indices = positive_indices.new_empty((0,))
    if hard_negative_count > 0:
        if negative_candidate_mask is None:
            raise ValueError(
                "hard contrastive negatives require "
                "predicate_contrastive_negative_mask"
            )
        if (
            negative_candidate_mask.ndim != 1
            or negative_candidate_mask.shape[0] != predicate_bank.shape[0]
            or negative_candidate_mask.dtype != torch.bool
        ):
            raise ValueError(
                "predicate_contrastive_negative_mask must be bool [V]"
            )
        candidates = negative_candidate_mask.to(
            device=query.device
        ) & ~contrast_mask
        if supervision_mask is not None:
            candidates = candidates & supervision_mask.to(
                device=query.device
            )
        candidate_indices = torch.nonzero(
            candidates, as_tuple=False
        ).flatten()
        if candidate_indices.numel() > 0:
            count = min(
                hard_negative_count,
                int(candidate_indices.numel()),
            )
            with torch.no_grad():
                candidate_cosine = torch.matmul(
                    query.detach(),
                    predicate_bank[
                        candidate_indices
                    ].detach().transpose(0, 1),
                )
                hardness = candidate_cosine.max(dim=0).values
                selected = torch.topk(
                    hardness,
                    k=count,
                    largest=True,
                    sorted=True,
                ).indices
                hard_negative_indices = candidate_indices[
                    selected
                ]

    contrast_indices = torch.cat(
        (positive_indices, hard_negative_indices)
    ).unique(sorted=True)
    bank = predicate_bank[contrast_indices]
    cosine = torch.matmul(query, bank.transpose(0, 1))
    logits = cosine / float(temperature)
    positive_strength = positive_strength[
        :, contrast_indices
    ]
    positive = positive_strength > 0

    weights = positive_strength.to(dtype=logits.dtype)

    positive_mass = weights.sum(dim=1)
    if (positive_mass <= 0).any():
        raise ValueError(
            "contrastive predicate rows must retain positive supervision"
        )
    denominator = torch.logsumexp(logits, dim=1)
    per_positive = denominator.unsqueeze(1) - logits
    row_loss = (
        per_positive * weights
    ).sum(dim=1) / positive_mass
    loss = row_loss.mean()

    positive_cosine = (
        cosine * weights
    ).sum() / weights.sum().clamp_min(1.0)

    negative = ~positive
    has_negative = negative.any(dim=1)
    if has_negative.any():
        negative_cosine = cosine.masked_fill(
            ~negative,
            torch.finfo(cosine.dtype).min,
        ).max(dim=1).values
        positive_mean = (
            cosine * weights
        ).sum(dim=1) / positive_mass
        hard_negative_margin = (
            positive_mean[has_negative]
            - negative_cosine[has_negative]
        ).mean()
    else:
        hard_negative_margin = cosine.sum() * 0.0

    contrast_size = query.new_tensor(
        float(contrast_indices.numel())
    )
    unobserved_fraction = query.new_tensor(
        1.0
        - float(contrast_indices.numel())
        / float(predicate_bank.shape[0])
    )
    return {
        "loss": loss,
        "rows_skipped": skipped,
        "contrast_set_size": contrast_size,
        "positive_cosine": positive_cosine,
        "hard_negative_margin": hard_negative_margin,
        "query_raw_norm": raw_query.norm(dim=-1).mean(),
        "unobserved_column_fraction": unobserved_fraction,
        "hard_negative_count": query.new_tensor(
            float(hard_negative_indices.numel())
        ),
        "soft_positive_count": soft_positive_count,
    }


def supervised_relation_loss(
    outputs: RelationTrainingOutputs
    | tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    pair_targets: Tensor,
    predicate_targets: Tensor,
    config: RelationLossConfig = RelationLossConfig(),
    predicate_positive_weights: Tensor | None = None,
    predicate_supervision_mask: Tensor | None = None,
    predicate_negative_weights: Tensor | None = None,
    explicit_holdout_mask: Tensor | None = None,
    explicit_holdout_row_policy: str = "dimension-only",
    predicate_contrastive_negative_mask: Tensor | None = None,
    predicate_soft_positive_weights: Tensor | None = None,
) -> dict[str, Tensor]:
    """Exhaustive supervised multi-label relation baseline.

    pair_targets is [B,N,N] with 1 for annotated related pairs and 0 otherwise.
    predicate_targets is [B,N,N,V] multi-hot over the complete benchmark
    predicate vocabulary for each ordered pair.

    Predicate BCE is evaluated only on positive related pairs. Zeros in their
    multi-hot predicate vectors are supervised negatives by default. Optional
    [V] negative weights scale only target=0 predicate BCE terms; positive
    target terms always keep multiplier 1. An optional bool [V] supervision
    mask can exclude predicate dimensions entirely. Positive targets may be
    hidden only when the same dimensions are explicitly declared through
    explicit_holdout_mask. The optional explicit holdout row policy can skip
    predicate BCE for positive pairs whose known positive predicate labels are
    entirely held out, while leaving pair/sampler supervision unchanged.

    The dense sampler loss is training-only. Runtime/export keeps the existing
    five-output ABI. Unannotated valid pairs are down-weighted negatives through
    negative_pair_weight.
    """
    if explicit_holdout_row_policy not in (
        "dimension-only",
        "skip-holdout-only",
    ):
        raise ValueError(
            "explicit_holdout_row_policy must be dimension-only "
            "or skip-holdout-only"
        )

    training_outputs = (
        outputs if isinstance(outputs, RelationTrainingOutputs) else None
    )
    if training_outputs is None:
        if config.sampler_loss_weight > 0.0:
            raise ValueError(
                "sampler loss requires KFRelationModel.forward_training outputs"
            )
        runtime = outputs
    else:
        runtime = training_outputs.runtime

    pred_logits, pair_logits, sub_idx, obj_idx, valid_mask = runtime
    _validate_targets(pair_targets, predicate_targets, pred_logits)
    if pair_logits.shape != sub_idx.shape or pair_logits.shape != obj_idx.shape:
        raise ValueError("pair output shapes do not match")
    if valid_mask.shape != pair_logits.shape:
        raise ValueError("valid_mask shape does not match pair logits")

    batch = torch.arange(
        pred_logits.shape[0], device=pred_logits.device, dtype=torch.int64
    ).unsqueeze(1)
    selected_pair_targets = pair_targets[
        batch, sub_idx.to(torch.int64), obj_idx.to(torch.int64)
    ].to(pair_logits.dtype)
    selected_predicate_targets = predicate_targets[
        batch, sub_idx.to(torch.int64), obj_idx.to(torch.int64), :
    ].to(pred_logits.dtype)

    valid = valid_mask.to(torch.bool)
    pair_loss = _weighted_pair_bce(
        pair_logits,
        selected_pair_targets,
        valid,
        negative_pair_weight=config.negative_pair_weight,
    )

    positive = valid & (selected_pair_targets > 0.5)
    pos_weight = None
    if predicate_positive_weights is not None:
        if predicate_positive_weights.ndim != 1:
            raise ValueError(
                "predicate_positive_weights must be [V]"
            )
        if predicate_positive_weights.shape[0] != pred_logits.shape[2]:
            raise ValueError(
                "predicate_positive_weights width does not match pred_logits"
            )
        if not torch.isfinite(predicate_positive_weights).all():
            raise ValueError(
                "predicate_positive_weights must be finite"
            )
        if (predicate_positive_weights <= 0).any():
            raise ValueError(
                "predicate_positive_weights must be positive"
            )
        pos_weight = predicate_positive_weights.to(
            device=pred_logits.device,
            dtype=pred_logits.dtype,
        )

    supervision_mask = None
    if predicate_supervision_mask is not None:
        if predicate_supervision_mask.ndim != 1:
            raise ValueError(
                "predicate_supervision_mask must be [V]"
            )
        if predicate_supervision_mask.shape[0] != pred_logits.shape[2]:
            raise ValueError(
                "predicate_supervision_mask width does not match pred_logits"
            )
        if predicate_supervision_mask.dtype != torch.bool:
            raise ValueError(
                "predicate_supervision_mask must have bool dtype"
            )
        supervision_mask = predicate_supervision_mask.to(
            device=pred_logits.device,
        )
        if not supervision_mask.any():
            raise ValueError(
                "predicate_supervision_mask must supervise at least one predicate"
            )

    holdout_mask = None
    if explicit_holdout_mask is not None:
        if supervision_mask is None:
            raise ValueError(
                "explicit_holdout_mask requires predicate_supervision_mask"
            )
        if explicit_holdout_mask.ndim != 1:
            raise ValueError("explicit_holdout_mask must be [V]")
        if explicit_holdout_mask.shape[0] != pred_logits.shape[2]:
            raise ValueError(
                "explicit_holdout_mask width does not match pred_logits"
            )
        if explicit_holdout_mask.dtype != torch.bool:
            raise ValueError("explicit_holdout_mask must have bool dtype")
        holdout_mask = explicit_holdout_mask.to(
            device=pred_logits.device,
        )
        if not holdout_mask.any():
            raise ValueError(
                "explicit_holdout_mask must contain at least one predicate"
            )
        if (holdout_mask & supervision_mask).any():
            raise ValueError(
                "explicit holdout predicates must be excluded from supervision"
            )

    if (
        explicit_holdout_row_policy == "skip-holdout-only"
        and holdout_mask is None
    ):
        raise ValueError(
            "skip-holdout-only requires an explicit_holdout_mask"
        )

    negative_weights = None
    if predicate_negative_weights is not None:
        if predicate_negative_weights.ndim != 1:
            raise ValueError(
                "predicate_negative_weights must be [V]"
            )
        if predicate_negative_weights.shape[0] != pred_logits.shape[2]:
            raise ValueError(
                "predicate_negative_weights width does not match pred_logits"
            )
        if not torch.isfinite(predicate_negative_weights).all():
            raise ValueError(
                "predicate_negative_weights must be finite"
            )
        if (predicate_negative_weights < 0).any():
            raise ValueError(
                "predicate_negative_weights must be non-negative"
            )
        negative_weights = predicate_negative_weights.to(
            device=pred_logits.device,
            dtype=pred_logits.dtype,
        )

    predicate_rows = positive.sum().to(dtype=pred_logits.dtype)
    predicate_rows_skipped = pred_logits.new_zeros(())
    predicate_contrast_set_size = pred_logits.new_zeros(())
    predicate_positive_cosine = pred_logits.new_zeros(())
    predicate_hard_negative_margin = pred_logits.new_zeros(())
    predicate_query_raw_norm = pred_logits.new_zeros(())
    predicate_unobserved_column_fraction = pred_logits.new_zeros(())
    predicate_hard_negative_count = pred_logits.new_zeros(())
    predicate_soft_positive_count = pred_logits.new_zeros(())

    if positive.any():
        positive_logits = pred_logits[positive]
        positive_targets = selected_predicate_targets[positive]
        if supervision_mask is not None:
            hidden_positive = (
                positive_targets > 0.5
            ) & (~supervision_mask).view(1, -1)
            allowed_hidden = (
                holdout_mask.view(1, -1)
                if holdout_mask is not None
                else torch.zeros_like(hidden_positive)
            )
            if (hidden_positive & ~allowed_hidden).any():
                raise ValueError(
                    "predicate supervision mask cannot hide positive labels"
                )

        if config.predicate_objective == "batch-local-infonce":
            if training_outputs is None:
                raise ValueError(
                    "batch-local-infonce requires forward_training outputs"
                )
            if (
                negative_weights is not None
                and not torch.allclose(
                    negative_weights,
                    torch.ones_like(negative_weights),
                )
            ):
                raise ValueError(
                    "batch-local-infonce does not consume BCE negative weights"
                )
            contrastive = _batch_local_predicate_infonce(
                training_outputs.predicate_query[positive],
                training_outputs.predicate_query_raw[positive],
                training_outputs.predicate_bank,
                positive_targets,
                supervision_mask=supervision_mask,
                positive_weights=pos_weight,
                negative_candidate_mask=(
                    predicate_contrastive_negative_mask
                ),
                semantic_positive_weights=(
                    predicate_soft_positive_weights
                ),
                hard_negative_count=(
                    config.predicate_contrastive_hard_negative_count
                ),
                temperature=config.predicate_contrastive_temperature,
            )
            predicate_loss = contrastive["loss"]
            predicate_rows_skipped = contrastive["rows_skipped"]
            predicate_contrast_set_size = contrastive[
                "contrast_set_size"
            ]
            predicate_positive_cosine = contrastive[
                "positive_cosine"
            ]
            predicate_hard_negative_margin = contrastive[
                "hard_negative_margin"
            ]
            predicate_query_raw_norm = contrastive[
                "query_raw_norm"
            ]
            predicate_unobserved_column_fraction = contrastive[
                "unobserved_column_fraction"
            ]
            predicate_hard_negative_count = contrastive[
                "hard_negative_count"
            ]
            predicate_soft_positive_count = contrastive[
                "soft_positive_count"
            ]
        else:
            if explicit_holdout_row_policy == "skip-holdout-only":
                assert holdout_mask is not None
                assert supervision_mask is not None
                positive_labels = positive_targets > 0.5
                has_holdout_positive = (
                    positive_labels & holdout_mask.view(1, -1)
                ).any(dim=1)
                has_supervised_positive = (
                    positive_labels & supervision_mask.view(1, -1)
                ).any(dim=1)
                skip_rows = (
                    has_holdout_positive & ~has_supervised_positive
                )
                predicate_rows_skipped = skip_rows.sum().to(
                    dtype=pred_logits.dtype
                )
                keep_rows = ~skip_rows
                positive_logits = positive_logits[keep_rows]
                positive_targets = positive_targets[keep_rows]

            if positive_logits.shape[0] > 0:
                raw_predicate_loss = F.binary_cross_entropy_with_logits(
                    positive_logits,
                    positive_targets,
                    pos_weight=pos_weight,
                    reduction="none",
                )
                element_weights = torch.ones_like(raw_predicate_loss)
                if negative_weights is not None:
                    element_weights = torch.where(
                        positive_targets > 0.5,
                        element_weights,
                        negative_weights.view(1, -1).expand_as(
                            raw_predicate_loss
                        ),
                    )
                if supervision_mask is not None:
                    raw_predicate_loss = raw_predicate_loss[
                        :, supervision_mask
                    ]
                    element_weights = element_weights[
                        :, supervision_mask
                    ]
                weight_sum = element_weights.sum()
                if weight_sum <= 0:
                    raise ValueError(
                        "predicate supervision weights must keep at least one term"
                    )
                predicate_loss = (
                    raw_predicate_loss * element_weights
                ).sum() / weight_sum
            else:
                predicate_loss = pred_logits.sum() * 0.0
    else:
        predicate_loss = pred_logits.sum() * 0.0

    if training_outputs is not None:
        sampler_logits = training_outputs.sampler_logits
        sampler_valid = training_outputs.sampler_valid.to(torch.bool)
        dense_pair_targets = pair_targets.reshape(pair_targets.shape[0], -1)
        if sampler_logits.shape != sampler_valid.shape:
            raise ValueError("sampler logits and validity mask shapes differ")
        if sampler_logits.shape != dense_pair_targets.shape:
            raise ValueError(
                "sampler output shape does not match dense pair targets"
            )
        sampler_loss = _weighted_pair_bce(
            sampler_logits,
            dense_pair_targets,
            sampler_valid,
            negative_pair_weight=config.negative_pair_weight,
        )
    else:
        sampler_loss = pair_logits.sum() * 0.0

    total = (
        sampler_loss * config.sampler_loss_weight
        + pair_loss * config.pair_loss_weight
        + predicate_loss * config.predicate_loss_weight
    )
    return {
        "loss": total,
        "sampler_loss": sampler_loss,
        "pair_loss": pair_loss,
        "predicate_loss": predicate_loss,
        "predicate_rows": predicate_rows,
        "predicate_rows_skipped": predicate_rows_skipped,
        "predicate_contrast_set_size": predicate_contrast_set_size,
        "predicate_positive_cosine": predicate_positive_cosine,
        "predicate_hard_negative_margin": predicate_hard_negative_margin,
        "predicate_query_raw_norm": predicate_query_raw_norm,
        "predicate_unobserved_column_fraction": (
            predicate_unobserved_column_fraction
        ),
        "predicate_hard_negative_count": predicate_hard_negative_count,
        "predicate_soft_positive_count": predicate_soft_positive_count,
    }
