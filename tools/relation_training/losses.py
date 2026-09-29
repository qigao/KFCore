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

    def __post_init__(self) -> None:
        if (
            self.sampler_loss_weight < 0.0
            or self.pair_loss_weight < 0.0
            or self.predicate_loss_weight < 0.0
        ):
            raise ValueError("loss weights must be non-negative")
        if not 0.0 <= self.negative_pair_weight <= 1.0:
            raise ValueError("negative_pair_weight must be within [0,1]")


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


def supervised_relation_loss(
    outputs: RelationTrainingOutputs
    | tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    pair_targets: Tensor,
    predicate_targets: Tensor,
    config: RelationLossConfig = RelationLossConfig(),
    predicate_positive_weights: Tensor | None = None,
) -> dict[str, Tensor]:
    """Exhaustive supervised multi-label relation baseline.

    pair_targets is [B,N,N] with 1 for annotated related pairs and 0 otherwise.
    predicate_targets is [B,N,N,V] multi-hot over the complete benchmark
    predicate vocabulary for each ordered pair.

    Predicate BCE is evaluated only on positive related pairs. Zeros in their
    multi-hot predicate vectors are supervised negatives, so this objective
    assumes exhaustive pair-level predicate labels. Incomplete/open-world labels
    require a separate PU-aware objective.

    The dense sampler loss is training-only. Runtime/export keeps the existing
    five-output ABI. Unannotated valid pairs are down-weighted negatives through
    negative_pair_weight.
    """
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

    if positive.any():
        predicate_loss = F.binary_cross_entropy_with_logits(
            pred_logits[positive],
            selected_predicate_targets[positive],
            pos_weight=pos_weight,
        )
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
    }
