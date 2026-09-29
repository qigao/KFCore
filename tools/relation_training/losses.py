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


def supervised_relation_loss(
    outputs: RelationTrainingOutputs
    | tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    pair_targets: Tensor,
    predicate_targets: Tensor,
    config: RelationLossConfig = RelationLossConfig(),
) -> dict[str, Tensor]:
    """Supervised relation baseline with an explicit sampler objective.

    pair_targets is [B,N,N] with 1 for annotated related pairs and 0 otherwise.
    predicate_targets is [B,N,N] with a predicate index for positive pairs and
    -1 where predicate identity is unavailable.

    The dense sampler loss is training-only. Runtime/export keeps the existing
    five-output ABI. Unannotated valid pairs are down-weighted negatives through
    negative_pair_weight; this is intentionally not a PU objective.
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
    if pair_targets.ndim != 3 or predicate_targets.shape != pair_targets.shape:
        raise ValueError("relation targets must both be [B,N,N]")
    if pair_targets.shape[1] != pair_targets.shape[2]:
        raise ValueError("relation target object dimensions must be square")
    if pair_targets.shape[0] != pred_logits.shape[0]:
        raise ValueError("target and output batch sizes must match")
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
        batch, sub_idx.to(torch.int64), obj_idx.to(torch.int64)
    ].to(torch.int64)

    valid = valid_mask.to(torch.bool)
    pair_loss = _weighted_pair_bce(
        pair_logits,
        selected_pair_targets,
        valid,
        negative_pair_weight=config.negative_pair_weight,
    )

    positive = (
        valid
        & (selected_pair_targets > 0.5)
        & (selected_predicate_targets >= 0)
    )
    if positive.any():
        predicate_loss = F.cross_entropy(
            pred_logits[positive],
            selected_predicate_targets[positive],
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
