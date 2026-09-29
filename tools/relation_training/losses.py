from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class RelationLossConfig:
    pair_loss_weight: float = 1.0
    predicate_loss_weight: float = 1.0
    negative_pair_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.pair_loss_weight < 0.0 or self.predicate_loss_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if not 0.0 <= self.negative_pair_weight <= 1.0:
            raise ValueError("negative_pair_weight must be within [0,1]")


def supervised_relation_loss(
    outputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    pair_targets: Tensor,
    predicate_targets: Tensor,
    config: RelationLossConfig = RelationLossConfig(),
) -> dict[str, Tensor]:
    """Supervised baseline loss.

    pair_targets is [B,N,N] with 1 for annotated related pairs and 0 otherwise.
    predicate_targets is [B,N,N] with a predicate index for positive pairs and
    -1 where predicate identity is unavailable.

    This is intentionally not a PU objective. Unannotated valid pairs are
    down-weighted negatives through negative_pair_weight.
    """
    pred_logits, pair_logits, sub_idx, obj_idx, valid_mask = outputs
    if pair_targets.ndim != 3 or predicate_targets.shape != pair_targets.shape:
        raise ValueError("relation targets must both be [B,N,N]")
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
    if valid.any():
        raw_pair_loss = F.binary_cross_entropy_with_logits(
            pair_logits[valid],
            selected_pair_targets[valid],
            reduction="none",
        )
        pair_weights = torch.where(
            selected_pair_targets[valid] > 0.5,
            torch.ones_like(raw_pair_loss),
            torch.full_like(raw_pair_loss, config.negative_pair_weight),
        )
        pair_loss = (raw_pair_loss * pair_weights).sum() / pair_weights.sum().clamp_min(1.0)
    else:
        pair_loss = pair_logits.sum() * 0.0

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

    total = (
        pair_loss * config.pair_loss_weight
        + predicate_loss * config.predicate_loss_weight
    )
    return {
        "loss": total,
        "pair_loss": pair_loss,
        "predicate_loss": predicate_loss,
    }
