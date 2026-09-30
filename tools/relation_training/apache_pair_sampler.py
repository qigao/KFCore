from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from apache_pair_evidence import RelGeomEncoder


@dataclass(frozen=True)
class ApachePairSamplerOutput:
    sub_idx: Tensor
    obj_idx: Tensor
    valid_mask: Tensor
    pair_logits: Tensor
    geo_logits: Tensor
    pair_valid: Tensor
    geo_loss: Tensor
    relatedness_loss: Tensor
    pair_negative_weights: Tensor | None


class ApacheRelatednessPairSampler(nn.Module):
    """Apache-reference two-stage ordered-pair sampler.

    Stage 1 ranks all valid ordered pairs using the reference 19-D geometry.
    Stage 2 ranks the survivors with an asymmetric learned relatedness score
    over pooled subject/object visual features.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        geo_budget: int = 400,
        final_budget: int = 128,
        rel_dim: int = 256,
        negative_weight: float = 0.3,
        swap_include: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or rel_dim <= 0:
            raise ValueError("sampler feature dimensions must be positive")
        if geo_budget <= 0 or final_budget <= 0:
            raise ValueError("sampler budgets must be positive")
        if final_budget > geo_budget:
            raise ValueError("final_budget must not exceed geo_budget")
        if not 0.0 <= negative_weight <= 1.0:
            raise ValueError("negative_weight must be within [0,1]")

        self.geo_budget = int(geo_budget)
        self.final_budget = int(final_budget)
        self.rel_dim = int(rel_dim)
        self.negative_weight = float(negative_weight)
        self.swap_include = bool(swap_include)

        self.geo_scorer = nn.Sequential(
            nn.Linear(RelGeomEncoder.NUM_GEO, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.f_sub = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, rel_dim),
            nn.GELU(),
            nn.Linear(rel_dim, rel_dim),
        )
        self.f_obj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, rel_dim),
            nn.GELU(),
            nn.Linear(rel_dim, rel_dim),
        )

    @staticmethod
    def _pair_valid(
        box_counts: Tensor,
        count: int,
    ) -> Tensor:
        box_index = torch.arange(
            count,
            device=box_counts.device,
            dtype=torch.int64,
        )
        valid_box = (
            box_index.view(1, -1)
            < box_counts.to(torch.int64).view(-1, 1)
        )
        not_self = (
            box_index.view(1, -1, 1)
            != box_index.view(1, 1, -1)
        )
        return (
            valid_box.unsqueeze(2)
            & valid_box.unsqueeze(1)
            & not_self
        ).reshape(box_counts.shape[0], count * count)

    @staticmethod
    def _validate_pair_targets(
        pair_targets: Tensor | None,
        *,
        batch: int,
        count: int,
        device: torch.device,
    ) -> Tensor | None:
        if pair_targets is None:
            return None
        if pair_targets.shape != (batch, count, count):
            raise ValueError(
                "pair_targets must be [B,N,N] for Apache sampler training"
            )
        if not torch.isfinite(pair_targets).all():
            raise ValueError("pair_targets must be finite")
        return pair_targets.to(
            device=device,
            dtype=torch.float32,
        ) > 0.5

    @staticmethod
    def _pad(
        value: Tensor,
        target: int,
        *,
        pad_value: float | int | bool = 0,
    ) -> Tensor:
        if value.shape[1] >= target:
            return value
        return F.pad(
            value,
            (0, target - value.shape[1]),
            value=pad_value,
        )

    def forward(
        self,
        boxes: Tensor,
        object_features: Tensor,
        box_counts: Tensor,
        *,
        pair_targets: Tensor | None = None,
    ) -> ApachePairSamplerOutput:
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("boxes must be [B,N,4]")
        if (
            object_features.ndim != 3
            or object_features.shape[:2] != boxes.shape[:2]
        ):
            raise ValueError(
                "object_features must be [B,N,C] matching boxes"
            )
        if box_counts.shape != (boxes.shape[0],):
            raise ValueError("box_counts must be [B]")

        batch, count, _ = boxes.shape
        pair_valid = self._pair_valid(box_counts, count)
        gt_grid = self._validate_pair_targets(
            pair_targets,
            batch=batch,
            count=count,
            device=boxes.device,
        )
        training_contract = gt_grid is not None

        subject = boxes.unsqueeze(2).expand(
            batch, count, count, 4
        )
        object_ = boxes.unsqueeze(1).expand(
            batch, count, count, 4
        )
        geo_features = RelGeomEncoder.features(
            subject,
            object_,
        ).reshape(
            batch,
            count * count,
            RelGeomEncoder.NUM_GEO,
        )
        geo_logits = self.geo_scorer(
            geo_features
        ).squeeze(-1)

        force = (
            gt_grid.reshape(batch, count * count)
            if gt_grid is not None
            else torch.zeros_like(pair_valid)
        )
        if training_contract and self.swap_include:
            swapped = (
                gt_grid.transpose(1, 2)
                .reshape(batch, count * count)
            )
            force = force | (swapped & pair_valid)

        negative_sentinel = (
            torch.finfo(geo_logits.dtype).min / 2
        )
        stage1_scores = geo_logits.masked_fill(
            ~pair_valid,
            negative_sentinel,
        )
        if training_contract:
            stage1_scores = stage1_scores.masked_fill(
                force & pair_valid,
                float("inf"),
            )

        stage1_count = min(
            self.geo_budget,
            count * count,
        )
        _, stage1 = torch.topk(
            stage1_scores,
            k=stage1_count,
            dim=1,
            largest=True,
            sorted=True,
        )
        stage1_valid = torch.gather(
            pair_valid,
            1,
            stage1,
        )

        z_sub = self.f_sub(object_features)
        z_obj = self.f_obj(object_features)
        subject_index_1 = stage1 // count
        object_index_1 = stage1 % count
        subject_z = torch.gather(
            z_sub,
            1,
            subject_index_1.unsqueeze(-1).expand(
                -1,
                -1,
                self.rel_dim,
            ),
        )
        object_z = torch.gather(
            z_obj,
            1,
            object_index_1.unsqueeze(-1).expand(
                -1,
                -1,
                self.rel_dim,
            ),
        )
        stage1_relatedness = (
            subject_z * object_z
        ).sum(dim=-1) / math.sqrt(self.rel_dim)

        geo_loss = geo_logits.new_zeros(())
        relatedness_loss = geo_logits.new_zeros(())
        if training_contract:
            gt_flat = gt_grid.reshape(
                batch,
                count * count,
            ).to(geo_logits.dtype)
            geo_raw = F.binary_cross_entropy_with_logits(
                geo_logits,
                gt_flat,
                reduction="none",
            )
            valid_float = pair_valid.to(geo_raw.dtype)
            geo_loss = (
                geo_raw * valid_float
            ).sum() / valid_float.sum().clamp_min(1.0)

            gt_stage1 = torch.gather(
                gt_flat,
                1,
                stage1,
            )
            weights = torch.where(
                gt_stage1 > 0.5,
                torch.ones_like(stage1_relatedness),
                torch.full_like(
                    stage1_relatedness,
                    self.negative_weight,
                ),
            )
            probability = torch.sigmoid(
                stage1_relatedness
            )
            p_t = (
                gt_stage1 * probability
                + (1.0 - gt_stage1)
                * (1.0 - probability)
            )
            rel_raw = (
                F.binary_cross_entropy_with_logits(
                    stage1_relatedness,
                    gt_stage1,
                    reduction="none",
                )
                * (1.0 - p_t).pow(2.0)
            )
            alive = stage1_valid.to(rel_raw.dtype)
            relatedness_loss = (
                rel_raw * weights * alive
            ).sum() / alive.sum().clamp_min(1.0)

        stage2_scores = stage1_relatedness.masked_fill(
            ~stage1_valid,
            negative_sentinel,
        )
        if training_contract:
            force_stage1 = torch.gather(
                force & pair_valid,
                1,
                stage1,
            )
            stage2_scores = stage2_scores.masked_fill(
                force_stage1,
                float("inf"),
            )

        stage2_count = min(
            self.final_budget,
            stage1_count,
        )
        _, stage2 = torch.topk(
            stage2_scores,
            k=stage2_count,
            dim=1,
            largest=True,
            sorted=True,
        )
        flat_pair = torch.gather(
            stage1,
            1,
            stage2,
        )
        selected_valid = torch.gather(
            stage1_valid,
            1,
            stage2,
        )
        pair_logits = torch.gather(
            stage1_relatedness,
            1,
            stage2,
        )
        pair_negative_weights = None
        if training_contract:
            selected_gt = torch.gather(
                gt_grid.reshape(batch, count * count),
                1,
                flat_pair,
            )
            pair_negative_weights = torch.where(
                selected_gt,
                torch.ones_like(pair_logits),
                torch.full_like(
                    pair_logits,
                    self.negative_weight,
                ),
            )
        sub_idx = flat_pair // count
        obj_idx = flat_pair % count

        if stage2_count < self.final_budget:
            sub_idx = self._pad(
                sub_idx,
                self.final_budget,
                pad_value=0,
            )
            obj_idx = self._pad(
                obj_idx,
                self.final_budget,
                pad_value=0,
            )
            selected_valid = self._pad(
                selected_valid,
                self.final_budget,
                pad_value=False,
            )
            pair_logits = self._pad(
                pair_logits,
                self.final_budget,
                pad_value=float("-inf"),
            )
            if pair_negative_weights is not None:
                pair_negative_weights = self._pad(
                    pair_negative_weights,
                    self.final_budget,
                    pad_value=0.0,
                )

        return ApachePairSamplerOutput(
            sub_idx=sub_idx.to(torch.int64),
            obj_idx=obj_idx.to(torch.int64),
            valid_mask=selected_valid.to(torch.bool),
            pair_logits=pair_logits,
            geo_logits=geo_logits,
            pair_valid=pair_valid,
            geo_loss=geo_loss,
            relatedness_loss=relatedness_loss,
            pair_negative_weights=pair_negative_weights,
        )
