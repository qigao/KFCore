from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model import RelationTrainingOutputs


SOURCE_ALLOW_SCHEMA = "kfcore.predicate-source-allow/1"


@dataclass
class PredicateOntology:
    predicates: tuple[str, ...]
    pos_w: Tensor
    neg_lw: Tensor
    sym: Tensor
    inverse_mask: Tensor

    def __post_init__(self) -> None:
        count = len(self.predicates)
        if count <= 0:
            raise ValueError("predicate ontology must not be empty")
        if self.pos_w.shape != (count, count):
            raise ValueError("ontology pos_w must be [V,V]")
        if self.neg_lw.shape != (count, count):
            raise ValueError("ontology neg_lw must be [V,V]")
        if self.sym.shape != (count,):
            raise ValueError("ontology sym must be [V]")
        if self.inverse_mask.shape != (count, count):
            raise ValueError("ontology inverse_mask must be [V,V]")
        if self.inverse_mask.dtype != torch.bool:
            raise ValueError("ontology inverse_mask must be bool")
        for name, value in (
            ("pos_w", self.pos_w),
            ("neg_lw", self.neg_lw),
            ("sym", self.sym),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"ontology {name} must be finite")
        if (self.pos_w < 0).any():
            raise ValueError("ontology pos_w must be non-negative")
        if (self.sym < 0).any() or (self.sym > 1).any():
            raise ValueError("ontology sym must be within [0,1]")

    @property
    def pos_mask(self) -> Tensor:
        return self.pos_w.float() > 0

    @classmethod
    def from_soft_supervision(
        cls,
        meta_path: str | Path,
        npz_path: str | Path,
    ) -> "PredicateOntology":
        meta = json.loads(
            Path(meta_path).read_text(encoding="utf-8")
        )
        if (
            not isinstance(meta, dict)
            or not isinstance(meta.get("predicates"), list)
        ):
            raise ValueError(
                "ontology metadata requires a predicates array"
            )
        meta_names = tuple(str(value) for value in meta["predicates"])

        with np.load(npz_path, allow_pickle=False) as data:
            names = tuple(str(value) for value in data["predicates"])
            if names != meta_names:
                raise ValueError(
                    "soft-supervision vocabulary does not match metadata"
                )
            count = len(names)

            pos_w = np.zeros(
                (count, count),
                dtype=np.float16,
            )
            pos_w[
                data["pos_i"],
                data["pos_j"],
            ] = data["pos_w"]
            np.fill_diagonal(pos_w, 1.0)

            neg_lw = np.zeros(
                (count, count),
                dtype=np.float16,
            )
            neg_lw[
                data["neg_i"],
                data["neg_j"],
            ] = np.maximum(
                data["neg_lw"].astype(np.float32),
                math.log(1.0e-6),
            ).astype(np.float16)
            np.fill_diagonal(neg_lw, 0.0)

            inverse_weight = np.zeros(
                (count, count),
                dtype=np.float32,
            )
            np.maximum.at(
                inverse_weight,
                (data["inv_i"], data["inv_j"]),
                data["inv_w"].astype(np.float32),
            )
            inverse_weight = np.maximum(
                inverse_weight,
                inverse_weight.T,
            )
            neg_lw[inverse_weight > 0] = 0.0

            sym = data["sym"].astype(np.float32)

        return cls(
            predicates=names,
            pos_w=torch.from_numpy(pos_w),
            neg_lw=torch.from_numpy(neg_lw),
            sym=torch.from_numpy(sym),
            inverse_mask=torch.from_numpy(
                inverse_weight > 0.5
            ),
        )

    def stats(self) -> dict[str, float | int]:
        negative_downweight = (
            1.0 - torch.exp(self.neg_lw.float())
        )
        return {
            "vocabulary_size": len(self.predicates),
            "avg_positive_members": float(
                self.pos_mask.float().sum(1).mean()
            ),
            "avg_positive_weight_mass": float(
                self.pos_w.float().sum(1).mean()
            ),
            "avg_negative_downweight_mass": float(
                negative_downweight.sum(1).mean()
            ),
            "inverse_pairs": int(
                self.inverse_mask.sum().item()
            ) // 2,
            "symmetric_over_half": int(
                (self.sym > 0.5).sum().item()
            ),
        }


def load_source_column_allow(
    path: str | Path,
    predicates: Sequence[str],
) -> tuple[tuple[str, ...], Tensor]:
    payload = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SOURCE_ALLOW_SCHEMA
    ):
        raise ValueError(
            "unsupported predicate source-allow schema"
        )
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(
            "predicate source-allow requires a non-empty sources array"
        )

    predicate_index = {
        name: index
        for index, name in enumerate(predicates)
    }
    names: list[str] = []
    rows: list[Tensor] = []
    seen: set[str] = set()

    for source in sources:
        if not isinstance(source, dict):
            raise ValueError(
                "predicate source entry must be an object"
            )
        name = source.get("name")
        allowed = source.get("predicates")
        if (
            not isinstance(name, str)
            or not name
            or name in seen
        ):
            raise ValueError(
                "source names must be unique non-empty strings"
            )
        if not isinstance(allowed, list) or not allowed:
            raise ValueError(
                "every source must allow at least one predicate"
            )
        row = torch.zeros(
            len(predicates),
            dtype=torch.bool,
        )
        local_seen: set[str] = set()
        for predicate in allowed:
            if (
                not isinstance(predicate, str)
                or predicate not in predicate_index
                or predicate in local_seen
            ):
                raise ValueError(
                    f"invalid/duplicate predicate for source {name}"
                )
            local_seen.add(predicate)
            row[predicate_index[predicate]] = True

        seen.add(name)
        names.append(name)
        rows.append(row)

    return tuple(names), torch.stack(rows, dim=0)


@dataclass(frozen=True)
class ApacheObjectiveConfig:
    infonce_temp: float = 0.07
    n_neg: int = 512
    hard_frac: float = 0.5
    lambda_obj: float = 0.10
    lambda_swap: float = 0.50
    lambda_sigmoid: float = 0.25
    lambda_bg: float = 0.05
    lambda_geo: float = 1.0
    lambda_rel: float = 1.0
    bg_topk: int = 5
    swap_margin: float = 0.05
    pair_negative_floor: float = 0.30

    def __post_init__(self) -> None:
        finite_nonnegative = (
            "lambda_obj",
            "lambda_swap",
            "lambda_sigmoid",
            "lambda_bg",
            "lambda_geo",
            "lambda_rel",
            "swap_margin",
            "pair_negative_floor",
        )
        for name in finite_nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative"
                )
        if (
            not math.isfinite(self.infonce_temp)
            or self.infonce_temp <= 0.0
        ):
            raise ValueError(
                "infonce_temp must be finite and positive"
            )
        if (
            isinstance(self.n_neg, bool)
            or not isinstance(self.n_neg, int)
            or self.n_neg < 0
        ):
            raise ValueError("n_neg must be a non-negative integer")
        if (
            not math.isfinite(self.hard_frac)
            or self.hard_frac < 0.0
            or self.hard_frac > 1.0
        ):
            raise ValueError("hard_frac must be within [0,1]")
        if (
            isinstance(self.bg_topk, bool)
            or not isinstance(self.bg_topk, int)
            or self.bg_topk <= 0
        ):
            raise ValueError("bg_topk must be a positive integer")
        if self.pair_negative_floor > 1.0:
            raise ValueError(
                "pair_negative_floor must be within [0,1]"
            )


class BatchLocalInfoNCE(nn.Module):
    """Apache-reference ontology-aware batch-local relation InfoNCE."""

    def __init__(
        self,
        ontology: PredicateOntology,
        *,
        temp: float = 0.07,
        n_neg: int = 512,
        hard_frac: float = 0.5,
    ) -> None:
        super().__init__()
        self.temp = float(temp)
        self.n_neg = int(n_neg)
        self.hard_frac = float(hard_frac)
        self.pos_mask = ontology.pos_mask
        self.inverse_mask = ontology.inverse_mask
        self.pos_w = ontology.pos_w
        self.neg_lw = ontology.neg_lw

    def _ensure_device(self, device: torch.device) -> None:
        if self.pos_mask.device != device:
            self.pos_mask = self.pos_mask.to(device)
            self.inverse_mask = self.inverse_mask.to(device)
            self.pos_w = self.pos_w.to(device)
            self.neg_lw = self.neg_lw.to(device)

    @torch.no_grad()
    def build_set(
        self,
        labels: Tensor,
        W: Tensor,
    ) -> Tensor:
        if labels.numel() == 0:
            return torch.empty(
                0,
                dtype=torch.int64,
                device=W.device,
            )
        self._ensure_device(W.device)
        vocabulary_size = int(W.shape[0])
        classes = labels.unique()
        parts = [
            classes,
            self.inverse_mask[classes]
            .any(0)
            .nonzero()
            .flatten(),
        ]

        n_hard = int(self.n_neg * self.hard_frac)
        if n_hard > 0:
            similarity = W[classes] @ W.T
            score = (
                similarity.amax(0)
                + self.neg_lw[classes].amin(0).float()
            )
            score = score.masked_fill(
                self.pos_mask[classes].any(0),
                -2.0,
            )
            parts.append(
                score.topk(
                    min(n_hard, vocabulary_size)
                ).indices
            )

        n_random = self.n_neg - n_hard
        if n_random > 0:
            parts.append(
                torch.randint(
                    0,
                    vocabulary_size,
                    (n_random,),
                    device=W.device,
                )
            )
        return torch.cat(parts).unique()

    def compute(
        self,
        feats: Tensor,
        labels: Tensor,
        W: Tensor,
        *,
        feats_spa: Tensor,
        alpha: Tensor,
        weights: Tensor | None = None,
        col_allow: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if feats.numel() == 0:
            zero = feats.new_zeros(())
            empty = torch.empty(
                0,
                dtype=torch.int64,
                device=feats.device,
            )
            return zero, empty
        if labels.ndim != 2 or labels.shape[0] != feats.shape[0]:
            raise ValueError(
                "InfoNCE labels must be [M,V] matching features"
            )
        if (
            W.ndim != 2
            or W.shape[0] != labels.shape[1]
            or W.shape[1] != feats.shape[1]
        ):
            raise ValueError(
                "InfoNCE W must be [V,D] matching labels/features"
            )
        if feats_spa.shape != feats.shape:
            raise ValueError(
                "InfoNCE spatial features must match semantic features"
            )
        if alpha.shape != (W.shape[0],):
            raise ValueError("InfoNCE alpha must be [V]")
        if col_allow is not None and col_allow.shape != labels.shape:
            raise ValueError(
                "InfoNCE source column mask must be [M,V]"
            )

        self._ensure_device(feats.device)
        flat_labels = labels.nonzero(as_tuple=True)[1]
        contrast = self.build_set(flat_labels, W)
        if contrast.numel() == 0:
            return feats.sum() * 0.0, contrast

        allow_contrast = (
            col_allow[:, contrast]
            if col_allow is not None
            else None
        )

        semantic = F.normalize(feats, dim=-1)
        spatial = F.normalize(feats_spa, dim=-1)
        cos = semantic @ W[contrast].T
        alpha_s = alpha[contrast]
        cos = (
            (1.0 - alpha_s) * cos
            + alpha_s * (spatial @ W[contrast].T)
        )
        logits = cos / self.temp
        neg_inf = torch.finfo(logits.dtype).min

        row_idx, predicate_idx = labels.nonzero(
            as_tuple=True
        )
        hot = labels.float()
        pos_any = (
            hot @ self.pos_mask[:, contrast].float()
        ) > 0
        inverse_any = (
            hot @ self.inverse_mask[:, contrast].float()
        ) > 0
        log_weight = (
            hot @ self.neg_lw[:, contrast].float()
        ).masked_fill(
            pos_any | inverse_any,
            0.0,
        )
        denominator_logits = logits + log_weight

        ignored = torch.zeros_like(pos_any)
        if allow_contrast is not None:
            ignored = (
                ignored
                | (~allow_contrast & ~pos_any)
            )
        denominator = denominator_logits.masked_fill(
            ignored,
            neg_inf,
        ).logsumexp(-1)

        positive_weights = self.pos_w[
            predicate_idx
        ][:, contrast].to(logits.dtype)
        per = denominator[row_idx].unsqueeze(-1) - logits[
            row_idx
        ]
        loss = (
            (per * positive_weights).sum(-1)
            / positive_weights.sum(-1).clamp_min(1.0e-6)
        )

        label_count = labels.sum(-1).clamp_min(1)
        row_weights = (
            weights
            if weights is not None
            else torch.ones_like(
                label_count,
                dtype=loss.dtype,
            )
        )
        weighted = (
            row_weights / label_count
        )[row_idx]
        result = (
            (loss * weighted).sum()
            / weighted.sum().clamp_min(1.0e-6)
        )
        return result, contrast

    def forward(
        self,
        feats: Tensor,
        labels: Tensor,
        W: Tensor,
        *,
        feats_spa: Tensor,
        alpha: Tensor,
        weights: Tensor | None = None,
        col_allow: Tensor | None = None,
    ) -> Tensor:
        loss, _ = self.compute(
            feats,
            labels,
            W,
            feats_spa=feats_spa,
            alpha=alpha,
            weights=weights,
            col_allow=col_allow,
        )
        return loss


def _flat_pair_lookup(
    sub_idx: Tensor,
    obj_idx: Tensor,
    valid_mask: Tensor,
):
    batch, _ = sub_idx.shape
    base = (
        torch.arange(
            batch,
            device=sub_idx.device,
            dtype=torch.int64,
        )
        << 20
    ).unsqueeze(1)
    keys = (
        base
        + sub_idx.to(torch.int64) * 1024
        + obj_idx.to(torch.int64)
    ).masked_fill(
        ~valid_mask.to(torch.bool),
        -1,
    ).reshape(-1)
    order = keys.argsort()
    sorted_keys = keys[order]

    def lookup(query: Tensor) -> Tensor:
        if sorted_keys.numel() == 0:
            return torch.full_like(query, -1)
        position = torch.searchsorted(
            sorted_keys,
            query,
        ).clamp(max=sorted_keys.numel() - 1)
        hit = sorted_keys[position] == query
        return torch.where(
            hit,
            order[position],
            torch.full_like(query, -1),
        )

    return lookup


def _selected_targets(
    runtime: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    predicate_targets: Tensor,
) -> tuple[Tensor, Tensor]:
    _, _, sub_idx, obj_idx, valid_mask = runtime
    if predicate_targets.ndim != 4:
        raise ValueError(
            "Apache objective predicate targets must be [B,N,N,V]"
        )
    if predicate_targets.shape[0] != sub_idx.shape[0]:
        raise ValueError(
            "predicate target batch does not match runtime"
        )
    batch = torch.arange(
        sub_idx.shape[0],
        device=sub_idx.device,
        dtype=torch.int64,
    ).unsqueeze(1)
    hot = predicate_targets[
        batch,
        sub_idx.to(torch.int64),
        obj_idx.to(torch.int64),
    ] > 0.5
    has_gt = hot.any(-1) & valid_mask.to(torch.bool)
    return hot, has_gt


def swap_direction_hinge_dense(
    q_sem: Tensor,
    q_spa: Tensor,
    alpha: Tensor,
    W: Tensor,
    sub_idx: Tensor,
    obj_idx: Tensor,
    valid_mask: Tensor,
    predicate_targets: Tensor,
    sym: Tensor,
    *,
    margin: float = 0.05,
) -> Tensor:
    positive = (predicate_targets > 0.5).nonzero(
        as_tuple=True
    )
    if not positive or positive[0].numel() == 0:
        return W.new_zeros(())
    batch_ids, subjects, objects, predicates = positive
    lookup = _flat_pair_lookup(
        sub_idx,
        obj_idx,
        valid_mask,
    )
    forward_slot = lookup(
        (batch_ids << 20)
        + subjects * 1024
        + objects
    )
    backward_slot = lookup(
        (batch_ids << 20)
        + objects * 1024
        + subjects
    )
    keep = (
        (forward_slot >= 0)
        & (backward_slot >= 0)
    )
    reciprocal_annotation = (
        predicate_targets[
            batch_ids,
            objects,
            subjects,
            predicates,
        ] > 0.5
    )
    keep = keep & ~reciprocal_annotation

    forward_slot = forward_slot.clamp_min(0)
    backward_slot = backward_slot.clamp_min(0)
    weight = (
        keep.to(W.dtype)
        * (1.0 - sym[predicates].to(W.dtype))
    )

    predicate_direction = W[predicates]
    alpha_direction = alpha[
        predicates
    ].unsqueeze(-1)
    dim = q_sem.shape[-1]
    flat_sem = q_sem.reshape(-1, dim)
    flat_spa = q_spa.reshape(-1, dim)

    def mixed_cos(slots: Tensor) -> Tensor:
        semantic = (
            F.normalize(
                flat_sem[slots],
                dim=-1,
            )
            * predicate_direction
        ).sum(-1, keepdim=True)
        spatial = (
            F.normalize(
                flat_spa[slots],
                dim=-1,
            )
            * predicate_direction
        ).sum(-1, keepdim=True)
        return (
            (1.0 - alpha_direction) * semantic
            + alpha_direction * spatial
        ).squeeze(-1)

    hinge = F.relu(
        margin
        + mixed_cos(backward_slot)
        - mixed_cos(forward_slot)
    )
    return (
        (hinge * weight).sum()
        / weight.sum().clamp_min(1.0)
    )


class ApacheReferenceObjective(nn.Module):
    """Apache RelateAnything reference objective over KFCore training tensors."""

    def __init__(
        self,
        ontology: PredicateOntology,
        *,
        config: ApacheObjectiveConfig = ApacheObjectiveConfig(),
        source_column_allow: Tensor | None = None,
        object_text_bank: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ontology = ontology
        self.config = config
        self.infonce = BatchLocalInfoNCE(
            ontology,
            temp=config.infonce_temp,
            n_neg=config.n_neg,
            hard_frac=config.hard_frac,
        )
        if source_column_allow is not None:
            if (
                source_column_allow.ndim != 2
                or source_column_allow.shape[1]
                != len(ontology.predicates)
                or source_column_allow.dtype != torch.bool
            ):
                raise ValueError(
                    "source_column_allow must be bool [S,V]"
                )
            if not source_column_allow.any(dim=1).all():
                raise ValueError(
                    "every source must allow at least one predicate"
                )
            self.source_column_allow = source_column_allow
        else:
            self.source_column_allow = None

        if object_text_bank is not None:
            if (
                object_text_bank.ndim != 2
                or object_text_bank.shape[0] <= 0
                or object_text_bank.shape[1] <= 0
                or not torch.isfinite(object_text_bank).all()
            ):
                raise ValueError(
                    "object_text_bank must be finite [O,D]"
                )
            self.object_text_bank = F.normalize(
                object_text_bank.float(),
                dim=-1,
            )
        else:
            self.object_text_bank = None

    def _source_allow(
        self,
        source_ids: Tensor,
        predicate_targets: Tensor,
    ) -> Tensor | None:
        if self.source_column_allow is None:
            return None
        if source_ids.ndim != 1:
            raise ValueError("source_ids must be [B]")
        allow = self.source_column_allow.to(
            predicate_targets.device
        )
        source_ids = source_ids.to(torch.int64)
        if (
            (source_ids < 0).any()
            or (source_ids >= allow.shape[0]).any()
        ):
            raise ValueError(
                "training source_id exceeds source-column table"
            )
        image_allow = allow[source_ids]
        observed = predicate_targets > 0.5
        observed_by_image = observed.any(dim=1).any(dim=1)
        if (observed_by_image & ~image_allow).any():
            raise ValueError(
                "source column mask hides an observed positive predicate"
            )
        return image_allow

    def _object_text_loss(
        self,
        outputs: RelationTrainingOutputs,
        object_label_indices: Tensor | None,
    ) -> Tensor:
        if self.config.lambda_obj <= 0.0:
            return outputs.runtime[0].new_zeros(())
        if self.object_text_bank is None:
            raise ValueError(
                "Apache object-text loss requires object_text_bank"
            )
        if (
            outputs.object_subject_query is None
            or outputs.object_object_query is None
        ):
            raise ValueError(
                "Apache object-text loss requires object query tensors"
            )
        if object_label_indices is None:
            raise ValueError(
                "Apache object-text loss requires object labels"
            )
        if object_label_indices.shape != (
            outputs.object_subject_query.shape[0],
            outputs.object_subject_query.shape[1],
        ):
            raise ValueError(
                "object label indices must match [B,N] object queries"
            )

        labels = object_label_indices.to(
            device=outputs.object_subject_query.device,
            dtype=torch.int64,
        )
        keep = labels >= 0
        if not keep.any():
            return outputs.runtime[0].new_zeros(())

        bank = self.object_text_bank.to(
            outputs.object_subject_query.device
        )
        if bank.shape[1] != outputs.object_subject_query.shape[-1]:
            raise ValueError(
                "object text bank width does not match query width"
            )
        if (labels[keep] >= bank.shape[0]).any():
            raise ValueError(
                "object label index exceeds object vocabulary"
            )

        loss = outputs.runtime[0].new_zeros(())
        for query in (
            outputs.object_subject_query,
            outputs.object_object_query,
        ):
            assert query is not None
            normalized = F.normalize(
                query[keep],
                dim=-1,
            )
            loss = loss + F.cross_entropy(
                normalized @ bank.T
                / self.config.infonce_temp,
                labels[keep],
            )
        return loss * 0.5

    def forward(
        self,
        outputs: RelationTrainingOutputs,
        predicate_targets: Tensor,
        *,
        source_ids: Tensor,
        object_label_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if outputs.predicate_spatial_query is None:
            raise ValueError(
                "Apache objective requires spatial predicate queries"
            )
        if outputs.predicate_alpha is None:
            raise ValueError(
                "Apache objective requires text-conditioned alpha"
            )
        if (
            outputs.sampler_geo_loss is None
            or outputs.sampler_relatedness_loss is None
        ):
            raise ValueError(
                "Apache objective requires Apache sampler losses"
            )

        (
            pred_logits,
            _pair_logits,
            sub_idx,
            obj_idx,
            valid_mask,
        ) = outputs.runtime
        W = outputs.predicate_bank
        q_sem = outputs.predicate_query_raw
        q_spa = outputs.predicate_spatial_query
        alpha = outputs.predicate_alpha
        if W.shape[0] != len(self.ontology.predicates):
            raise ValueError(
                "predicate bank width does not match ontology"
            )
        if pred_logits.shape[-1] != W.shape[0]:
            raise ValueError(
                "runtime predicate width does not match ontology"
            )

        hot_all, has_gt = _selected_targets(
            outputs.runtime,
            predicate_targets,
        )

        image_allow = self._source_allow(
            source_ids,
            predicate_targets,
        )
        slot_allow = None
        if image_allow is not None:
            batch_index = torch.arange(
                pred_logits.shape[0],
                device=pred_logits.device,
                dtype=torch.int64,
            ).unsqueeze(1).expand_as(has_gt)
            slot_allow = image_allow[
                batch_index[has_gt]
            ]

        hot = hot_all[has_gt]
        nce_loss, contrast_set = self.infonce.compute(
            q_sem[has_gt],
            hot,
            W,
            feats_spa=q_spa[has_gt],
            alpha=alpha,
            col_allow=slot_allow,
        )

        object_loss = self._object_text_loss(
            outputs,
            object_label_indices,
        )

        sym = self.ontology.sym.to(W.device)
        swap_loss = pred_logits.new_zeros(())
        if self.config.lambda_swap > 0.0:
            swap_loss = swap_direction_hinge_dense(
                q_sem,
                q_spa,
                alpha,
                W,
                sub_idx,
                obj_idx,
                valid_mask,
                predicate_targets,
                sym,
                margin=self.config.swap_margin,
            )

        sigmoid_loss = pred_logits.new_zeros(())
        if self.config.lambda_sigmoid > 0.0 and has_gt.any():
            logits_gt = pred_logits[has_gt]
            target = hot.to(logits_gt.dtype)
            allow_float = (
                slot_allow.to(target.dtype)
                if slot_allow is not None
                else torch.ones_like(target)
            )
            positive_mass = target.sum(
                -1,
                keepdim=True,
            )
            negative_count = (
                (target <= 0)
                & (allow_float > 0)
            ).sum(
                -1,
                keepdim=True,
            ).clamp_min(1)
            weight = torch.where(
                target > 0,
                torch.ones_like(target),
                positive_mass
                / negative_count
                * allow_float,
            )
            raw = F.binary_cross_entropy_with_logits(
                logits_gt,
                target.clamp(0, 1),
                weight=weight,
                reduction="none",
            )
            sigmoid_loss = raw.sum(-1).mean()

        background_loss = pred_logits.new_zeros(())
        background_slots = (
            valid_mask.to(torch.bool)
            & ~has_gt
        )
        if (
            self.config.lambda_bg > 0.0
            and background_slots.any()
        ):
            background_logits = pred_logits[
                background_slots
            ]
            if image_allow is not None:
                batch_index = torch.arange(
                    pred_logits.shape[0],
                    device=pred_logits.device,
                    dtype=torch.int64,
                ).unsqueeze(1).expand_as(background_slots)
                background_allow = image_allow[
                    batch_index[background_slots]
                ]
                background_logits = background_logits.masked_fill(
                    ~background_allow,
                    torch.finfo(
                        background_logits.dtype
                    ).min,
                )
            topk = min(
                self.config.bg_topk,
                background_logits.shape[-1],
            )
            per_slot = F.softplus(
                background_logits.topk(
                    topk,
                    dim=-1,
                ).values
            ).mean(-1)

            if outputs.sampler_pair_negative_weights is not None:
                slot_weight = outputs.sampler_pair_negative_weights[
                    background_slots
                ].to(per_slot.dtype)
            else:
                slot_weight = torch.full_like(
                    per_slot,
                    self.config.pair_negative_floor,
                )
            background_loss = (
                (per_slot * slot_weight).sum()
                / slot_weight.sum().clamp_min(1.0e-6)
            )

        geo_loss = outputs.sampler_geo_loss
        relatedness_loss = outputs.sampler_relatedness_loss

        config = self.config
        total = (
            nce_loss
            + config.lambda_obj * object_loss
            + config.lambda_swap * swap_loss
            + config.lambda_sigmoid * sigmoid_loss
            + config.lambda_bg * background_loss
            + config.lambda_geo * geo_loss
            + config.lambda_rel * relatedness_loss
        )

        exact_labels = hot.nonzero(as_tuple=True)[1]
        synonym_mass = pred_logits.new_zeros(())
        inverse_count = pred_logits.new_zeros(())
        if exact_labels.numel() > 0:
            pos_w = self.ontology.pos_w.to(
                pred_logits.device,
                dtype=pred_logits.dtype,
            )
            synonym_mass = (
                pos_w[exact_labels].sum(-1) - 1.0
            ).mean()
            inverse_mask = self.ontology.inverse_mask.to(
                pred_logits.device
            )
            inverse_count = inverse_mask[
                exact_labels
            ].any(0).sum().to(pred_logits.dtype)

        source_masked_fraction = pred_logits.new_zeros(())
        if slot_allow is not None and slot_allow.numel() > 0:
            source_masked_fraction = (
                1.0
                - slot_allow.to(
                    pred_logits.dtype
                ).mean()
            )

        return {
            "loss": total,
            "loss_nce": nce_loss,
            "loss_obj": object_loss,
            "loss_swap": swap_loss,
            "loss_sigmoid": sigmoid_loss,
            "loss_background": background_loss,
            "loss_geo": geo_loss,
            "loss_relatedness": relatedness_loss,
            "contrast_set_size": pred_logits.new_tensor(
                float(contrast_set.numel())
            ),
            "synonym_positive_mass": synonym_mass,
            "inverse_negative_count": inverse_count,
            "source_masked_column_fraction": source_masked_fraction,
            "labelled_slots": has_gt.sum().to(
                pred_logits.dtype
            ),
            "background_slots": background_slots.sum().to(
                pred_logits.dtype
            ),
        }
