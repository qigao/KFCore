from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SPATIAL_FLAGS_SCHEMA = "kfcore.predicate-spatial-flags/1"


def load_predicate_spatial_flags(
    path: str | Path,
    predicates: Sequence[str],
) -> Tensor:
    """Load an exact-order spatial/semantic routing warm-start sidecar."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SPATIAL_FLAGS_SCHEMA
    ):
        raise ValueError("unsupported predicate spatial-flags schema")
    names = payload.get("predicates")
    flags = payload.get("is_spatial")
    if not isinstance(names, list) or not isinstance(flags, list):
        raise ValueError(
            "predicate spatial flags require predicates/is_spatial arrays"
        )
    if names != list(predicates):
        raise ValueError(
            "predicate spatial flags order does not match vocabulary"
        )
    if len(flags) != len(names):
        raise ValueError(
            "predicate spatial flags length does not match vocabulary"
        )
    if any(not isinstance(value, bool) for value in flags):
        raise ValueError("predicate spatial flags must be booleans")
    result = torch.tensor(flags, dtype=torch.bool)
    if not bool(result.any()) or bool(result.all()):
        raise ValueError(
            "predicate spatial flags require both spatial and semantic rows"
        )
    return result


class ApacheVocabHead(nn.Module):
    """Apache-reference open-vocabulary dual-expert predicate head."""

    def __init__(
        self,
        *,
        d_model: int,
        text_dim: int,
        logit_scale_init: float = 5.0,
        projection_layers: int = 2,
        gate_hidden: int = 128,
    ) -> None:
        super().__init__()
        if d_model <= 0 or text_dim <= 0:
            raise ValueError("vocab head dimensions must be positive")
        if logit_scale_init <= 0.0:
            raise ValueError("logit_scale_init must be positive")
        if projection_layers <= 0 or gate_hidden <= 0:
            raise ValueError("invalid vocab head depth/width")

        self.d_model = int(d_model)
        self.text_dim = int(text_dim)

        if projection_layers == 1:
            self.proj: nn.Module = nn.Linear(
                d_model,
                text_dim,
                bias=False,
            )
            nn.init.xavier_uniform_(self.proj.weight)
        else:
            hidden = max(
                d_model * 2,
                text_dim // 2,
            )
            layers: list[nn.Module] = []
            in_dim = d_model
            for _ in range(projection_layers - 1):
                linear = nn.Linear(in_dim, hidden)
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                layers.extend((linear, nn.GELU()))
                in_dim = hidden
            layers.append(nn.LayerNorm(in_dim))
            final = nn.Linear(
                in_dim,
                text_dim,
                bias=False,
            )
            nn.init.xavier_uniform_(final.weight)
            layers.append(final)
            self.proj = nn.Sequential(*layers)

        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(logit_scale_init))
        )
        self.logit_bias = nn.Parameter(torch.zeros(()))

        self.gate_mlp = nn.Sequential(
            nn.Linear(text_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        for module in self.gate_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def routing_alpha(self, predicate_bank: Tensor) -> Tensor:
        if (
            predicate_bank.ndim != 2
            or predicate_bank.shape[1] != self.text_dim
        ):
            raise ValueError(
                "predicate_bank must be [V,text_dim]"
            )
        return torch.sigmoid(
            self.gate_mlp(predicate_bank).squeeze(-1)
        )

    def score_query_dual(
        self,
        semantic_query: Tensor,
        spatial_query: Tensor,
        predicate_bank: Tensor,
        *,
        alpha: Tensor | None = None,
    ) -> Tensor:
        if semantic_query.shape != spatial_query.shape:
            raise ValueError(
                "semantic/spatial queries must have matching shapes"
            )
        if semantic_query.shape[-1] != self.text_dim:
            raise ValueError(
                "query width does not match text dimension"
            )
        if (
            predicate_bank.ndim != 2
            or predicate_bank.shape[1] != self.text_dim
        ):
            raise ValueError(
                "predicate_bank must be [V,text_dim]"
            )
        if alpha is None:
            alpha = self.routing_alpha(predicate_bank)
        if alpha.shape != (predicate_bank.shape[0],):
            raise ValueError("alpha must be [V]")
        if not torch.isfinite(alpha).all():
            raise ValueError("alpha must be finite")

        bank = F.normalize(
            predicate_bank,
            dim=-1,
        )
        semantic_cosine = (
            F.normalize(semantic_query, dim=-1)
            @ bank.transpose(0, 1)
        )
        spatial_cosine = (
            F.normalize(spatial_query, dim=-1)
            @ bank.transpose(0, 1)
        )
        mixed = (
            (1.0 - alpha) * semantic_cosine
            + alpha * spatial_cosine
        )
        scale = self.logit_scale.exp().clamp(max=100.0)
        return mixed * scale + self.logit_bias

    def warm_start_gate(
        self,
        predicate_bank: Tensor,
        target_alpha: Tensor,
        *,
        steps: int = 300,
        learning_rate: float = 1.0e-2,
    ) -> float:
        """Regress gate_mlp(W) onto a precomputed spatial-probe target."""
        if steps <= 0 or learning_rate <= 0.0:
            raise ValueError(
                "warm-start steps and learning rate must be positive"
            )
        if target_alpha.shape != (predicate_bank.shape[0],):
            raise ValueError(
                "target_alpha must have one value per predicate"
            )
        bank = predicate_bank.detach()
        target = target_alpha.detach().to(
            device=bank.device,
            dtype=bank.dtype,
        )
        optimizer = torch.optim.Adam(
            self.gate_mlp.parameters(),
            lr=learning_rate,
        )
        final_loss = None
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = self.routing_alpha(bank)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()
            final_loss = loss.detach()
        if final_loss is None:
            raise RuntimeError("gate warm start did not execute")
        return float(final_loss.cpu())


def balanced_spatial_probe_targets(
    predicate_bank: Tensor,
    is_spatial: Tensor,
    *,
    steps: int = 1000,
    learning_rate: float = 5.0e-2,
) -> Tensor:
    """Fit a balanced logistic probe W -> spatialness using only torch.

    This provides the target probabilities consumed by ApacheVocabHead's
    reference gate warm start while keeping the optional training tool free of
    an additional scikit-learn runtime dependency.
    """
    if predicate_bank.ndim != 2:
        raise ValueError("predicate_bank must be [V,D]")
    if (
        is_spatial.shape != (predicate_bank.shape[0],)
        or is_spatial.dtype != torch.bool
    ):
        raise ValueError(
            "is_spatial must be bool [V]"
        )
    positive = int(is_spatial.sum())
    negative = int((~is_spatial).sum())
    if positive <= 0 or negative <= 0:
        raise ValueError(
            "spatial probe requires both spatial and semantic predicates"
        )
    if steps <= 0 or learning_rate <= 0.0:
        raise ValueError("invalid spatial probe optimizer settings")

    bank = predicate_bank.detach().float()
    labels = is_spatial.to(
        device=bank.device,
        dtype=bank.dtype,
    )
    weight = torch.zeros(
        bank.shape[1],
        device=bank.device,
        dtype=bank.dtype,
        requires_grad=True,
    )
    bias = torch.zeros(
        (),
        device=bank.device,
        dtype=bank.dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        (weight, bias),
        lr=learning_rate,
    )
    positive_weight = torch.tensor(
        negative / positive,
        device=bank.device,
        dtype=bank.dtype,
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = bank @ weight + bias
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=positive_weight,
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return torch.sigmoid(
            bank @ weight + bias
        )
