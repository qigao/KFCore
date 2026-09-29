from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model import KFRelationModel, RelationModelConfig


CHECKPOINT_SCHEMA = "kfcore.relation-training/1"


def save_checkpoint(
    path: str | Path,
    model: KFRelationModel,
    *,
    backbone_model: str,
    predicates: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    if len(predicates) != int(model.predicate_bank.shape[0]):
        raise ValueError("predicate names do not match predicate bank rows")
    if any(not name for name in predicates):
        raise ValueError("predicate names must be non-empty")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "backbone_model": backbone_model,
        "config": {
            "image_size": model.config.image_size,
            "max_boxes": model.config.max_boxes,
            "pair_budget": model.config.pair_budget,
            "hidden_dim": model.config.hidden_dim,
            "geometry_dim": model.config.geometry_dim,
            "num_heads": model.config.num_heads,
            "num_layers": model.config.num_layers,
            "dropout": model.config.dropout,
            "tap_indices": tuple(model.config.tap_indices),
            "predicate_adapter_rank": model.config.predicate_adapter_rank,
            "pair_visual_evidence": model.config.pair_visual_evidence,
            "pair_geometry_evidence": model.config.pair_geometry_evidence,
            "pair_evidence_contract": model.config.pair_evidence_contract,
            "pair_sampler_contract": model.config.pair_sampler_contract,
        },
        "predicates": list(predicates),
        "predicate_embeddings": model.predicate_bank.detach().cpu(),
        "state_dict": model.state_dict(),
        "extra": dict(extra or {}),
    }
    torch.save(payload, Path(path))


def load_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported relation checkpoint schema")
    required = {
        "backbone_model",
        "config",
        "predicates",
        "predicate_embeddings",
        "state_dict",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"relation checkpoint is missing fields: {sorted(missing)}")
    return payload


def config_from_payload(payload: dict[str, Any]) -> RelationModelConfig:
    raw = dict(payload["config"])
    raw["tap_indices"] = tuple(raw["tap_indices"])
    return RelationModelConfig(**raw)
