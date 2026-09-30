from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


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
