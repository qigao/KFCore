from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from apache_training_recipe import RELEASED_TEXT_DIM


OBJECT_BANK_SCHEMA = "kfcore.apache-object-text-bank/1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1 << 20),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _names_sha256(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_object_text_bank(
    path: str | Path,
    object_labels: Sequence[str],
    *,
    text_dim: int = RELEASED_TEXT_DIM,
) -> tuple[Tensor, dict[str, object]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if (
        isinstance(text_dim, bool)
        or not isinstance(text_dim, int)
        or text_dim <= 0
    ):
        raise ValueError(
            "object text dimension must be a positive integer"
        )
    labels = tuple(object_labels)
    if not labels:
        raise ValueError(
            "object text bank requires a non-empty object vocabulary"
        )
    if any(
        not isinstance(name, str) or not name
        for name in labels
    ):
        raise ValueError(
            "object labels must be non-empty strings"
        )
    if len(set(labels)) != len(labels):
        raise ValueError(
            "object labels must be unique"
        )

    try:
        with np.load(
            source,
            allow_pickle=False,
        ) as payload:
            if (
                "names" not in payload
                or "embeddings" not in payload
            ):
                raise ValueError(
                    "Apache object bank requires names and embeddings arrays"
                )
            raw_names = payload["names"]
            embeddings = payload["embeddings"]
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            "failed to read Apache object bank NPZ"
        ) from error

    if raw_names.ndim != 1:
        raise ValueError(
            "object bank names must be [O]"
        )
    names = tuple(str(value) for value in raw_names.tolist())
    if names != labels:
        raise ValueError(
            "object bank names/order do not match relation vocabulary"
        )

    array = np.asarray(
        embeddings,
        dtype=np.float32,
    )
    if array.ndim != 2:
        raise ValueError(
            "object bank embeddings must be [O,D]"
        )
    if array.shape != (
        len(labels),
        text_dim,
    ):
        raise ValueError(
            f"object bank embeddings must be [{len(labels)},{text_dim}]"
        )
    if not np.isfinite(array).all():
        raise ValueError(
            "object bank embeddings must be finite"
        )

    tensor = torch.from_numpy(
        np.ascontiguousarray(array)
    )
    report = {
        "schema": OBJECT_BANK_SCHEMA,
        "artifact_sha256": sha256_file(
            source
        ),
        "shape": [
            int(array.shape[0]),
            int(array.shape[1]),
        ],
        "source_dtype": str(
            np.asarray(embeddings).dtype
        ),
        "runtime_dtype": str(tensor.dtype),
        "object_label_count": len(labels),
        "object_label_order": list(labels),
        "object_label_order_sha256": (
            _names_sha256(labels)
        ),
        "text_dim": text_dim,
    }
    return tensor, report
