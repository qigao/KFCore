from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from apache_vocab_head import SPATIAL_FLAGS_SCHEMA


DERIVATION_SCHEMA = "kfcore.apache-spatial-flags-derivation/1"
DERIVATION_ALGORITHM = (
    "per-source predicate spatial majority "
    "then union-any-source"
)
SPATIAL_BIT = 1
MAJORITY_THRESHOLD = 0.5
MAJORITY_COMPARATOR = ">="


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1 << 20),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_pack_flags(
    split_dir: Path,
) -> tuple[list[str], np.ndarray, dict[str, object]]:
    meta_path = split_dir / "meta.json"
    rels_path = split_dir / "rels.npy"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    if not rels_path.is_file():
        raise FileNotFoundError(rels_path)

    meta = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(meta, dict):
        raise ValueError(
            "Apache pack meta.json must be an object"
        )
    predicates = meta.get("predicates")
    if (
        not isinstance(predicates, list)
        or any(
            not isinstance(name, str)
            or not name
            for name in predicates
        )
        or len(set(predicates)) != len(predicates)
    ):
        raise ValueError(
            "Apache pack predicates must be unique non-empty strings"
        )

    rels = np.load(
        rels_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        rels.ndim != 2
        or rels.shape[1] != 5
        or not np.issubdtype(
            rels.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "Apache pack rels.npy must be integer [R,5]"
        )

    if rels.shape[0]:
        predicate_ids = np.asarray(
            rels[:, 2],
            dtype=np.int64,
        )
        if (
            predicate_ids.min() < 0
            or predicate_ids.max()
            >= len(predicates)
        ):
            raise ValueError(
                "Apache pack relation predicate id is out of range"
            )
    else:
        predicate_ids = np.empty(
            (0,),
            dtype=np.int64,
        )

    flags = np.asarray(
        rels[:, 3],
        dtype=np.int64,
    )
    spatial_bits = (
        flags & np.int64(1)
    ).astype(
        np.float64,
        copy=False,
    )

    counts = np.bincount(
        predicate_ids,
        minlength=len(predicates),
    )
    spatial_counts = np.bincount(
        predicate_ids,
        weights=spatial_bits,
        minlength=len(predicates),
    )
    majority = (
        spatial_counts
        >= 0.5
        * np.maximum(
            counts,
            1,
        )
    )

    source_name = (
        split_dir.parent.name
        if split_dir.name == "train"
        else split_dir.name
    )
    report = {
        "source_name": source_name,
        "pack_split": str(
            split_dir.resolve()
        ),
        "meta_sha256": sha256_file(
            meta_path
        ),
        "rels_sha256": sha256_file(
            rels_path
        ),
        "relations": int(
            rels.shape[0]
        ),
        "local_predicate_count": len(
            predicates
        ),
        "supported_predicate_count": int(
            (counts > 0).sum()
        ),
        "local_spatial_majority_count": int(
            (
                majority
                & (counts > 0)
            ).sum()
        ),
    }
    return (
        list(predicates),
        np.stack(
            (
                counts.astype(np.int64),
                spatial_counts.astype(
                    np.float64
                ),
                majority.astype(bool),
            ),
            axis=1,
        ),
        report,
    )


def derive_spatial_flags(
    pack_splits: Sequence[
        str | Path
    ],
    predicates: Sequence[str],
) -> tuple[
    dict[str, object],
    dict[str, object],
]:
    union = tuple(predicates)
    if not union:
        raise ValueError(
            "union predicate vocabulary must not be empty"
        )
    if any(
        not isinstance(name, str)
        or not name
        for name in union
    ):
        raise ValueError(
            "union predicates must be non-empty strings"
        )
    if len(set(union)) != len(union):
        raise ValueError(
            "union predicates must be unique"
        )
    if not pack_splits:
        raise ValueError(
            "at least one Apache pack split is required"
        )

    union_index = {
        name: index
        for index, name in enumerate(
            union
        )
    }
    flags = np.zeros(
        len(union),
        dtype=bool,
    )
    support = np.zeros(
        len(union),
        dtype=np.int64,
    )
    source_reports: list[
        dict[str, object]
    ] = []

    for raw_path in pack_splits:
        split_dir = Path(
            raw_path
        )
        (
            local_predicates,
            stats,
            source_report,
        ) = _load_pack_flags(
            split_dir
        )
        ignored: list[str] = []
        for local_index, name in enumerate(
            local_predicates
        ):
            union_id = union_index.get(
                name
            )
            if union_id is None:
                ignored.append(name)
                continue
            count = int(
                stats[
                    local_index,
                    0,
                ]
            )
            support[union_id] += (
                count
            )
            if (
                count > 0
                and bool(
                    stats[
                        local_index,
                        2,
                    ]
                )
            ):
                flags[union_id] = True

        source_report[
            "ignored_predicates"
        ] = ignored
        source_reports.append(
            source_report
        )

    payload = {
        "schema": SPATIAL_FLAGS_SCHEMA,
        "predicates": list(
            union
        ),
        "is_spatial": [
            bool(value)
            for value in flags.tolist()
        ],
    }
    raw = stable_json_bytes(
        payload
    )
    evidence = {
        "schema": DERIVATION_SCHEMA,
        "algorithm": DERIVATION_ALGORITHM,
        "spatial_bit": SPATIAL_BIT,
        "majority_threshold": MAJORITY_THRESHOLD,
        "majority_comparator": MAJORITY_COMPARATOR,
        "sources": source_reports,
        "union_predicate_count": len(
            union
        ),
        "supported_union_predicate_count": int(
            (support > 0).sum()
        ),
        "spatial_union_predicate_count": int(
            flags.sum()
        ),
        "unsupported_predicates": [
            union[index]
            for index in np.nonzero(
                support == 0
            )[0].tolist()
        ],
        "sidecar_sha256": hashlib.sha256(
            raw
        ).hexdigest(),
    }
    return payload, evidence


def write_spatial_flags(
    *,
    pack_splits: Sequence[
        str | Path
    ],
    predicates: Sequence[str],
    output: str | Path,
    evidence_output: str | Path,
) -> dict[str, object]:
    output_path = Path(output)
    evidence_path = Path(
        evidence_output
    )
    if output_path.exists():
        raise FileExistsError(
            output_path
        )
    if evidence_path.exists():
        raise FileExistsError(
            evidence_path
        )

    payload, evidence = (
        derive_spatial_flags(
            pack_splits,
            predicates,
        )
    )
    raw = stable_json_bytes(
        payload
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_bytes(
        raw
    )

    evidence = {
        **evidence,
        "sidecar_sha256": sha256_file(
            output_path
        ),
    }
    evidence_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    evidence_path.write_text(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack-split",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--predicates",
        required=True,
        help=(
            "JSON array or relation vocabulary JSON "
            "containing predicates."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
    )
    parser.add_argument(
        "--evidence",
        required=True,
    )
    args = parser.parse_args()

    raw = json.loads(
        Path(
            args.predicates
        ).read_text(
            encoding="utf-8"
        )
    )
    if isinstance(raw, dict):
        raw = raw.get(
            "predicates"
        )
    if not isinstance(raw, list):
        raise ValueError(
            "predicate input must contain an array"
        )

    write_spatial_flags(
        pack_splits=args.pack_split,
        predicates=raw,
        output=args.out,
        evidence_output=args.evidence,
    )


if __name__ == "__main__":
    main()
