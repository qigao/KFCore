from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from apache_objective import SOURCE_ALLOW_SCHEMA


DERIVATION_SCHEMA = "kfcore.apache-source-allow-derivation/1"
DERIVATION_ALGORITHM = "unrestricted-all; restricted-local-intersection"
RELEASED_RESTRICTED_SOURCES = ("hicodet",)


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


def _source_name(split_dir: Path) -> str:
    return (
        split_dir.parent.name
        if split_dir.name == "train"
        else split_dir.name
    )


def _load_predicates(
    split_dir: Path,
) -> tuple[list[str], dict[str, object]]:
    meta_path = split_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    payload = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(payload, dict):
        raise ValueError(
            "Apache pack meta.json must be an object"
        )
    predicates = payload.get("predicates")
    if (
        not isinstance(predicates, list)
        or not predicates
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
    name = _source_name(
        split_dir
    )
    return (
        list(predicates),
        {
            "source_name": name,
            "pack_split": (
                f"{name}/train"
                if split_dir.name == "train"
                else name
            ),
            "meta_sha256": sha256_file(
                meta_path
            ),
            "local_predicate_count": len(
                predicates
            ),
        },
    )


def derive_source_allow(
    pack_splits: Sequence[
        str | Path
    ],
    predicates: Sequence[str],
    *,
    restricted_sources: Sequence[
        str
    ] = RELEASED_RESTRICTED_SOURCES,
) -> tuple[
    dict[str, object],
    dict[str, object],
]:
    union = tuple(
        predicates
    )
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

    restricted = tuple(
        restricted_sources
    )
    if (
        not restricted
        or any(
            not isinstance(name, str)
            or not name
            for name in restricted
        )
        or len(set(restricted)) != len(
            restricted
        )
    ):
        raise ValueError(
            "restricted source names must be unique non-empty strings"
        )
    if not pack_splits:
        raise ValueError(
            "at least one Apache pack split is required"
        )

    rows: list[
        dict[str, object]
    ] = []
    source_reports: list[
        dict[str, object]
    ] = []
    seen_sources: set[str] = set()

    for raw_path in pack_splits:
        split_dir = Path(
            raw_path
        )
        local_predicates, report = (
            _load_predicates(
                split_dir
            )
        )
        source_name = str(
            report["source_name"]
        )
        if source_name in seen_sources:
            raise ValueError(
                f"duplicate source pack: {source_name}"
            )
        seen_sources.add(
            source_name
        )
        local_set = set(
            local_predicates
        )
        if source_name in restricted:
            allowed = [
                name
                for name in union
                if name in local_set
            ]
            if not allowed:
                raise ValueError(
                    f"restricted source {source_name} has no predicates in union vocabulary"
                )
            ignored = [
                name
                for name in local_predicates
                if name not in set(union)
            ]
        else:
            allowed = list(
                union
            )
            ignored = []

        rows.append(
            {
                "name": source_name,
                "predicates": allowed,
            }
        )
        report[
            "restricted"
        ] = (
            source_name in restricted
        )
        report[
            "allowed_predicate_count"
        ] = len(allowed)
        report[
            "ignored_predicates"
        ] = ignored
        source_reports.append(
            report
        )

    missing_restricted = [
        name
        for name in restricted
        if name not in seen_sources
    ]
    if missing_restricted:
        raise ValueError(
            "missing restricted source pack: "
            + ", ".join(
                missing_restricted
            )
        )

    payload = {
        "schema": SOURCE_ALLOW_SCHEMA,
        "sources": rows,
    }
    sidecar_sha256 = hashlib.sha256(
        stable_json_bytes(
            payload
        )
    ).hexdigest()
    evidence = {
        "schema": DERIVATION_SCHEMA,
        "algorithm": DERIVATION_ALGORITHM,
        "restricted_sources": list(
            restricted
        ),
        "union_predicate_count": len(
            union
        ),
        "sources": source_reports,
        "sidecar_sha256": sidecar_sha256,
    }
    return payload, evidence


def write_source_allow(
    *,
    pack_splits: Sequence[
        str | Path
    ],
    predicates: Sequence[str],
    output: str | Path,
    evidence_output: str | Path,
    restricted_sources: Sequence[
        str
    ] = RELEASED_RESTRICTED_SOURCES,
) -> dict[str, object]:
    out = Path(
        output
    )
    evidence_path = Path(
        evidence_output
    )
    if out.exists():
        raise FileExistsError(
            out
        )
    if evidence_path.exists():
        raise FileExistsError(
            evidence_path
        )

    payload, evidence = (
        derive_source_allow(
            pack_splits,
            predicates,
            restricted_sources=restricted_sources,
        )
    )
    raw = stable_json_bytes(
        payload
    )
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out.write_bytes(
        raw
    )
    evidence = {
        **evidence,
        "sidecar_sha256": sha256_file(
            out
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

    write_source_allow(
        pack_splits=args.pack_split,
        predicates=raw,
        output=args.out,
        evidence_output=args.evidence,
    )


if __name__ == "__main__":
    main()
