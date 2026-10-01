from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from apache_mixture import (
    RELEASED_SOURCE_NAMES,
)
from apache_objective import (
    SOURCE_ALLOW_SCHEMA,
    load_source_column_allow,
)


DERIVATION_SCHEMA = "kfcore.apache-source-column-derivation/1"
RELEASED_RESTRICTED_SOURCES = (
    "hicodet",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1 << 20),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_strings_sha256(
    values: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(
            value.encode("utf-8")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def stable_json_bytes(
    payload: object,
) -> bytes:
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


def _validate_names(
    values: Sequence[str],
    name: str,
) -> tuple[str, ...]:
    result = tuple(values)
    if (
        not result
        or any(
            not isinstance(value, str)
            or not value
            for value in result
        )
        or len(set(result)) != len(result)
    ):
        raise ValueError(
            f"{name} must be unique non-empty strings"
        )
    return result


def _source_name(
    split_dir: Path,
) -> str:
    return (
        split_dir.parent.name
        if split_dir.name == "train"
        else split_dir.name
    )


def derive_source_column_allow(
    pack_splits: Sequence[
        str | Path
    ],
    predicates: Sequence[str],
    *,
    restricted_sources: Sequence[str] = (
        RELEASED_RESTRICTED_SOURCES
    ),
) -> tuple[
    dict[str, object],
    dict[str, object],
]:
    union = _validate_names(
        predicates,
        "union predicates",
    )
    restricted = tuple(
        restricted_sources
    )
    if restricted != (
        RELEASED_RESTRICTED_SOURCES
    ):
        raise ValueError(
            "released source-column derivation requires restricted_sources=['hicodet']"
        )
    if len(pack_splits) != len(
        RELEASED_SOURCE_NAMES
    ):
        raise ValueError(
            "released source-column derivation requires exactly three pack splits"
        )

    union_set = set(union)
    source_payloads: list[
        dict[str, object]
    ] = []
    source_reports: list[
        dict[str, object]
    ] = []
    source_names: list[str] = []

    for raw_path in pack_splits:
        split_dir = Path(
            raw_path
        )
        meta_path = (
            split_dir
            / "meta.json"
        )
        if not meta_path.is_file():
            raise FileNotFoundError(
                meta_path
            )
        source_name = _source_name(
            split_dir
        )
        source_names.append(
            source_name
        )

        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(meta, dict):
            raise ValueError(
                f"{source_name} meta.json must be an object"
            )
        local = meta.get(
            "predicates"
        )
        if not isinstance(
            local,
            list,
        ):
            raise ValueError(
                f"{source_name} meta predicates must be an array"
            )
        local_names = _validate_names(
            local,
            f"{source_name} local predicates",
        )
        local_set = set(
            local_names
        )
        unknown_local = [
            name
            for name in local_names
            if name not in union_set
        ]

        if source_name in restricted:
            allowed = [
                name
                for name in union
                if name in local_set
            ]
            if not allowed:
                raise ValueError(
                    f"restricted source {source_name} has no predicates in the union vocabulary"
                )
        else:
            allowed = list(
                union
            )

        source_payloads.append(
            {
                "name": source_name,
                "predicates": (
                    allowed
                ),
            }
        )
        source_reports.append(
            {
                "source_name": (
                    source_name
                ),
                "pack_split": str(
                    split_dir.resolve()
                ),
                "meta_sha256": (
                    sha256_file(
                        meta_path
                    )
                ),
                "local_predicates": list(
                    local_names
                ),
                "local_predicate_count": (
                    len(local_names)
                ),
                "local_predicate_order_sha256": (
                    ordered_strings_sha256(
                        local_names
                    )
                ),
                "unknown_local_predicates": (
                    unknown_local
                ),
                "restricted": (
                    source_name
                    in restricted
                ),
                "allowed_predicates": list(
                    allowed
                ),
                "allowed_predicate_count": (
                    len(allowed)
                ),
                "allowed_predicate_order_sha256": (
                    ordered_strings_sha256(
                        allowed
                    )
                ),
            }
        )

    if tuple(source_names) != (
        RELEASED_SOURCE_NAMES
    ):
        raise ValueError(
            "released source-column derivation source order must be megasg_clean/vg_raw/hicodet"
        )

    payload = {
        "schema": SOURCE_ALLOW_SCHEMA,
        "sources": (
            source_payloads
        ),
    }
    raw = stable_json_bytes(
        payload
    )
    evidence = {
        "schema": (
            DERIVATION_SCHEMA
        ),
        "source_order": list(
            RELEASED_SOURCE_NAMES
        ),
        "restricted_sources": list(
            RELEASED_RESTRICTED_SOURCES
        ),
        "union_predicate_order": list(
            union
        ),
        "union_predicate_count": (
            len(union)
        ),
        "union_predicate_order_sha256": (
            ordered_strings_sha256(
                union
            )
        ),
        "sources": source_reports,
        "sidecar_sha256": (
            hashlib.sha256(
                raw
            ).hexdigest()
        ),
    }
    return payload, evidence


def write_source_column_allow(
    *,
    pack_splits: Sequence[
        str | Path
    ],
    predicates: Sequence[str],
    output: str | Path,
    evidence_output: str | Path,
) -> dict[str, object]:
    output_path = Path(
        output
    )
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
        derive_source_column_allow(
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

    names, allow = (
        load_source_column_allow(
            output_path,
            predicates,
        )
    )
    if names != RELEASED_SOURCE_NAMES:
        raise RuntimeError(
            "written source-column sidecar failed source-order round-trip"
        )
    if allow.shape != (
        len(
            RELEASED_SOURCE_NAMES
        ),
        len(predicates),
    ):
        raise RuntimeError(
            "written source-column sidecar failed shape round-trip"
        )

    evidence = {
        **evidence,
        "sidecar_sha256": (
            sha256_file(
                output_path
            )
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
        "--vocabulary",
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

    vocab = json.loads(
        Path(
            args.vocabulary
        ).read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(vocab, dict)
        or not isinstance(
            vocab.get(
                "predicates"
            ),
            list,
        )
    ):
        raise ValueError(
            "relation vocabulary must contain predicates"
        )

    report = write_source_column_allow(
        pack_splits=args.pack_split,
        predicates=vocab[
            "predicates"
        ],
        output=args.out,
        evidence_output=(
            args.evidence
        ),
    )
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
