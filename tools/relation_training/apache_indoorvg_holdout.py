from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence


DERIVATION_SCHEMA = "kfcore.apache-indoorvg-holdout-derivation/1"
RELEASED_SOURCE = "IndoorVG_coco_format"
RELEASED_SPLITS = (
    "val",
    "test",
)
RELEASED_NOTE = (
    "Training-time exclusion: VG ids plus their zero-padded COCO "
    "twins, so one stem-set covers VG- and COCO-keyed packs alike."
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


def stable_json_sha256(
    value: object,
) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        raw
    ).hexdigest()


def holdout_bytes(
    payload: dict[str, object],
) -> bytes:
    # Match the upstream writer's ordinary json.dumps semantics.
    return json.dumps(
        payload
    ).encode("utf-8")


def _parse_coco_id(
    value: object,
    *,
    vg_id: str,
) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"vg2coco value for {vg_id} must be an integer"
        )
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        text = value.strip()
        if (
            not text
            or (
                text[0] in "+-"
                and not text[1:].isdigit()
            )
            or (
                text[0] not in "+-"
                and not text.isdigit()
            )
        ):
            raise ValueError(
                f"vg2coco value for {vg_id} must be an integer"
            )
        result = int(
            text,
            10,
        )
    else:
        raise ValueError(
            f"vg2coco value for {vg_id} must be an integer"
        )
    if result < 0:
        raise ValueError(
            f"vg2coco value for {vg_id} must be non-negative"
        )
    return result


def _validate_splits(
    splits: Sequence[str],
) -> tuple[str, ...]:
    values = tuple(
        splits
    )
    if values != RELEASED_SPLITS:
        raise ValueError(
            "released IndoorVG holdout requires splits=['val','test'] in that order"
        )
    return values


def derive_indoorvg_holdout(
    indoorvg_root: str | Path,
    vg2coco_path: str | Path,
    *,
    splits: Sequence[str] = (
        RELEASED_SPLITS
    ),
) -> tuple[
    dict[str, object],
    dict[str, object],
]:
    released_splits = (
        _validate_splits(
            splits
        )
    )
    root = Path(
        indoorvg_root
    )
    mapping_path = Path(
        vg2coco_path
    )
    if not root.is_dir():
        raise FileNotFoundError(
            root
        )
    if not mapping_path.is_file():
        raise FileNotFoundError(
            mapping_path
        )

    split_reports: list[
        dict[str, object]
    ] = []
    vg_ids: set[str] = set()

    for split in released_splits:
        directory = root / split
        if not directory.is_dir():
            raise FileNotFoundError(
                directory
            )
        # Upstream uses os.listdir and a case-sensitive ".jpg" suffix.
        ids = sorted(
            {
                filename[:-4]
                for filename in os.listdir(
                    directory
                )
                if filename.endswith(
                    ".jpg"
                )
            }
        )
        if any(
            not value
            for value in ids
        ):
            raise ValueError(
                f"IndoorVG {split} contains an empty .jpg stem"
            )
        split_reports.append(
            {
                "split": split,
                "vg_ids": ids,
                "vg_id_count": len(
                    ids
                ),
                "vg_ids_sha256": (
                    ordered_strings_sha256(
                        ids
                    )
                ),
            }
        )
        vg_ids.update(
            ids
        )

    mapping_payload = json.loads(
        mapping_path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(
        mapping_payload,
        dict,
    ):
        raise ValueError(
            "vg2coco.json must be an object"
        )

    mapping: dict[
        str,
        int
    ] = {}
    for raw_key, raw_value in (
        mapping_payload.items()
    ):
        if (
            not isinstance(raw_key, str)
            or not raw_key
        ):
            raise ValueError(
                "vg2coco keys must be non-empty strings"
            )
        mapping[raw_key] = (
            _parse_coco_id(
                raw_value,
                vg_id=raw_key,
            )
        )

    ordered_vg_ids = sorted(
        vg_ids
    )
    mapped_pairs: list[
        dict[str, str]
    ] = []
    unmapped_vg_ids: list[
        str
    ] = []
    coco_stems: set[str] = set()

    for vg_id in ordered_vg_ids:
        if vg_id not in mapping:
            unmapped_vg_ids.append(
                vg_id
            )
            continue
        coco_stem = (
            f"{mapping[vg_id]:012d}"
        )
        mapped_pairs.append(
            {
                "vg_id": vg_id,
                "coco_stem": (
                    coco_stem
                ),
            }
        )
        coco_stems.add(
            coco_stem
        )

    ordered_coco_stems = sorted(
        coco_stems
    )
    stems = sorted(
        vg_ids | coco_stems
    )

    payload = {
        "source": RELEASED_SOURCE,
        "splits": list(
            RELEASED_SPLITS
        ),
        "note": RELEASED_NOTE,
        "vg_ids": ordered_vg_ids,
        "coco_stems": (
            ordered_coco_stems
        ),
        "stems": stems,
    }
    raw = holdout_bytes(
        payload
    )

    evidence = {
        "schema": (
            DERIVATION_SCHEMA
        ),
        "source": (
            RELEASED_SOURCE
        ),
        "splits": list(
            RELEASED_SPLITS
        ),
        "split_inputs": (
            split_reports
        ),
        "vg_ids": (
            ordered_vg_ids
        ),
        "vg_id_count": len(
            ordered_vg_ids
        ),
        "vg_ids_sha256": (
            ordered_strings_sha256(
                ordered_vg_ids
            )
        ),
        "vg2coco_sha256": (
            sha256_file(
                mapping_path
            )
        ),
        "mapped_pairs": (
            mapped_pairs
        ),
        "mapped_vg_count": len(
            mapped_pairs
        ),
        "mapped_pairs_sha256": (
            stable_json_sha256(
                mapped_pairs
            )
        ),
        "unmapped_vg_ids": (
            unmapped_vg_ids
        ),
        "unmapped_vg_count": len(
            unmapped_vg_ids
        ),
        "coco_stems": (
            ordered_coco_stems
        ),
        "coco_stem_count": len(
            ordered_coco_stems
        ),
        "coco_stems_sha256": (
            ordered_strings_sha256(
                ordered_coco_stems
            )
        ),
        "stems": stems,
        "stem_count": len(
            stems
        ),
        "stems_sha256": (
            ordered_strings_sha256(
                stems
            )
        ),
        "output_sha256": (
            hashlib.sha256(
                raw
            ).hexdigest()
        ),
    }
    return payload, evidence


def write_indoorvg_holdout(
    *,
    indoorvg_root: str | Path,
    vg2coco_path: str | Path,
    output: str | Path,
    evidence_output: str | Path,
    splits: Sequence[str] = (
        RELEASED_SPLITS
    ),
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
        derive_indoorvg_holdout(
            indoorvg_root,
            vg2coco_path,
            splits=splits,
        )
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_bytes(
        holdout_bytes(
            payload
        )
    )

    actual_sha = sha256_file(
        output_path
    )
    if (
        actual_sha
        != evidence[
            "output_sha256"
        ]
    ):
        raise RuntimeError(
            "written IndoorVG holdout hash differs from derivation"
        )

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
        "--indoorvg-root",
        required=True,
    )
    parser.add_argument(
        "--vg2coco",
        required=True,
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
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

    report = write_indoorvg_holdout(
        indoorvg_root=(
            args.indoorvg_root
        ),
        vg2coco_path=(
            args.vg2coco
        ),
        output=args.out,
        evidence_output=(
            args.evidence
        ),
        splits=(
            tuple(args.splits)
            if args.splits
            else RELEASED_SPLITS
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
