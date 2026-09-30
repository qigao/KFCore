from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Sequence
import zipfile

import numpy as np

from apache_pair_sampler import PairOpportunityTable


REBUILD_SCHEMA = "kfcore.apache-pair-opportunity-rebuild/1"
RELEASED_SOURCE_NAME = "megasg_clean"
RELEASED_SCAN_BOX_CAP = 400
RELEASED_MIN_SUPPORT = 50
NPZ_FORMAT = "deterministic-zip-stored-v1"

_REQUIRED_COMPONENTS = (
    "meta.json",
    "img_meta.npy",
    "box_cats.npy",
    "rels.npy",
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


def _names_sha256(
    names: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_names(
    values: object,
    name: str,
) -> tuple[str, ...]:
    if (
        not isinstance(values, list)
        or not values
        or any(
            not isinstance(value, str)
            or not value
            for value in values
        )
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            f"{name} must be unique non-empty strings"
        )
    return tuple(values)


def _npy_bytes(
    array: np.ndarray,
) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(array),
        allow_pickle=False,
    )
    return stream.getvalue()


def _write_deterministic_npz(
    path: Path,
    arrays: Sequence[
        tuple[str, np.ndarray]
    ],
) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name, array in arrays:
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(
                    1980,
                    1,
                    1,
                    0,
                    0,
                    0,
                ),
            )
            info.compress_type = (
                zipfile.ZIP_STORED
            )
            info.create_system = 3
            info.external_attr = (
                0o100644 << 16
            )
            archive.writestr(
                info,
                _npy_bytes(
                    np.asarray(array)
                ),
            )


def _load_pack(
    pack_split: Path,
    object_labels: Sequence[str],
) -> tuple[
    dict[str, object],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    for name in _REQUIRED_COMPONENTS:
        if not (
            pack_split / name
        ).is_file():
            raise FileNotFoundError(
                pack_split / name
            )

    meta = json.loads(
        (
            pack_split
            / "meta.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(meta, dict):
        raise ValueError(
            "Apache pack meta.json must be an object"
        )
    categories = _validate_names(
        meta.get("categories"),
        "Apache pack categories",
    )
    expected_labels = tuple(
        object_labels
    )
    if (
        not expected_labels
        or any(
            not isinstance(value, str)
            or not value
            for value in expected_labels
        )
        or len(set(expected_labels))
        != len(expected_labels)
    ):
        raise ValueError(
            "relation object labels must be unique non-empty strings"
        )
    if categories != expected_labels:
        raise ValueError(
            "Apache pack category order does not match relation object vocabulary"
        )

    img_meta = np.load(
        pack_split
        / "img_meta.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    box_cats = np.load(
        pack_split
        / "box_cats.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    rels = np.load(
        pack_split
        / "rels.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    if (
        img_meta.ndim != 2
        or img_meta.shape[1] != 7
        or not np.issubdtype(
            img_meta.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "img_meta.npy must be integer [N,7]"
        )
    if (
        box_cats.ndim != 1
        or not np.issubdtype(
            box_cats.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "box_cats.npy must be integer [B]"
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
            "rels.npy must be integer [R,5]"
        )

    if (
        "num_images" in meta
        and int(
            meta["num_images"]
        )
        != img_meta.shape[0]
    ):
        raise ValueError(
            "pack meta num_images does not match img_meta"
        )
    if (
        "num_boxes" in meta
        and int(
            meta["num_boxes"]
        )
        != box_cats.shape[0]
    ):
        raise ValueError(
            "pack meta num_boxes does not match box_cats"
        )
    if (
        "num_rels" in meta
        and int(
            meta["num_rels"]
        )
        != rels.shape[0]
    ):
        raise ValueError(
            "pack meta num_rels does not match rels"
        )

    return (
        meta,
        img_meta,
        box_cats,
        rels,
        categories,
    )


def rebuild_pair_opportunity(
    pack_split: str | Path,
    object_labels: Sequence[str],
    *,
    source_name: str = (
        RELEASED_SOURCE_NAME
    ),
    scan_box_cap: int = (
        RELEASED_SCAN_BOX_CAP
    ),
    min_support: int = (
        RELEASED_MIN_SUPPORT
    ),
) -> tuple[
    dict[str, np.ndarray],
    dict[str, object],
]:
    if source_name != RELEASED_SOURCE_NAME:
        raise ValueError(
            "released pair-opportunity rebuild requires source_name=megasg_clean"
        )
    if (
        isinstance(scan_box_cap, bool)
        or not isinstance(
            scan_box_cap,
            int,
        )
        or scan_box_cap
        != RELEASED_SCAN_BOX_CAP
    ):
        raise ValueError(
            "released pair-opportunity rebuild requires scan_box_cap=400"
        )
    if (
        isinstance(min_support, bool)
        or not isinstance(
            min_support,
            int,
        )
        or min_support
        != RELEASED_MIN_SUPPORT
    ):
        raise ValueError(
            "released pair-opportunity rebuild requires min_support=50"
        )

    root = Path(
        pack_split
    )
    (
        _meta,
        img_meta,
        box_cats,
        rels,
        categories,
    ) = _load_pack(
        root,
        object_labels,
    )
    category_count = len(
        categories
    )
    opportunities = np.zeros(
        (
            category_count,
            category_count,
        ),
        dtype=np.int64,
    )
    relation_counts = np.zeros(
        category_count
        * category_count,
        dtype=np.int64,
    )

    scanned_images = 0
    scanned_boxes = 0
    scanned_relations = 0
    dropped_relations_by_cap = 0

    for image_index, row in enumerate(
        img_meta
    ):
        (
            _image_id,
            _width,
            _height,
            raw_box_start,
            raw_box_count,
            raw_rel_start,
            raw_rel_count,
        ) = (
            int(value)
            for value in row.tolist()
        )
        box_start = raw_box_start
        full_box_count = (
            raw_box_count
        )
        rel_start = raw_rel_start
        rel_count = raw_rel_count

        if (
            box_start < 0
            or full_box_count < 0
            or box_start
            + full_box_count
            > box_cats.shape[0]
        ):
            raise ValueError(
                f"pack image {image_index} has invalid box range"
            )
        if (
            rel_start < 0
            or rel_count < 0
            or rel_start
            + rel_count
            > rels.shape[0]
        ):
            raise ValueError(
                f"pack image {image_index} has invalid relation range"
            )

        box_count = min(
            full_box_count,
            scan_box_cap,
        )
        if box_count < 2:
            continue

        cats = np.asarray(
            box_cats[
                box_start :
                box_start
                + box_count
            ],
            dtype=np.int64,
        )
        if (
            cats.size
            and (
                cats.min() < 0
                or cats.max()
                >= category_count
            )
        ):
            raise ValueError(
                f"pack image {image_index} has category id outside vocabulary"
            )

        if rel_count:
            image_rels = np.asarray(
                rels[
                    rel_start :
                    rel_start
                    + rel_count
                ],
                dtype=np.int64,
            )
            endpoints = (
                image_rels[:, :2]
            )
            if (
                endpoints.size
                and (
                    endpoints.min() < 0
                    or endpoints.max()
                    >= full_box_count
                )
            ):
                raise ValueError(
                    f"pack image {image_index} has invalid relation endpoints"
                )
            keep = (
                (image_rels[:, 0] < box_count)
                & (
                    image_rels[:, 1]
                    < box_count
                )
            )
            dropped_relations_by_cap += int(
                (~keep).sum()
            )
            image_rels = (
                image_rels[keep]
            )
            if image_rels.size:
                subject_cats = cats[
                    image_rels[:, 0]
                ]
                object_cats = cats[
                    image_rels[:, 1]
                ]
                flat = (
                    subject_cats
                    * category_count
                    + object_cats
                )
                np.add.at(
                    relation_counts,
                    flat,
                    1,
                )
                scanned_relations += int(
                    image_rels.shape[0]
                )

        if cats.size < 2:
            continue

        unique_cats, counts = (
            np.unique(
                cats,
                return_counts=True,
            )
        )
        block = np.outer(
            counts,
            counts,
        ).astype(
            np.int64,
            copy=False,
        )
        np.fill_diagonal(
            block,
            counts
            * (
                counts - 1
            ),
        )
        opportunities[
            np.ix_(
                unique_cats,
                unique_cats,
            )
        ] += block
        scanned_images += 1
        scanned_boxes += int(
            cats.size
        )

    opportunities_flat = (
        opportunities.reshape(-1)
    )
    rate = np.zeros(
        category_count
        * category_count,
        dtype=np.float32,
    )
    valid = (
        opportunities_flat > 0
    )
    rate[valid] = np.minimum(
        1.0,
        relation_counts[valid]
        / opportunities_flat[valid],
    )

    arrays = {
        "rate": rate,
        "opportunities": (
            opportunities_flat
            .astype(
                np.int64,
                copy=False,
            )
        ),
        "relations": (
            relation_counts.astype(
                np.int64,
                copy=False,
            )
        ),
        "num_cats": np.asarray(
            category_count,
            dtype=np.int32,
        ),
        "min_support": np.asarray(
            min_support,
            dtype=np.int32,
        ),
    }

    component_hashes = {
        name: sha256_file(
            root / name
        )
        for name in (
            "meta.json",
            "img_meta.npy",
            "box_cats.npy",
            "rels.npy",
        )
    }
    trusted = (
        opportunities_flat
        >= min_support
    )
    evidence = {
        "schema": REBUILD_SCHEMA,
        "source_name": source_name,
        "pack_split": str(
            root.resolve()
        ),
        "component_sha256": (
            component_hashes
        ),
        "object_label_order": list(
            categories
        ),
        "object_label_order_sha256": (
            _names_sha256(
                categories
            )
        ),
        "num_cats": (
            category_count
        ),
        "scan_box_cap": (
            scan_box_cap
        ),
        "min_support": min_support,
        "full_scan": True,
        "extrapolated": False,
        "opportunity_semantics": (
            "ordered-instance-pairs-minus-self-on-diagonal"
        ),
        "numerator_semantics": (
            "same-pack-relations-after-400-box-cap"
        ),
        "rate_semantics": (
            "min(1,relations/opportunities)"
        ),
        "images_total": int(
            img_meta.shape[0]
        ),
        "images_scanned": int(
            scanned_images
        ),
        "boxes_scanned": int(
            scanned_boxes
        ),
        "relations_scanned": int(
            scanned_relations
        ),
        "relations_dropped_by_box_cap": int(
            dropped_relations_by_cap
        ),
        "category_pairs_with_opportunity": int(
            valid.sum()
        ),
        "trusted_category_pairs": int(
            trusted.sum()
        ),
        "opportunity_sum": int(
            opportunities_flat.sum()
        ),
        "relation_sum": int(
            relation_counts.sum()
        ),
        "npz_format": NPZ_FORMAT,
    }
    return arrays, evidence


def write_pair_opportunity(
    *,
    pack_split: str | Path,
    object_labels: Sequence[str],
    output: str | Path,
    evidence_output: str | Path,
    source_name: str = (
        RELEASED_SOURCE_NAME
    ),
    scan_box_cap: int = (
        RELEASED_SCAN_BOX_CAP
    ),
    min_support: int = (
        RELEASED_MIN_SUPPORT
    ),
) -> dict[str, object]:
    output_path = Path(output)
    evidence_path = Path(
        evidence_output
    )
    if evidence_path.exists():
        raise FileExistsError(
            evidence_path
        )

    arrays, evidence = (
        rebuild_pair_opportunity(
            pack_split,
            object_labels,
            source_name=source_name,
            scan_box_cap=(
                scan_box_cap
            ),
            min_support=min_support,
        )
    )

    stable_meta = {
        "source_name": (
            source_name
        ),
        "scan_box_cap": (
            scan_box_cap
        ),
        "min_support": (
            min_support
        ),
        "note": (
            "rate = relations / ordered INSTANCE-pair opportunities; "
            "1-rate approximates P(unannotated candidate is a true negative)"
        ),
    }
    arrays_with_meta = [
        (
            "rate",
            arrays["rate"],
        ),
        (
            "opportunities",
            arrays[
                "opportunities"
            ],
        ),
        (
            "relations",
            arrays["relations"],
        ),
        (
            "num_cats",
            arrays["num_cats"],
        ),
        (
            "min_support",
            arrays[
                "min_support"
            ],
        ),
        (
            "meta",
            np.asarray(
                [
                    json.dumps(
                        stable_meta,
                        sort_keys=True,
                        separators=(
                            ",",
                            ":",
                        ),
                        ensure_ascii=False,
                    )
                ],
                dtype=np.str_,
            ),
        ),
    ]
    _write_deterministic_npz(
        output_path,
        arrays_with_meta,
    )

    # Validate emitted bytes through the runtime consumer.
    table = PairOpportunityTable.load(
        output_path
    )
    if (
        table.num_cats
        != int(
            arrays["num_cats"]
        )
        or table.min_support
        != min_support
    ):
        raise RuntimeError(
            "written pair-opportunity table failed runtime round-trip"
        )

    evidence = {
        **evidence,
        "output_sha256": (
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
        "--pack",
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
    parser.add_argument(
        "--source-name",
        default=(
            RELEASED_SOURCE_NAME
        ),
    )
    args = parser.parse_args()

    payload = json.loads(
        Path(
            args.vocabulary
        ).read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(payload, dict)
        or not isinstance(
            payload.get("objects"),
            list,
        )
    ):
        raise ValueError(
            "relation vocabulary JSON must contain objects"
        )

    write_pair_opportunity(
        pack_split=args.pack,
        object_labels=payload[
            "objects"
        ],
        output=args.out,
        evidence_output=(
            args.evidence
        ),
        source_name=(
            args.source_name
        ),
    )


if __name__ == "__main__":
    main()
