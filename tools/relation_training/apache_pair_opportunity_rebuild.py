from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from apache_pair_sampler import PairOpportunityTable


DERIVATION_SCHEMA = "kfcore.apache-pair-opportunity-derivation/1"
DERIVATION_ALGORITHM = (
    "ordered-instance-opportunities; same-pack-relations; clamp-rate"
)
RELEASED_SOURCE = "megasg_clean"
RELEASED_SCAN_BOX_CAP = 400
RELEASED_MIN_SUPPORT = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1 << 20),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(
    names: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(
            name.encode("utf-8")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _source_name(
    split_dir: Path,
) -> str:
    return (
        split_dir.parent.name
        if split_dir.name == "train"
        else split_dir.name
    )


def _load_pack(
    split_dir: Path,
    object_labels: Sequence[str],
) -> tuple[
    dict[str, object],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    meta_path = split_dir / "meta.json"
    img_meta_path = (
        split_dir / "img_meta.npy"
    )
    box_cats_path = (
        split_dir / "box_cats.npy"
    )
    rels_path = split_dir / "rels.npy"
    for path in (
        meta_path,
        img_meta_path,
        box_cats_path,
        rels_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_name = _source_name(
        split_dir
    )
    if source_name != RELEASED_SOURCE:
        raise ValueError(
            "released pair-opportunity rebuild requires megasg_clean/train"
        )

    meta = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(meta, dict):
        raise ValueError(
            "Apache pack meta.json must be an object"
        )
    categories = meta.get(
        "categories"
    )
    if (
        not isinstance(categories, list)
        or any(
            not isinstance(name, str)
            or not name
            for name in categories
        )
        or len(set(categories))
        != len(categories)
    ):
        raise ValueError(
            "Apache pack categories must be unique non-empty strings"
        )
    labels = list(
        object_labels
    )
    if categories != labels:
        raise ValueError(
            "MegaSG pack category order does not match relation object vocabulary"
        )

    img_meta = np.load(
        img_meta_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    box_cats = np.load(
        box_cats_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    rels = np.load(
        rels_path,
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

    report = {
        "source_name": source_name,
        "pack_split": (
            f"{source_name}/train"
        ),
        "meta_sha256": sha256_file(
            meta_path
        ),
        "img_meta_sha256": (
            sha256_file(
                img_meta_path
            )
        ),
        "box_cats_sha256": (
            sha256_file(
                box_cats_path
            )
        ),
        "rels_sha256": sha256_file(
            rels_path
        ),
        "images": int(
            img_meta.shape[0]
        ),
        "boxes": int(
            box_cats.shape[0]
        ),
        "relations_rows": int(
            rels.shape[0]
        ),
        "object_label_order": labels,
        "object_label_order_sha256": (
            names_sha256(
                labels
            )
        ),
    }
    return (
        report,
        img_meta,
        box_cats,
        rels,
    )


def rebuild_pair_opportunity(
    pack_split: str | Path,
    object_labels: Sequence[str],
    *,
    scan_box_cap: int = RELEASED_SCAN_BOX_CAP,
    min_support: int = RELEASED_MIN_SUPPORT,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, object],
]:
    if scan_box_cap != RELEASED_SCAN_BOX_CAP:
        raise ValueError(
            "released pair-opportunity rebuild requires scan_box_cap=400"
        )
    if min_support != RELEASED_MIN_SUPPORT:
        raise ValueError(
            "released pair-opportunity rebuild requires min_support=50"
        )
    labels = tuple(
        object_labels
    )
    if not labels:
        raise ValueError(
            "object vocabulary must not be empty"
        )
    if (
        any(
            not isinstance(name, str)
            or not name
            for name in labels
        )
        or len(set(labels))
        != len(labels)
    ):
        raise ValueError(
            "object labels must be unique non-empty strings"
        )

    split_dir = Path(
        pack_split
    )
    (
        pack_report,
        img_meta,
        box_cats,
        rels,
    ) = _load_pack(
        split_dir,
        labels,
    )
    count = len(labels)
    opportunities = np.zeros(
        (count, count),
        dtype=np.int64,
    )
    relations = np.zeros(
        count * count,
        dtype=np.int64,
    )
    scanned_images = 0

    for row_index, row in enumerate(
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
        if (
            raw_box_start < 0
            or raw_box_count < 0
            or raw_box_start
            + raw_box_count
            > box_cats.shape[0]
        ):
            raise ValueError(
                f"image {row_index} has invalid box range"
            )
        if (
            raw_rel_start < 0
            or raw_rel_count < 0
            or raw_rel_start
            + raw_rel_count
            > rels.shape[0]
        ):
            raise ValueError(
                f"image {row_index} has invalid relation range"
            )

        box_count = min(
            raw_box_count,
            scan_box_cap,
        )
        if box_count < 2:
            continue
        cats_all = np.asarray(
            box_cats[
                raw_box_start :
                raw_box_start
                + box_count
            ],
            dtype=np.int64,
        )
        if (
            (cats_all < 0).any()
            or (cats_all >= count).any()
        ):
            raise ValueError(
                f"image {row_index} has category id outside object vocabulary"
            )

        if raw_rel_count:
            image_rels = np.asarray(
                rels[
                    raw_rel_start :
                    raw_rel_start
                    + raw_rel_count
                ],
                dtype=np.int64,
            )
            endpoints = image_rels[
                :, :2
            ]
            if (
                (endpoints < 0).any()
                or (
                    endpoints
                    >= raw_box_count
                ).any()
            ):
                raise ValueError(
                    f"image {row_index} has relation endpoint outside packed boxes"
                )
            keep = (
                (image_rels[:, 0] < box_count)
                & (
                    image_rels[:, 1]
                    < box_count
                )
            )
            image_rels = image_rels[
                keep
            ]
            if image_rels.size:
                subject_categories = (
                    cats_all[
                        image_rels[
                            :, 0
                        ]
                    ]
                )
                object_categories = (
                    cats_all[
                        image_rels[
                            :, 1
                        ]
                    ]
                )
                np.add.at(
                    relations,
                    subject_categories
                    * count
                    + object_categories,
                    1,
                )

        unique, frequency = np.unique(
            cats_all,
            return_counts=True,
        )
        block = np.outer(
            frequency,
            frequency,
        ).astype(
            np.int64,
            copy=False,
        )
        np.fill_diagonal(
            block,
            frequency
            * (
                frequency
                - 1
            ),
        )
        opportunities[
            np.ix_(
                unique,
                unique,
            )
        ] += block
        scanned_images += 1

    opportunities_flat = (
        opportunities.reshape(-1)
    )
    rate = np.zeros(
        count * count,
        dtype=np.float32,
    )
    observed = (
        opportunities_flat > 0
    )
    if observed.any():
        rate[observed] = (
            relations[observed]
            / opportunities_flat[
                observed
            ]
        ).astype(
            np.float32
        )
        np.minimum(
            rate,
            np.float32(1.0),
            out=rate,
        )

    arrays = {
        "rate": rate,
        "opportunities": (
            opportunities_flat
        ),
        "relations": relations,
        "num_cats": np.asarray(
            count,
            dtype=np.int32,
        ),
        "min_support": np.asarray(
            min_support,
            dtype=np.int32,
        ),
        "meta": np.asarray(
            [
                json.dumps(
                    {
                        "algorithm": (
                            DERIVATION_ALGORITHM
                        ),
                        "source": (
                            "megasg_clean/train"
                        ),
                        "scan_box_cap": (
                            scan_box_cap
                        ),
                        "min_support": (
                            min_support
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ],
            dtype=np.str_,
        ),
    }
    evidence = {
        "schema": DERIVATION_SCHEMA,
        "algorithm": (
            DERIVATION_ALGORITHM
        ),
        "source_name": (
            RELEASED_SOURCE
        ),
        "pack": pack_report,
        "scan_box_cap": (
            scan_box_cap
        ),
        "min_support": (
            min_support
        ),
        "ordered_instance_pairs": True,
        "diagonal_excludes_self_pairs": True,
        "same_pack_relation_numerator": True,
        "rate_clamp_max": 1.0,
        "num_cats": count,
        "images_scanned": (
            scanned_images
        ),
        "category_pairs_with_opportunity": int(
            observed.sum()
        ),
        "trusted_category_pairs": int(
            (
                opportunities_flat
                >= min_support
            ).sum()
        ),
        "total_opportunities": int(
            opportunities_flat.sum()
        ),
        "total_relations": int(
            relations.sum()
        ),
    }
    return arrays, evidence


def write_pair_opportunity(
    *,
    pack_split: str | Path,
    object_labels: Sequence[str],
    output: str | Path,
    evidence_output: str | Path,
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

    arrays, evidence = (
        rebuild_pair_opportunity(
            pack_split,
            object_labels,
        )
    )
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.savez_compressed(
        out,
        **arrays,
    )
    # Reuse the runtime loader as the final format validator.
    loaded = PairOpportunityTable.load(
        out
    )
    evidence = {
        **evidence,
        "output_sha256": (
            sha256_file(out)
        ),
        "runtime_stats": (
            loaded.stats()
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
            "relation vocabulary must contain objects"
        )

    write_pair_opportunity(
        pack_split=args.pack_split,
        object_labels=payload[
            "objects"
        ],
        output=args.out,
        evidence_output=args.evidence,
    )


if __name__ == "__main__":
    main()
