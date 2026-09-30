from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from benchmark import DatasetManifest, RelationVocabulary


EVIDENCE_SCHEMA = "kfcore.apache-pack-materialization/1"
REQUIRED_PACK_FILES = (
    "meta.json",
    "file_names.json",
    "img_meta.npy",
    "boxes.npy",
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


def stable_json_line(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _require_string_list(
    value: object,
    name: str,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(
            f"{name} must be an array"
        )
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or item in seen
        ):
            raise ValueError(
                f"{name} must contain unique non-empty strings"
            )
        seen.add(item)
        result.append(item)
    return result


def _load_pack(
    pack_dir: Path,
) -> tuple[
    dict[str, Any],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    for name in REQUIRED_PACK_FILES:
        if not (pack_dir / name).is_file():
            raise FileNotFoundError(
                pack_dir / name
            )

    meta = json.loads(
        (pack_dir / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(meta, dict):
        raise ValueError(
            "pack meta.json must be an object"
        )
    file_names = json.loads(
        (pack_dir / "file_names.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(file_names, list)
        or any(
            not isinstance(name, str) or not name
            for name in file_names
        )
    ):
        raise ValueError(
            "pack file_names.json must be non-empty strings"
        )

    img_meta = np.load(
        pack_dir / "img_meta.npy",
        allow_pickle=False,
    )
    boxes = np.load(
        pack_dir / "boxes.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    box_cats = np.load(
        pack_dir / "box_cats.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    rels = np.load(
        pack_dir / "rels.npy",
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
        boxes.ndim != 2
        or boxes.shape[1] != 4
        or not np.issubdtype(
            boxes.dtype,
            np.floating,
        )
    ):
        raise ValueError(
            "boxes.npy must be floating [B,4]"
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
    if len(file_names) != img_meta.shape[0]:
        raise ValueError(
            "file_names/img_meta image count mismatch"
        )
    if boxes.shape[0] != box_cats.shape[0]:
        raise ValueError(
            "boxes/box_cats row count mismatch"
        )

    for key, actual in (
        ("num_images", img_meta.shape[0]),
        ("num_boxes", boxes.shape[0]),
        ("num_rels", rels.shape[0]),
    ):
        if key in meta and meta[key] != actual:
            raise ValueError(
                f"pack meta {key} does not match arrays"
            )

    return (
        meta,
        file_names,
        img_meta,
        boxes,
        box_cats,
        rels,
    )


def _box_to_pixel_xyxy(
    box: np.ndarray,
    width: int,
    height: int,
    *,
    atol: float = 1.0e-6,
) -> list[float]:
    values = np.asarray(
        box,
        dtype=np.float32,
    )
    if (
        values.shape != (4,)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "packed box must be finite cxcywh"
        )
    cx, cy, bw, bh = (
        float(value)
        for value in values.tolist()
    )
    if (
        cx < 0.0
        or cx > 1.0
        or cy < 0.0
        or cy > 1.0
        or bw <= 0.0
        or bh <= 0.0
        or bw > 1.0
        or bh > 1.0
    ):
        raise ValueError(
            "packed box is degenerate or outside normalized bounds"
        )

    left = (cx - 0.5 * bw) * width
    right = (cx + 0.5 * bw) * width
    top = (cy - 0.5 * bh) * height
    bottom = (cy + 0.5 * bh) * height
    if (
        left < 0.0
        or top < 0.0
        or right > float(width)
        or bottom > float(height)
        or right <= left
        or bottom <= top
    ):
        raise ValueError(
            "packed box cannot be represented by canonical in-image xyxy"
        )

    roundtrip = np.asarray(
        (
            ((left + right) * 0.5) / width,
            ((top + bottom) * 0.5) / height,
            (right - left) / width,
            (bottom - top) / height,
        ),
        dtype=np.float32,
    )
    if not np.allclose(
        roundtrip,
        values,
        rtol=0.0,
        atol=atol,
    ):
        raise ValueError(
            "packed box does not round-trip through canonical xyxy"
        )

    return [
        float(left),
        float(top),
        float(right),
        float(bottom),
    ]


def materialize_pack(
    pack_dir: str | Path,
    vocabulary: RelationVocabulary,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(pack_dir)
    out = Path(output_path)
    if out.exists():
        raise FileExistsError(out)
    if not vocabulary.object_labels:
        raise ValueError(
            "Apache pack materialization requires an object vocabulary"
        )

    (
        meta,
        file_names,
        img_meta,
        boxes,
        box_cats,
        rels,
    ) = _load_pack(root)

    local_predicates = _require_string_list(
        meta.get("predicates"),
        "pack predicates",
    )
    local_categories = _require_string_list(
        meta.get("categories"),
        "pack categories",
    )

    predicate_index = {
        name: index
        for index, name in enumerate(
            vocabulary.predicates
        )
    }
    category_set = set(
        vocabulary.object_labels
    )
    missing_predicates = [
        name
        for name in local_predicates
        if name not in predicate_index
    ]
    if missing_predicates:
        raise ValueError(
            "pack predicate is absent from union vocabulary: "
            + missing_predicates[0]
        )
    missing_categories = [
        name
        for name in local_categories
        if name not in category_set
    ]
    if missing_categories:
        raise ValueError(
            "pack category is absent from union vocabulary: "
            + missing_categories[0]
        )

    output_lines: list[str] = []
    relation_count = 0
    max_boxes = 0
    seen_images: set[str] = set()

    for row_index, row in enumerate(img_meta):
        (
            _image_id,
            raw_width,
            raw_height,
            raw_box_start,
            raw_box_count,
            raw_rel_start,
            raw_rel_count,
        ) = (
            int(value)
            for value in row.tolist()
        )
        width = raw_width
        height = raw_height
        box_start = raw_box_start
        box_count = raw_box_count
        rel_start = raw_rel_start
        rel_count = raw_rel_count

        if width <= 0 or height <= 0:
            raise ValueError(
                f"pack image {row_index} has invalid dimensions"
            )
        if (
            box_start < 0
            or box_count < 2
            or box_start + box_count > boxes.shape[0]
        ):
            raise ValueError(
                f"pack image {row_index} has invalid box range"
            )
        if (
            rel_start < 0
            or rel_count < 0
            or rel_start + rel_count > rels.shape[0]
        ):
            raise ValueError(
                f"pack image {row_index} has invalid relation range"
            )

        image = file_names[row_index].replace(
            "\\",
            "/",
        )
        if image in seen_images:
            raise ValueError(
                f"duplicate pack image path: {image}"
            )
        seen_images.add(image)

        image_boxes = boxes[
            box_start :
            box_start + box_count
        ]
        image_cats = box_cats[
            box_start :
            box_start + box_count
        ]
        boxes_xyxy = [
            _box_to_pixel_xyxy(
                image_boxes[index],
                width,
                height,
            )
            for index in range(box_count)
        ]

        object_labels: list[str] = []
        for value in image_cats.tolist():
            category_index = int(value)
            if (
                category_index < 0
                or category_index >= len(
                    local_categories
                )
            ):
                raise ValueError(
                    f"pack image {row_index} has invalid category index"
                )
            object_labels.append(
                local_categories[category_index]
            )

        canonical_relations: list[list[int]] = []
        seen_relations: set[
            tuple[int, int, int]
        ] = set()
        image_rels = rels[
            rel_start :
            rel_start + rel_count
        ]
        for relation in image_rels:
            subject = int(relation[0])
            object_ = int(relation[1])
            local_predicate = int(
                relation[2]
            )
            if (
                subject < 0
                or subject >= box_count
                or object_ < 0
                or object_ >= box_count
                or subject == object_
            ):
                raise ValueError(
                    f"pack image {row_index} has invalid relation endpoints"
                )
            if (
                local_predicate < 0
                or local_predicate >= len(
                    local_predicates
                )
            ):
                raise ValueError(
                    f"pack image {row_index} has invalid predicate index"
                )
            predicate = predicate_index[
                local_predicates[
                    local_predicate
                ]
            ]
            key = (
                subject,
                predicate,
                object_,
            )
            if key in seen_relations:
                raise ValueError(
                    f"pack image {row_index} has duplicate canonical relation"
                )
            seen_relations.add(key)
            canonical_relations.append(
                [
                    subject,
                    predicate,
                    object_,
                ]
            )

        payload = {
            "image": image,
            "width": width,
            "height": height,
            "boxes_xyxy": boxes_xyxy,
            "object_labels": object_labels,
            "relations": canonical_relations,
        }
        output_lines.append(
            stable_json_line(payload)
        )
        relation_count += len(
            canonical_relations
        )
        max_boxes = max(
            max_boxes,
            box_count,
        )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out.write_text(
        "".join(output_lines),
        encoding="utf-8",
    )

    manifest = DatasetManifest.load(
        out,
        vocabulary,
    )
    if len(manifest.examples) != img_meta.shape[0]:
        raise RuntimeError(
            "materialized manifest image count mismatch"
        )

    component_sha256 = {
        name: sha256_file(root / name)
        for name in REQUIRED_PACK_FILES
    }
    return {
        "schema": EVIDENCE_SCHEMA,
        "pack_dir": str(root.resolve()),
        "dataset": meta.get("dataset"),
        "split": meta.get("split"),
        "component_sha256": component_sha256,
        "canonical_annotations_sha256": (
            manifest.annotations_sha256
        ),
        "vocabulary_sha256": (
            manifest.vocabulary_sha256
        ),
        "images": len(
            manifest.examples
        ),
        "relations": relation_count,
        "max_boxes": max_boxes,
        "local_predicate_count": len(
            local_predicates
        ),
        "local_category_count": len(
            local_categories
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", required=True)
    parser.add_argument(
        "--vocabulary",
        required=True,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--evidence",
        required=True,
    )
    args = parser.parse_args()

    vocabulary = RelationVocabulary.load(
        args.vocabulary
    )
    evidence = materialize_pack(
        args.pack,
        vocabulary,
        args.out,
    )
    evidence_path = Path(args.evidence)
    if evidence_path.exists():
        raise FileExistsError(
            evidence_path
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
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
