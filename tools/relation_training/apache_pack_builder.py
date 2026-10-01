from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


EVIDENCE_SCHEMA = "kfcore.apache-coco-sgg-pack-rebuild/1"
RELEASED_SPLIT = "train"
RELEASED_MAX_OBJECTS = 40
RELEASED_MIN_RELS = 1
FLAG_SPATIAL = 1
FLAG_GEOMETRIC = 2
ROUND_SHIFT = 2

PACK_COMPONENTS = (
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
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_strings_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _require_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_unique_id_rows(
    rows: object,
    name: str,
    *,
    require_name: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{name} must be an array")
    result: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{name}[{index}] must be an object")
        row_id = _require_int(row.get("id"), f"{name}[{index}].id")
        if row_id in seen_ids:
            raise ValueError(f"{name} contains duplicate id {row_id}")
        seen_ids.add(row_id)
        if require_name:
            _require_non_empty_string(
                row.get("name"),
                f"{name}[{index}].name",
            )
        result.append(row)
    return result


def _stable_file_names_bytes(file_names: Sequence[str]) -> bytes:
    # Upstream uses json.dump(list, file) with default separators and no newline.
    return json.dumps(list(file_names)).encode("utf-8")


def _stable_meta_bytes(meta: dict[str, object]) -> bytes:
    # Upstream uses json.dump(meta, file, indent=2) with insertion order.
    return json.dumps(meta, indent=2).encode("utf-8")


def _write_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)


def pack_coco_sgg_train(
    annotations_path: str | Path,
    output_dir: str | Path,
    *,
    dataset_name: str,
    ann_source_label: str,
    img_dir_label: str,
    exclude_file_names: Iterable[str] = (),
) -> dict[str, object]:
    """Pack a released-style COCO-SGG train source into Apache memmaps.

    The released train contract is fixed at max_objects=40 and min_rels=1.
    Logical path labels are explicit so deterministic rebuild bytes do not
    depend on a machine's absolute filesystem layout.
    """
    source = Path(annotations_path)
    out = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if out.exists():
        raise FileExistsError(out)

    dataset_name = _require_non_empty_string(dataset_name, "dataset_name")
    ann_source_label = _require_non_empty_string(
        ann_source_label,
        "ann_source_label",
    )
    img_dir_label = _require_non_empty_string(img_dir_label, "img_dir_label")

    excluded = set()
    for value in exclude_file_names:
        excluded.add(_require_non_empty_string(value, "exclude_file_name"))

    raw_bytes = source.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("COCO-SGG input must be a JSON object")

    images_raw = payload.get("images")
    annotations_raw = payload.get("annotations")
    relations_raw = payload.get("rel_annotations")
    if not isinstance(images_raw, list):
        raise ValueError("COCO-SGG images must be an array")
    if not isinstance(annotations_raw, list):
        raise ValueError("COCO-SGG annotations must be an array")
    if not isinstance(relations_raw, list):
        raise ValueError("COCO-SGG rel_annotations must be an array")

    rel_categories = _require_unique_id_rows(
        payload.get("rel_categories"),
        "rel_categories",
        require_name=True,
    )
    categories = _require_unique_id_rows(
        payload.get("categories"),
        "categories",
        require_name=True,
    )
    if not rel_categories:
        raise ValueError("rel_categories must not be empty")
    if not categories:
        raise ValueError("categories must not be empty")

    predicates = [str(row["name"]) for row in rel_categories]
    category_names = [str(row["name"]) for row in categories]
    if len(set(predicates)) != len(predicates):
        raise ValueError("rel_categories names must be unique")
    if len(set(category_names)) != len(category_names):
        raise ValueError("categories names must be unique")

    rel_id_to_idx = {
        int(row["id"]): index
        for index, row in enumerate(rel_categories)
    }
    cat_id_to_idx = {
        int(row["id"]): index
        for index, row in enumerate(categories)
    }

    anns_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    annotation_ids: set[int] = set()
    for index, annotation in enumerate(annotations_raw):
        if not isinstance(annotation, dict):
            raise ValueError(f"annotations[{index}] must be an object")
        ann_id = _require_int(annotation.get("id"), f"annotations[{index}].id")
        image_id = _require_int(
            annotation.get("image_id"),
            f"annotations[{index}].image_id",
        )
        category_id = _require_int(
            annotation.get("category_id"),
            f"annotations[{index}].category_id",
        )
        if ann_id in annotation_ids:
            raise ValueError(f"duplicate annotation id {ann_id}")
        annotation_ids.add(ann_id)
        if category_id not in cat_id_to_idx:
            raise ValueError(
                f"annotation category id {category_id} is absent from categories"
            )
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"annotations[{index}].bbox must contain four values")
        for dim, value in enumerate(bbox):
            _require_number(value, f"annotations[{index}].bbox[{dim}]")
        anns_by_img[image_id].append(annotation)

    rels_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, relation in enumerate(relations_raw):
        if not isinstance(relation, dict):
            raise ValueError(f"rel_annotations[{index}] must be an object")
        image_id = _require_int(
            relation.get("image_id"),
            f"rel_annotations[{index}].image_id",
        )
        _require_int(
            relation.get("subject_id"),
            f"rel_annotations[{index}].subject_id",
        )
        _require_int(
            relation.get("object_id"),
            f"rel_annotations[{index}].object_id",
        )
        predicate_id = _require_int(
            relation.get("predicate_id"),
            f"rel_annotations[{index}].predicate_id",
        )
        if predicate_id not in rel_id_to_idx:
            raise ValueError(
                f"relation predicate id {predicate_id} is absent from rel_categories"
            )
        round_value = relation.get("round", 0)
        if isinstance(round_value, bool) or not isinstance(round_value, int):
            raise ValueError("relation round must be an integer")
        if round_value < 0:
            raise ValueError("relation round must be non-negative")
        raw_predicate = relation.get("predicate_raw")
        if raw_predicate is not None and raw_predicate != "":
            _require_non_empty_string(raw_predicate, "predicate_raw")
        rels_by_img[image_id].append(relation)

    images: list[dict[str, Any]] = []
    seen_image_ids: set[int] = set()
    excluded_image_count = 0
    for index, image in enumerate(images_raw):
        if not isinstance(image, dict):
            raise ValueError(f"images[{index}] must be an object")
        image_id = _require_int(image.get("id"), f"images[{index}].id")
        file_name = _require_non_empty_string(
            image.get("file_name"),
            f"images[{index}].file_name",
        )
        if file_name in excluded:
            excluded_image_count += 1
            continue
        if image_id in seen_image_ids:
            raise ValueError(f"duplicate image id {image_id}")
        seen_image_ids.add(image_id)
        width = _require_number(image.get("width"), f"images[{index}].width")
        height = _require_number(image.get("height"), f"images[{index}].height")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"image {image_id} has invalid dimensions")
        if not float(width).is_integer() or not float(height).is_integer():
            raise ValueError(f"image {image_id} dimensions must be integers")
        images.append(image)

    img_meta: list[tuple[int, int, int, int, int, int, int]] = []
    file_names: list[str] = []
    boxes_out: list[np.ndarray] = []
    cats_out: list[int] = []
    rels_out: list[tuple[int, int, int, int, int]] = []
    raw_vocab: dict[str, int] = {}
    raw_links: Counter[tuple[str, str]] = Counter()
    pred_counts: Counter[int] = Counter()

    drop_img_fewbox = 0
    drop_img_norel = 0
    drop_rel_range = 0
    drop_rel_self = 0
    degenerate_boxes = 0

    for image in images:
        image_id = int(image["id"])
        annotations = anns_by_img.get(image_id, [])[:RELEASED_MAX_OBJECTS]
        if len(annotations) < 2:
            drop_img_fewbox += 1
            continue

        gid_to_local = {
            int(annotation["id"]): index
            for index, annotation in enumerate(annotations)
        }
        width = int(image["width"])
        height = int(image["height"])

        image_rels: list[tuple[int, int, int, int, int]] = []
        for relation in rels_by_img.get(image_id, []):
            subject = gid_to_local.get(int(relation["subject_id"]))
            object_ = gid_to_local.get(int(relation["object_id"]))
            if subject is None or object_ is None:
                drop_rel_range += 1
                continue
            if subject == object_:
                drop_rel_self += 1
                continue

            predicate = rel_id_to_idx[int(relation["predicate_id"])]
            flags = 0
            if bool(relation.get("spatial")):
                flags |= FLAG_SPATIAL
            if relation.get("source") == "geometric":
                flags |= FLAG_GEOMETRIC
            flags |= int(relation.get("round", 0)) << ROUND_SHIFT

            raw = relation.get("predicate_raw")
            if raw:
                assert isinstance(raw, str)
                raw_id = raw_vocab.setdefault(raw, len(raw_vocab))
                raw_links[(raw, predicates[predicate])] += 1
            else:
                raw_id = -1

            image_rels.append(
                (subject, object_, predicate, flags, raw_id)
            )
            pred_counts[predicate] += 1

        if len(image_rels) < RELEASED_MIN_RELS:
            drop_img_norel += 1
            continue

        image_boxes = np.empty((len(annotations), 4), dtype=np.float32)
        for local_index, annotation in enumerate(annotations):
            x_raw, y_raw, w_raw, h_raw = annotation["bbox"]
            x = float(x_raw)
            y = float(y_raw)
            width_box = abs(float(w_raw))
            height_box = abs(float(h_raw))
            if width_box <= 1.0 or height_box <= 1.0:
                degenerate_boxes += 1
            cx = np.clip(
                (x + width_box * 0.5) / float(width),
                0.0,
                1.0,
            )
            cy = np.clip(
                (y + height_box * 0.5) / float(height),
                0.0,
                1.0,
            )
            image_boxes[local_index] = (
                cx,
                cy,
                min(width_box / float(width), 1.0),
                min(height_box / float(height), 1.0),
            )
            cats_out.append(cat_id_to_idx[int(annotation["category_id"])])

        img_meta.append(
            (
                image_id,
                width,
                height,
                len(boxes_out),
                len(annotations),
                len(rels_out),
                len(image_rels),
            )
        )
        file_names.append(str(image["file_name"]))
        boxes_out.extend(image_boxes)
        rels_out.extend(image_rels)

    img_meta_array = np.asarray(img_meta, dtype=np.int64).reshape(-1, 7)
    boxes_array = np.asarray(boxes_out, dtype=np.float32).reshape(-1, 4)
    cats_array = np.asarray(cats_out, dtype=np.int32).reshape(-1)
    rels_array = np.asarray(rels_out, dtype=np.int32).reshape(-1, 5)

    raw_predicates: list[str | None] = [None] * len(raw_vocab)
    for value, index in raw_vocab.items():
        raw_predicates[index] = value

    predicate_counts = {
        predicates[predicate]: count
        for predicate, count in pred_counts.most_common()
    }
    raw_link_rows = [
        {
            "raw": raw,
            "predicate": predicate,
            "count": count,
        }
        for (raw, predicate), count in raw_links.most_common()
    ]

    meta: dict[str, object] = {
        "dataset": dataset_name,
        "split": RELEASED_SPLIT,
        "ann_source": ann_source_label,
        "img_dir": img_dir_label,
        "max_objects": RELEASED_MAX_OBJECTS,
        "min_rels": RELEASED_MIN_RELS,
        "num_images": len(img_meta),
        "num_boxes": len(boxes_out),
        "num_rels": len(rels_out),
        "predicates": predicates,
        "predicate_counts": predicate_counts,
        "categories": category_names,
        "category_ids": [int(row["id"]) for row in categories],
        "raw_predicates": raw_predicates,
        "raw_links": raw_link_rows,
        "flags_legend": {
            "bit0": "spatial",
            "bit1": "source==geometric",
            "bits2+": "round",
        },
        "drops": {
            "images_fewer_than_2_boxes": drop_img_fewbox,
            "images_below_min_rels": drop_img_norel,
            "rels_beyond_max_objects": drop_rel_range,
            "rels_self_loop": drop_rel_self,
            "degenerate_boxes_kept": degenerate_boxes,
        },
    }

    out.mkdir(parents=True, exist_ok=False)
    (out / "file_names.json").write_bytes(_stable_file_names_bytes(file_names))
    (out / "meta.json").write_bytes(_stable_meta_bytes(meta))
    _write_npy(out / "img_meta.npy", img_meta_array)
    _write_npy(out / "boxes.npy", boxes_array)
    _write_npy(out / "box_cats.npy", cats_array)
    _write_npy(out / "rels.npy", rels_array)

    component_sha256 = {
        name: sha256_file(out / name)
        for name in PACK_COMPONENTS
    }
    excluded_sorted = sorted(excluded)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "dataset": dataset_name,
        "split": RELEASED_SPLIT,
        "input_annotations_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "ann_source_label": ann_source_label,
        "img_dir_label": img_dir_label,
        "max_objects": RELEASED_MAX_OBJECTS,
        "min_rels": RELEASED_MIN_RELS,
        "exclude_file_names": excluded_sorted,
        "exclude_file_name_count": len(excluded_sorted),
        "exclude_file_names_sha256": ordered_strings_sha256(excluded_sorted),
        "excluded_image_count": excluded_image_count,
        "input_image_count": len(images_raw),
        "packed_image_count": int(img_meta_array.shape[0]),
        "packed_box_count": int(boxes_array.shape[0]),
        "packed_relation_count": int(rels_array.shape[0]),
        "predicate_count": len(predicates),
        "category_count": len(category_names),
        "raw_predicate_count": len(raw_vocab),
        "drop_counts": dict(meta["drops"]),
        "component_sha256": component_sha256,
    }
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--ann-source-label", required=True)
    parser.add_argument("--img-dir-label", required=True)
    parser.add_argument(
        "--exclude-file-names",
        default="",
        help="Optional JSON array of exact file names to exclude before packing.",
    )
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()

    excluded: Sequence[str] = ()
    if args.exclude_file_names:
        raw = json.loads(Path(args.exclude_file_names).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("exclude-file-names JSON must be an array")
        excluded = raw

    evidence = pack_coco_sgg_train(
        args.annotations,
        args.out,
        dataset_name=args.dataset_name,
        ann_source_label=args.ann_source_label,
        img_dir_label=args.img_dir_label,
        exclude_file_names=excluded,
    )
    evidence_path = Path(args.evidence)
    if evidence_path.exists():
        raise FileExistsError(evidence_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
