from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from apache_hico_train_rebuild import merge_boxes
from benchmark import RelationVocabulary
from prepare_released_relsgg import PREDICATE_BANK_SHA256, sha256_file


DATASET_ID = "zhimeng/hico_det"
DATASET_REVISION = "4d65d61c99931737c2b7a9636c7e3501eab69850"
DATASET_LICENSE = "MIT"
REFERENCE_SOURCE_COMMIT = "4a07de9d06f2e3f14309753b7907cf1d3a263b08"
IOU_MERGE = 0.5
REPORT_SCHEMA = "kfcore.hico-test-qualification-slice/1"


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_array(value: object, name: str) -> list[Any]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, list):
        raise ValueError(f"HICO {name} must be an array")
    return value


def _load_actions(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 600:
        raise ValueError("HICO list_action.csv must contain exactly 600 rows")
    required = {"nname", "vname", "vname_ing"}
    for index, row in enumerate(rows):
        if not required.issubset(row):
            raise ValueError(f"HICO action row {index} is missing columns")
    return rows


def _load_bank_names(path: str | Path) -> set[str]:
    source = Path(path)
    if sha256_file(source) != PREDICATE_BANK_SHA256:
        raise ValueError("released predicate bank SHA-256 differs")
    with np.load(source, allow_pickle=True) as bank:
        if "names" not in bank.files:
            raise ValueError("released predicate bank lacks names")
        names = [str(value) for value in bank["names"]]
    if len(set(names)) != len(names):
        raise ValueError("released predicate bank contains duplicate names")
    return set(names)


def _image_bytes(value: object) -> tuple[bytes, str]:
    if isinstance(value, Image.Image):
        buffer = io.BytesIO()
        value.convert("RGB").save(buffer, format="JPEG", quality=95)
        return buffer.getvalue(), ""
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            path = value.get("path")
            return bytes(raw), Path(str(path)).name if path else ""
    raise ValueError("HICO image payload is unsupported")


def _decode_row(
    row: dict[str, Any],
    actions: list[dict[str, str]],
    allowed_predicates: set[str],
) -> tuple[
    int,
    int,
    bytes,
    list[tuple[float, float, float, float]],
    list[str],
    list[tuple[int, str, int]],
] | None:
    objects = _parse_array(row.get("objects"), "objects")
    image_raw, _ = _image_bytes(row.get("image"))
    with Image.open(io.BytesIO(image_raw)) as image:
        width, height = image.size

    local_boxes: list[tuple[list[float], str]] = []
    box_index: dict[tuple[int, int, int, int, str], int] = {}
    relations: list[tuple[int, int, str]] = []

    def add_box(raw_box: object, category: str) -> int:
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            raise ValueError("HICO box must be [x1,x2,y1,y2]")
        x1, x2, y1, y2 = [float(value) for value in raw_box]
        if not all(np.isfinite([x1, x2, y1, y2])):
            raise ValueError("HICO box coordinate must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("HICO box has non-positive extent")
        x1 = min(max(x1, 0.0), float(width))
        x2 = min(max(x2, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        y2 = min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("HICO clipped box has non-positive extent")
        key = (round(x1), round(x2), round(y1), round(y2), category)
        if key in box_index:
            return box_index[key]
        index = len(local_boxes)
        box_index[key] = index
        local_boxes.append(([x1, y1, x2 - x1, y2 - y1], category))
        return index

    for entry in objects:
        if not isinstance(entry, dict):
            raise ValueError("HICO object entry must be an object")
        hoi_id = entry.get("id")
        if (
            isinstance(hoi_id, bool)
            or not isinstance(hoi_id, int)
            or not 1 <= hoi_id <= 600
            or entry.get("invis")
        ):
            continue
        action = actions[hoi_id - 1]
        if action["vname"] == "no_interaction":
            continue
        predicate = action["vname_ing"].replace("_", " ")
        if predicate not in allowed_predicates:
            continue
        subject = add_box(entry.get("bbox_human"), "person")
        object_name = action["nname"]
        object_ = add_box(entry.get("bbox_object"), object_name)
        relations.append((subject, object_, predicate))

    if not relations:
        return None

    if len(local_boxes) > 1:
        merged, remap = merge_boxes(local_boxes, IOU_MERGE)
    else:
        merged = local_boxes
        remap = list(range(len(local_boxes)))

    deduped: list[tuple[int, str, int]] = []
    seen: set[tuple[int, str, int]] = set()
    for subject, object_, predicate in relations:
        value = (remap[subject], predicate, remap[object_])
        if value[0] == value[2] or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    if not deduped or len(merged) > 32:
        return None

    boxes = [
        (
            float(box[0]),
            float(box[1]),
            float(box[0] + box[2]),
            float(box[1] + box[3]),
        )
        for box, _ in merged
    ]
    labels = [category for _, category in merged]
    return width, height, image_raw, boxes, labels, deduped


def materialize(
    *,
    actions_path: str | Path,
    predicate_bank_path: str | Path,
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    max_images: int,
    dataset_revision: str = DATASET_REVISION,
) -> dict[str, object]:
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    actions = _load_actions(actions_path)
    allowed = _load_bank_names(predicate_bank_path)

    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(root)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=False)

    selected: list[
        tuple[
            int,
            int,
            int,
            bytes,
            list[tuple[float, float, float, float]],
            list[str],
            list[tuple[int, str, int]],
        ]
    ] = []
    scanned = 0
    for row_index, row in enumerate(rows):
        scanned += 1
        decoded = _decode_row(row, actions, allowed)
        if decoded is None:
            continue
        width, height, raw, boxes, labels, relations = decoded
        selected.append(
            (
                row_index,
                width,
                height,
                raw,
                boxes,
                labels,
                relations,
            )
        )
        if len(selected) == max_images:
            break

    if len(selected) != max_images:
        raise ValueError(
            f"HICO stream ended before {max_images} qualifying rows"
        )

    predicates = sorted(
        {
            predicate
            for _, _, _, _, _, _, relations in selected
            for _, predicate, _ in relations
        }
    )
    vocabulary = RelationVocabulary(tuple(predicates))
    predicate_index = {
        name: index for index, name in enumerate(vocabulary.predicates)
    }

    annotation_lines: list[bytes] = []
    source_rows: list[dict[str, object]] = []
    relation_count = 0
    box_count = 0
    for output_index, (
        row_index,
        width,
        height,
        raw,
        boxes,
        labels,
        relations,
    ) in enumerate(selected):
        image_name = f"hico_test_{row_index:06d}.jpg"
        image_path = images_dir / image_name
        with Image.open(io.BytesIO(raw)) as image:
            image.convert("RGB").save(
                image_path,
                format="JPEG",
                quality=95,
            )
        payload = {
            "image": image_name,
            "width": width,
            "height": height,
            "boxes_xyxy": [list(box) for box in boxes],
            "object_labels": labels,
            "relations": [
                [
                    subject,
                    predicate_index[predicate],
                    object_,
                ]
                for subject, predicate, object_ in relations
            ],
            "source_id": 0,
        }
        encoded = stable_json_bytes(payload)
        annotation_lines.append(encoded)
        relation_count += len(relations)
        box_count += len(boxes)
        source_rows.append(
            {
                "stream_row_index": row_index,
                "image": image_name,
                "image_sha256": sha256_file(image_path),
                "boxes": len(boxes),
                "relations": len(relations),
            }
        )

    vocabulary_path = root / "vocabulary.json"
    vocabulary_path.write_bytes(
        stable_json_bytes(vocabulary.canonical_payload())
    )
    annotations_path = root / "test.jsonl"
    annotations_path.write_bytes(b"".join(annotation_lines))

    report = {
        "schema": REPORT_SCHEMA,
        "dataset": {
            "id": DATASET_ID,
            "revision": dataset_revision,
            "split": "test",
            "license": DATASET_LICENSE,
            "reference_source_commit": REFERENCE_SOURCE_COMMIT,
            "iou_merge": IOU_MERGE,
        },
        "selection": {
            "strategy": (
                "first-stream-rows-with-visible-positive-relation-"
                "in-released-predicate-bank-and-max32-boxes"
            ),
            "stream_rows_scanned": scanned,
            "images": len(selected),
            "boxes": box_count,
            "relations": relation_count,
            "predicate_count": len(predicates),
            "predicates": predicates,
            "source_rows": source_rows,
        },
        "artifacts": {
            "list_action_sha256": sha256_file(Path(actions_path)),
            "predicate_bank_sha256": PREDICATE_BANK_SHA256,
            "vocabulary_sha256": vocabulary.sha256(),
            "annotations_sha256": sha256_file(annotations_path),
        },
    }
    (root / "evidence.json").write_bytes(stable_json_bytes(report))
    return report


def stream_hico_test(
    *,
    revision: str,
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("HICO streaming requires huggingface datasets") from error
    return load_dataset(
        DATASET_ID,
        split="test",
        revision=revision,
        streaming=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-action", required=True)
    parser.add_argument("--predicate-bank", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--revision", default=DATASET_REVISION)
    args = parser.parse_args()

    report = materialize(
        actions_path=args.list_action,
        predicate_bank_path=args.predicate_bank,
        rows=stream_hico_test(revision=args.revision),
        output_dir=args.out_dir,
        max_images=args.max_images,
        dataset_revision=args.revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
