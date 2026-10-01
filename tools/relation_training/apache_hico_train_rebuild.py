from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from apache_pack_builder import PACK_COMPONENTS, pack_coco_sgg_train
from apache_pack_materializer import materialize_pack
from benchmark import RelationVocabulary


EVIDENCE_SCHEMA = "kfcore.apache-hico-train-rebuild/1"
REFERENCE_SOURCE_COMMIT = "4a07de9d06f2e3f14309753b7907cf1d3a263b08"
RELEASED_SPLIT = "train"
RELEASED_IOU_MERGE = 0.5
RELEASED_DATASET_NAME = "hicodet"
RELEASED_ANN_SOURCE_LABEL = "HICO_DET/hicodet_train_coco.json"
RELEASED_IMG_DIR_LABEL = "HICO_DET/train_images"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _upstream_json_bytes(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be finite")
    return result


def _iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    iw = max(0.0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    ih = max(0.0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1.0e-9)


def merge_boxes(
    local_boxes: Sequence[tuple[list[float], str]],
    threshold: float,
) -> tuple[list[tuple[list[float], str]], list[int]]:
    n = len(local_boxes)
    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(n):
        for j in range(i + 1, n):
            if (
                local_boxes[i][1] == local_boxes[j][1]
                and _iou_xywh(local_boxes[i][0], local_boxes[j][0]) >= threshold
            ):
                parent[find(j)] = find(i)

    groups: dict[int, int] = {}
    remap = [0] * n
    merged: list[list[Any]] = []
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = len(merged)
            merged.append([[0.0, 0.0, 0.0, 0.0], local_boxes[i][1], 0])
        group = groups[root]
        remap[i] = group
        for dim in range(4):
            merged[group][0][dim] += local_boxes[i][0][dim]
        merged[group][2] += 1

    output = [
        ([value / count for value in box], category)
        for box, category, count in merged
    ]
    return output, remap


def load_actions_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"nname", "vname", "vname_ing"}
    if len(rows) != 600:
        raise ValueError("released HICO action table must contain exactly 600 rows")
    for index, row in enumerate(rows):
        if not required.issubset(row):
            raise ValueError(f"list_action row {index} is missing required columns")
        if any(not row[name] for name in required):
            raise ValueError(f"list_action row {index} has an empty required value")
    return rows


def _parse_literal_columns(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for name in (
        "objects",
        "positive_captions",
        "negative_captions",
        "ambiguous_captions",
    ):
        value = result.get(name)
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if not isinstance(value, list):
            raise ValueError(f"HICO row {name} must be an array")
        result[name] = value
    return result


def iter_train_parquet_rows(
    parquet_dir: str | Path,
) -> tuple[Iterator[dict[str, Any]], list[dict[str, object]]]:
    root = Path(parquet_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)

    shards = sorted(root.glob("train-*.parquet"))
    if not shards:
        raise ValueError("no train-*.parquet shards found")

    reports: list[dict[str, object]] = []

    def iterator() -> Iterator[dict[str, Any]]:
        try:
            import pyarrow
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("real HICO rebuild requires pyarrow") from exc

        for shard in shards:
            parquet = pq.ParquetFile(shard)
            row_count = 0
            for row_group in range(parquet.num_row_groups):
                batch = parquet.read_row_group(row_group).to_pydict()
                columns = list(batch)
                if "image" not in batch:
                    raise ValueError(f"{shard.name} is missing image column")
                rows = len(batch["image"])
                for row_index in range(rows):
                    row_count += 1
                    yield {
                        column: batch[column][row_index]
                        for column in columns
                    }
            reports.append(
                {
                    "name": shard.name,
                    "sha256": sha256_file(shard),
                    "row_groups": parquet.num_row_groups,
                    "rows": row_count,
                    "pyarrow_version": pyarrow.__version__,
                }
            )

    return iterator(), reports


def _decode_image(
    image_payload: object,
    *,
    fallback_image_id: int,
    image_dir: Path | None,
) -> tuple[int, int, str]:
    if not isinstance(image_payload, dict):
        raise ValueError("HICO image payload must be an object")
    raw = image_payload.get("bytes")
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("HICO image bytes are required")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("HICO rebuild requires Pillow") from exc

    with Image.open(io.BytesIO(bytes(raw))) as image:
        width, height = image.size
        source_path = image_payload.get("path")
        filename = Path(
            source_path or f"hico_train_{fallback_image_id:06d}.jpg"
        ).name
        if not filename:
            raise ValueError("HICO image filename is empty")
        if image_dir is not None:
            image_dir.mkdir(parents=True, exist_ok=True)
            destination = image_dir / filename
            if not destination.exists():
                image.convert("RGB").save(destination, quality=95)
    return int(width), int(height), filename


def convert_hico_rows(
    *,
    actions: Sequence[dict[str, str]],
    rows: Iterable[dict[str, Any]],
    iou_merge: float = RELEASED_IOU_MERGE,
    image_dir: str | Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if iou_merge != RELEASED_IOU_MERGE:
        raise ValueError("released HICO train rebuild requires iou_merge=0.5")
    if len(actions) != 600:
        raise ValueError("released HICO action table must contain 600 rows")

    hoi_obj = [row["nname"] for row in actions]
    hoi_verb = [row["vname"] for row in actions]
    hoi_pred = [row["vname_ing"].replace("_", " ") for row in actions]
    verbs_of_obj: dict[str, set[str]] = defaultdict(set)
    action_to_predicate: dict[tuple[str, str], str] = {}
    for row in actions:
        predicate = row["vname_ing"].replace("_", " ")
        action_to_predicate[(row["nname"], row["vname"])] = predicate
        if row["vname"] != "no_interaction":
            verbs_of_obj[row["nname"]].add(predicate)

    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    rel_annotations: list[dict[str, object]] = []
    negatives_by_image: dict[str, list[list[object]]] = {}
    category_ids: dict[str, int] = {"person": 1}
    predicate_ids: dict[str, int] = {}
    annotation_id = 0
    image_id = 0
    image_root = Path(image_dir) if image_dir is not None else None

    counts = {
        "positive_relations": 0,
        "negative_caption_expansions": 0,
        "negative_no_interaction_expansions": 0,
        "invalid_or_invisible_objects": 0,
        "boxes_raw": 0,
        "boxes_merged": 0,
        "self_loops_dropped": 0,
        "positive_duplicates_dropped": 0,
        "rows_skipped_no_positive_or_negative": 0,
        "relations_outside_40_boxes": 0,
    }

    for raw_row in rows:
        row = _parse_literal_columns(raw_row)
        width, height, filename = _decode_image(
            row.get("image"),
            fallback_image_id=image_id,
            image_dir=image_root,
        )

        box_key_to_local: dict[tuple[int, int, int, int], int] = {}
        local_boxes: list[tuple[list[float], str]] = []

        def add_box(raw_box: object, category: str) -> int:
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                raise ValueError("HICO box must contain [x1,x2,y1,y2]")
            x1, x2, y1, y2 = [
                _require_number(value, "HICO box coordinate")
                for value in raw_box
            ]
            if x2 < x1 or y2 < y1:
                raise ValueError(f"invalid HICO box in {filename}")
            key = (round(x1), round(x2), round(y1), round(y2))
            if key in box_key_to_local:
                return box_key_to_local[key]
            local = len(local_boxes)
            box_key_to_local[key] = local
            local_boxes.append(([x1, y1, x2 - x1, y2 - y1], category))
            return local

        positives: list[tuple[int, int, str]] = []
        no_interaction_pairs: list[tuple[int, int, str]] = []
        humans: set[int] = set()
        object_boxes_of: dict[str, set[int]] = defaultdict(set)

        for object_index, entry in enumerate(row["objects"]):
            if not isinstance(entry, dict):
                raise ValueError(f"HICO objects[{object_index}] must be an object")
            category_id = _require_int(entry.get("id"), f"HICO objects[{object_index}].id")
            if not 1 <= category_id <= 600 or entry.get("invis"):
                counts["invalid_or_invisible_objects"] += 1
                continue

            object_name = hoi_obj[category_id - 1]
            verb_name = hoi_verb[category_id - 1]
            human = add_box(entry.get("bbox_human"), "person")
            object_ = add_box(entry.get("bbox_object"), object_name)
            humans.add(human)
            object_boxes_of[object_name].add(object_)
            if verb_name == "no_interaction":
                no_interaction_pairs.append((human, object_, object_name))
            else:
                positives.append((human, object_, hoi_pred[category_id - 1]))

        counts["boxes_raw"] += len(local_boxes)
        if len(local_boxes) > 1:
            local_boxes, remap = merge_boxes(local_boxes, iou_merge)
            counts["boxes_merged"] += len(local_boxes)

            remapped_positive: list[tuple[int, int, str]] = []
            seen_positive: set[tuple[int, int, str]] = set()
            for human, object_, predicate in positives:
                value = (remap[human], remap[object_], predicate)
                if value[0] == value[1]:
                    counts["self_loops_dropped"] += 1
                    continue
                if value in seen_positive:
                    counts["positive_duplicates_dropped"] += 1
                    continue
                seen_positive.add(value)
                remapped_positive.append(value)
            positives = remapped_positive

            remapped_no_interaction: list[tuple[int, int, str]] = []
            seen_no_interaction: set[tuple[int, int, str]] = set()
            for human, object_, object_name in no_interaction_pairs:
                value = (remap[human], remap[object_], object_name)
                if value[0] != value[1] and value not in seen_no_interaction:
                    seen_no_interaction.add(value)
                    remapped_no_interaction.append(value)
            no_interaction_pairs = remapped_no_interaction
            humans = {remap[value] for value in humans}
            object_boxes_of = defaultdict(
                set,
                {
                    name: {remap[value] for value in values}
                    for name, values in object_boxes_of.items()
                },
            )
        else:
            counts["boxes_merged"] += len(local_boxes)

        positive_cells = set(positives)
        ambiguous: set[tuple[object, str]] = set()
        for value in row["ambiguous_captions"]:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("ambiguous caption must be [object,action]")
            ambiguous.add((value[0], str(value[1]).replace("_", " ")))

        negatives: set[tuple[int, int, str]] = set()
        for value in row["negative_captions"]:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("negative caption must be [object,action]")
            object_name = str(value[0])
            action = str(value[1])
            if action == "no_interaction":
                continue
            predicate = action_to_predicate.get(
                (object_name, action),
                action.replace("_", " "),
            )
            if (object_name, predicate) in ambiguous:
                continue
            for human in humans:
                for object_ in object_boxes_of.get(object_name, ()):
                    candidate = (human, object_, predicate)
                    if candidate not in positive_cells:
                        negatives.add(candidate)
        counts["negative_caption_expansions"] += len(negatives)

        for human, object_, object_name in no_interaction_pairs:
            for predicate in verbs_of_obj[object_name]:
                if (
                    (human, object_, predicate) not in positive_cells
                    and (object_name, predicate) not in ambiguous
                ):
                    negatives.add((human, object_, predicate))
                    counts["negative_no_interaction_expansions"] += 1

        if not positives and not negatives:
            counts["rows_skipped_no_positive_or_negative"] += 1
            continue

        images.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
            }
        )
        global_id_of_local: dict[int, int] = {}
        for local, (bbox, category) in enumerate(local_boxes[:40]):
            category_id = category_ids.setdefault(category, len(category_ids) + 1)
            global_id_of_local[local] = annotation_id
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "bbox": bbox,
                    "category_id": category_id,
                }
            )
            annotation_id += 1

        for human, object_, predicate in positives:
            if human not in global_id_of_local or object_ not in global_id_of_local:
                counts["relations_outside_40_boxes"] += 1
                continue
            predicate_id = predicate_ids.setdefault(predicate, len(predicate_ids))
            rel_annotations.append(
                {
                    "image_id": image_id,
                    "subject_id": global_id_of_local[human],
                    "object_id": global_id_of_local[object_],
                    "predicate_id": predicate_id,
                }
            )
            counts["positive_relations"] += 1

        negative_rows = [
            [human, object_, predicate]
            for human, object_, predicate in sorted(negatives)
            if human in global_id_of_local
            and object_ in global_id_of_local
            and human != object_
        ]
        if negative_rows:
            stem = Path(filename).stem
            try:
                image_number = int(stem.split("_")[-1])
            except ValueError as exc:
                raise ValueError(
                    f"HICO filename lacks numeric suffix required by upstream negatives: {filename}"
                ) from exc
            negatives_by_image[str(image_number)] = negative_rows
        image_id += 1

    negative_only_predicates = sorted(
        {
            predicate
            for values in negatives_by_image.values()
            for _, _, predicate in values
            if predicate not in predicate_ids
        }
    )
    # Upstream inserts a Python set here, whose order depends on hash seed.
    # Released pack provenance can only be deterministic when every negative
    # predicate is already established by positive insertion order.
    if negative_only_predicates:
        raise ValueError(
            "released deterministic HICO rebuild encountered negative-only predicates: "
            + negative_only_predicates[0]
        )

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name}
            for name, index in category_ids.items()
        ],
        "rel_categories": [
            {"id": index, "name": name}
            for name, index in sorted(predicate_ids.items(), key=lambda item: item[1])
        ],
        "rel_annotations": rel_annotations,
    }
    negatives_payload = {
        "predicates": sorted(predicate_ids),
        "by_image_id": negatives_by_image,
    }
    evidence = {
        "rows_kept": image_id,
        "annotation_count": len(annotations),
        "positive_relation_count": len(rel_annotations),
        "category_count": len(category_ids),
        "predicate_count": len(predicate_ids),
        "counts": counts,
        "coco_sgg_sha256": hashlib.sha256(_upstream_json_bytes(coco)).hexdigest(),
        "negatives_sha256": hashlib.sha256(
            _upstream_json_bytes(negatives_payload)
        ).hexdigest(),
    }
    return coco, negatives_payload, evidence


def _expected_input_contract(
    *,
    action_csv: Path,
    shard_reports: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "list_action_csv": sha256_file(action_csv),
        "parquets": [
            {
                "name": report["name"],
                "sha256": report["sha256"],
            }
            for report in shard_reports
        ],
    }


def rebuild_hico_train(
    *,
    action_csv_path: str | Path,
    parquet_dir: str | Path,
    image_output_dir: str | Path,
    coco_output: str | Path,
    negatives_output: str | Path,
    pack_output: str | Path,
    evidence_output: str | Path,
    expected_input: dict[str, object] | None = None,
    source_revision: str | None = None,
    vocabulary_path: str | Path | None = None,
    canonical_output: str | Path | None = None,
) -> dict[str, object]:
    action_csv = Path(action_csv_path)
    if not action_csv.is_file():
        raise FileNotFoundError(action_csv)

    coco_path = Path(coco_output)
    negatives_path = Path(negatives_output)
    pack_path = Path(pack_output)
    evidence_path = Path(evidence_output)
    for path in (coco_path, negatives_path, evidence_path):
        if path.exists():
            raise FileExistsError(path)
    if pack_path.exists():
        raise FileExistsError(pack_path)

    actions = load_actions_csv(action_csv)
    rows, shard_reports = iter_train_parquet_rows(parquet_dir)
    coco, negatives, conversion = convert_hico_rows(
        actions=actions,
        rows=rows,
        image_dir=image_output_dir,
    )

    input_contract = _expected_input_contract(
        action_csv=action_csv,
        shard_reports=shard_reports,
    )
    if expected_input is not None and input_contract != expected_input:
        raise ValueError("HICO semantic input hashes/shard order drifted from expected contract")

    coco_path.parent.mkdir(parents=True, exist_ok=True)
    coco_path.write_bytes(_upstream_json_bytes(coco))
    negatives_path.parent.mkdir(parents=True, exist_ok=True)
    negatives_path.write_bytes(_upstream_json_bytes(negatives))

    pack_evidence = pack_coco_sgg_train(
        coco_path,
        pack_path,
        dataset_name=RELEASED_DATASET_NAME,
        ann_source_label=RELEASED_ANN_SOURCE_LABEL,
        img_dir_label=RELEASED_IMG_DIR_LABEL,
    )
    for component in PACK_COMPONENTS:
        if not (pack_path / component).is_file():
            raise RuntimeError(f"HICO train pack is missing {component}")

    canonical_evidence = None
    if vocabulary_path is not None or canonical_output is not None:
        if vocabulary_path is None or canonical_output is None:
            raise ValueError("vocabulary_path and canonical_output must be supplied together")
        canonical_evidence = materialize_pack(
            pack_path,
            RelationVocabulary.load(vocabulary_path),
            canonical_output,
        )

    corpus_source_candidate = None
    if (
        expected_input is not None
        and source_revision
        and canonical_evidence is not None
    ):
        corpus_source_candidate = {
            "name": RELEASED_DATASET_NAME,
            "provenance_kind": "deterministic-rebuild",
            "origin": "zhimeng/hico_det train parquet snapshot + HICO list_action.csv",
            "revision": source_revision,
            "recipe": (
                REFERENCE_SOURCE_COMMIT
                + ":training/convert_hicodet.py+training/pack_megasg.py"
            ),
            "annotations_sha256": canonical_evidence[
                "canonical_annotations_sha256"
            ],
            "post_exclusion_count": canonical_evidence["images"],
        }

    report = {
        "schema": EVIDENCE_SCHEMA,
        "reference_source_commit": REFERENCE_SOURCE_COMMIT,
        "split": RELEASED_SPLIT,
        "iou_merge": RELEASED_IOU_MERGE,
        "input_contract": input_contract,
        "input_hashes_pinned": expected_input is not None,
        "source_revision": source_revision,
        "shards": shard_reports,
        "conversion": conversion,
        "pack": pack_evidence,
        "canonical_materialization": canonical_evidence,
        "corpus_source_candidate": corpus_source_candidate,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-action", required=True)
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--image-out-dir", required=True)
    parser.add_argument("--coco-out", required=True)
    parser.add_argument("--negatives-out", required=True)
    parser.add_argument("--pack-out", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--expected-input")
    parser.add_argument("--source-revision")
    parser.add_argument("--vocabulary")
    parser.add_argument("--canonical-out")
    args = parser.parse_args()

    expected = None
    if args.expected_input:
        expected = json.loads(
            Path(args.expected_input).read_text(encoding="utf-8")
        )
        if not isinstance(expected, dict):
            raise ValueError("expected HICO input contract must be an object")

    report = rebuild_hico_train(
        action_csv_path=args.list_action,
        parquet_dir=args.parquet_dir,
        image_output_dir=args.image_out_dir,
        coco_output=args.coco_out,
        negatives_output=args.negatives_out,
        pack_output=args.pack_out,
        evidence_output=args.evidence,
        expected_input=expected,
        source_revision=args.source_revision,
        vocabulary_path=args.vocabulary,
        canonical_output=args.canonical_out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
