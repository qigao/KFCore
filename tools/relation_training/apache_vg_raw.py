from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

from apache_pack_builder import pack_coco_sgg_train
from apache_pack_materializer import materialize_pack
from benchmark import RelationVocabulary


REBUILD_SCHEMA = "kfcore.apache-vg-raw-rebuild/1"
REFERENCE_SOURCE_COMMIT = (
    "4a07de9d06f2e3f14309753b7907cf1d3a263b08"
)
RELEASED_MAX_WORDS = 5


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


def _normalize_text(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _writer_predicate(value: str) -> str:
    result = re.sub(r"\s+", " ", value.strip().lower())
    return result if 0 < len(result) <= 60 else ""


def _object_name(value: object) -> str:
    if not isinstance(value, dict):
        raise ValueError("VG relation endpoint must be an object")
    name = value.get("name")
    if not name:
        names = value.get("names") or []
        if not isinstance(names, list):
            raise ValueError("VG endpoint names must be an array")
        name = names[0] if names else ""
    return _normalize_text(name)


def _require_json_object(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _parse_int_map(path: Path, name: str) -> dict[str, int]:
    payload = _require_json_object(path, name)
    result: dict[str, int] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if isinstance(value, bool):
            raise ValueError(f"{name} values must be integers")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} values must be integers") from error
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{name} values must be integers")
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or str(parsed) != stripped.lstrip("+").lstrip("0") and not (
                parsed == 0 and set(stripped.lstrip("+")) <= {"0"}
            ):
                # Keep the accepted surface intentionally narrow: decimal integers.
                if not re.fullmatch(r"[+]?[0-9]+", stripped):
                    raise ValueError(f"{name} values must be decimal integers")
        if parsed < 0:
            raise ValueError(f"{name} values must be non-negative")
        result[key] = parsed
    return result


class _CocoSGGWriter:
    def __init__(self) -> None:
        self.images: list[dict[str, object]] = []
        self.annotations: list[dict[str, object]] = []
        self.rel_annotations: list[dict[str, object]] = []
        self.cat_ids: dict[str, int] = {}
        self.pred_ids: dict[str, int] = {}
        self.pred_counts: Counter[str] = Counter()
        self.next_annotation_id = 1
        self.dropped_writer_predicate = 0
        self.dropped_writer_self_loop = 0
        self.dropped_writer_duplicate = 0
        self.rolled_back_images = 0

    def _cat(self, name: str) -> int:
        return self.cat_ids.setdefault(name, len(self.cat_ids))

    def _pred(self, name: str) -> int:
        return self.pred_ids.setdefault(name, len(self.pred_ids))

    def add_image(
        self,
        image_id: int,
        file_name: str,
        width: int,
        height: int,
        boxes: list[list[float]],
        categories: list[str],
        relations: list[tuple[int, int, str]],
    ) -> None:
        if len(boxes) < 2 or not relations:
            return
        if len(boxes) != len(categories):
            raise ValueError("VG converted boxes/categories length mismatch")

        global_ids: list[int] = []
        for box, category in zip(boxes, categories):
            global_ids.append(self.next_annotation_id)
            self.annotations.append(
                {
                    "id": self.next_annotation_id,
                    "image_id": image_id,
                    "bbox": [float(value) for value in box],
                    "category_id": self._cat(category),
                }
            )
            self.next_annotation_id += 1

        kept = 0
        seen: set[tuple[int, int, str]] = set()
        for subject, object_, raw_predicate in relations:
            predicate = _writer_predicate(raw_predicate)
            if not predicate:
                self.dropped_writer_predicate += 1
                continue
            if subject == object_:
                self.dropped_writer_self_loop += 1
                continue
            key = (subject, object_, predicate)
            if key in seen:
                self.dropped_writer_duplicate += 1
                continue
            if not (0 <= subject < len(global_ids)) or not (
                0 <= object_ < len(global_ids)
            ):
                raise ValueError("VG converted relation endpoint is out of range")
            seen.add(key)
            self.rel_annotations.append(
                {
                    "image_id": image_id,
                    "subject_id": global_ids[subject],
                    "object_id": global_ids[object_],
                    "predicate_id": self._pred(predicate),
                }
            )
            self.pred_counts[predicate] += 1
            kept += 1

        if kept == 0:
            del self.annotations[-len(global_ids) :]
            self.next_annotation_id -= len(global_ids)
            self.rolled_back_images += 1
            return

        self.images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(width),
                "height": int(height),
            }
        )

    def payload(self) -> dict[str, object]:
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [
                {"id": index, "name": name}
                for name, index in sorted(
                    self.cat_ids.items(),
                    key=lambda item: item[1],
                )
            ],
            "rel_categories": [
                {"id": index, "name": name}
                for name, index in sorted(
                    self.pred_ids.items(),
                    key=lambda item: item[1],
                )
            ],
            "rel_annotations": self.rel_annotations,
        }


def convert_vg_raw(
    *,
    relationships_path: str | Path,
    image_data_path: str | Path,
    image_root: str | Path,
    registry_path: str | Path,
    vg2coco_path: str | Path,
    psg2coco_path: str | Path,
    max_words: int = RELEASED_MAX_WORDS,
) -> tuple[dict[str, object], dict[str, object]]:
    if max_words != RELEASED_MAX_WORDS:
        raise ValueError("released vg_raw rebuild requires max_words=5")

    relationships_file = Path(relationships_path)
    image_data_file = Path(image_data_path)
    image_directory = Path(image_root)
    registry_file = Path(registry_path)
    vg2coco_file = Path(vg2coco_path)
    psg2coco_file = Path(psg2coco_path)
    for path in (
        relationships_file,
        image_data_file,
        registry_file,
        vg2coco_file,
        psg2coco_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not image_directory.is_dir():
        raise FileNotFoundError(image_directory)

    registry = _require_json_object(registry_file, "registry.json")
    banned_raw = registry.get("banned_coco_ids") or registry.get(
        "protected_coco_ids"
    )
    protected_vg_raw = registry.get("protected_vg_ids")
    if not isinstance(banned_raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in banned_raw
    ):
        raise ValueError("registry protected/banned COCO ids must be non-negative integers")
    if not isinstance(protected_vg_raw, list) or any(
        not isinstance(value, str) or not value for value in protected_vg_raw
    ):
        raise ValueError("registry protected VG ids must be non-empty strings")
    protected_coco = set(banned_raw)
    protected_vg = set(protected_vg_raw)

    vg2coco = _parse_int_map(vg2coco_file, "vg2coco.json")
    # Upstream load_registry parses this even though convert_vg_raw ignores it.
    psg2coco = _parse_int_map(psg2coco_file, "psg2coco.json")

    allowed_stems = sorted(
        {
            name[:-4]
            for name in os.listdir(image_directory)
            if name.endswith(".jpg")
        }
    )
    allowed = set(allowed_stems)

    image_data = json.loads(image_data_file.read_text(encoding="utf-8"))
    if not isinstance(image_data, list):
        raise ValueError("image_data.json must be an array")
    dimensions: dict[str, tuple[int, int]] = {}
    for index, image in enumerate(image_data):
        if not isinstance(image, dict):
            raise ValueError(f"image_data[{index}] must be an object")
        image_id = str(image.get("image_id"))
        if not image_id or image_id == "None":
            raise ValueError(f"image_data[{index}] has invalid image_id")
        if image_id in dimensions:
            raise ValueError(f"duplicate image_data image_id {image_id}")
        width = image.get("width")
        height = image.get("height")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            raise ValueError(f"image_data[{index}] has invalid dimensions")
        dimensions[image_id] = (width, height)

    data = json.loads(relationships_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("relationships.json must be an array")

    writer = _CocoSGGWriter()
    drops: Counter[str] = Counter()

    for record_index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"relationships[{record_index}] must be an object")
        raw_image_id = record.get("image_id")
        if isinstance(raw_image_id, bool):
            raise ValueError("VG image_id must be integer-like")
        try:
            image_id = int(raw_image_id)
        except (TypeError, ValueError) as error:
            raise ValueError("VG image_id must be integer-like") from error
        vg_id = str(raw_image_id)
        if vg_id not in allowed:
            drops["image_not_on_disk"] += 1
            continue
        mapped_coco = vg2coco.get(vg_id)
        if vg_id in protected_vg or (
            mapped_coco is not None and mapped_coco in protected_coco
        ):
            drops["eval_protected"] += 1
            continue
        dimensions_row = dimensions.get(vg_id)
        if dimensions_row is None:
            drops["no_dims"] += 1
            continue

        box_of: dict[int, int] = {}
        boxes: list[list[float]] = []
        categories: list[str] = []
        relations: list[tuple[int, int, str]] = []

        def box_index(endpoint: object) -> int | None:
            if not isinstance(endpoint, dict):
                raise ValueError("VG relationship endpoint must be an object")
            name = _object_name(endpoint)
            if not name:
                drops["obj_no_name"] += 1
                return None
            if len(name.split()) > max_words:
                drops["obj_name_too_long"] += 1
                return None
            object_id = endpoint.get("object_id")
            if isinstance(object_id, bool) or not isinstance(object_id, int):
                raise ValueError("VG object_id must be an integer")
            if object_id in box_of:
                return box_of[object_id]
            for key in ("x", "y", "w", "h"):
                value = endpoint.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"VG object {key} must be numeric")
                if not np_isfinite(float(value)):
                    raise ValueError(f"VG object {key} must be finite")
            box_of[object_id] = len(boxes)
            boxes.append(
                [
                    float(endpoint["x"]),
                    float(endpoint["y"]),
                    max(float(endpoint["w"]), 1.0),
                    max(float(endpoint["h"]), 1.0),
                ]
            )
            categories.append(name)
            return box_of[object_id]

        raw_relations = record.get("relationships") or []
        if not isinstance(raw_relations, list):
            raise ValueError("VG relationships field must be an array")
        for relation_index, relation in enumerate(raw_relations):
            if not isinstance(relation, dict):
                raise ValueError(
                    f"VG relationship {record_index}:{relation_index} must be an object"
                )
            predicate = _normalize_text(relation.get("predicate") or "")
            if not predicate:
                drops["pred_empty"] += 1
                continue
            if len(predicate.split()) > max_words:
                drops["pred_too_long"] += 1
                continue
            subject = box_index(relation.get("subject"))
            object_ = box_index(relation.get("object"))
            if subject is None or object_ is None or subject == object_:
                continue
            relations.append((subject, object_, predicate))

        if len(boxes) < 2 or not relations:
            drops["image_no_usable_rels"] += 1
            continue

        width, height = dimensions_row
        writer.add_image(
            image_id,
            f"{vg_id}.jpg",
            width,
            height,
            boxes,
            categories,
            relations,
        )

    output = writer.payload()

    kept_ids = {str(image["id"]) for image in writer.images}
    leaked = {
        vg_id
        for vg_id in kept_ids
        if vg_id in protected_vg
        or (
            vg_id in vg2coco
            and vg2coco[vg_id] in protected_coco
        )
    }
    if leaked:
        raise RuntimeError(
            f"VG raw leakage check failed for {len(leaked)} images"
        )

    evidence = {
        "schema": REBUILD_SCHEMA,
        "reference_source_commit": REFERENCE_SOURCE_COMMIT,
        "max_words": RELEASED_MAX_WORDS,
        "relationships_sha256": sha256_file(relationships_file),
        "image_data_sha256": sha256_file(image_data_file),
        "registry_sha256": sha256_file(registry_file),
        "vg2coco_sha256": sha256_file(vg2coco_file),
        "psg2coco_sha256": sha256_file(psg2coco_file),
        "psg2coco_entry_count": len(psg2coco),
        "allowed_image_stems": allowed_stems,
        "allowed_image_stem_count": len(allowed_stems),
        "allowed_image_stems_sha256": ordered_strings_sha256(allowed_stems),
        "protected_vg_ids": sorted(protected_vg),
        "protected_vg_count": len(protected_vg),
        "protected_vg_ids_sha256": ordered_strings_sha256(sorted(protected_vg)),
        "protected_coco_ids": sorted(protected_coco),
        "protected_coco_count": len(protected_coco),
        "drops": dict(drops),
        "writer_drops": {
            "predicate": writer.dropped_writer_predicate,
            "self_loop": writer.dropped_writer_self_loop,
            "duplicate_triplet": writer.dropped_writer_duplicate,
            "rolled_back_images": writer.rolled_back_images,
        },
        "output_images": len(writer.images),
        "output_annotations": len(writer.annotations),
        "output_relations": len(writer.rel_annotations),
        "output_categories": len(writer.cat_ids),
        "output_predicates": len(writer.pred_ids),
        "leaked_image_count": 0,
    }
    return output, evidence


def np_isfinite(value: float) -> bool:
    # Avoid importing numpy for a scalar-only check in callers of this module.
    return value == value and value not in (float("inf"), float("-inf"))


def coco_bytes(payload: dict[str, object]) -> bytes:
    # Match upstream CocoSGGWriter.write: json.dump without sort/indent/newline.
    return json.dumps(payload).encode("utf-8")


def rebuild_vg_raw_source(
    *,
    relationships_path: str | Path,
    image_data_path: str | Path,
    image_root: str | Path,
    registry_path: str | Path,
    vg2coco_path: str | Path,
    psg2coco_path: str | Path,
    vocabulary_path: str | Path,
    coco_output: str | Path,
    pack_output: str | Path,
    canonical_output: str | Path,
    ann_source_label: str = "rebuild/vg_raw_train_coco.json",
    img_dir_label: str = "datasets/VG150_coco_format/train",
) -> dict[str, object]:
    coco_path = Path(coco_output)
    canonical_path = Path(canonical_output)
    if coco_path.exists():
        raise FileExistsError(coco_path)
    if canonical_path.exists():
        raise FileExistsError(canonical_path)

    converted, conversion = convert_vg_raw(
        relationships_path=relationships_path,
        image_data_path=image_data_path,
        image_root=image_root,
        registry_path=registry_path,
        vg2coco_path=vg2coco_path,
        psg2coco_path=psg2coco_path,
    )
    coco_path.parent.mkdir(parents=True, exist_ok=True)
    raw_coco = coco_bytes(converted)
    coco_path.write_bytes(raw_coco)
    conversion = {
        **conversion,
        "coco_sgg_sha256": hashlib.sha256(raw_coco).hexdigest(),
    }

    pack = pack_coco_sgg_train(
        coco_path,
        pack_output,
        dataset_name="vg_raw",
        ann_source_label=ann_source_label,
        img_dir_label=img_dir_label,
    )

    vocabulary = RelationVocabulary.load(vocabulary_path)
    canonical = materialize_pack(
        pack_output,
        vocabulary,
        canonical_path,
    )

    return {
        "schema": "kfcore.apache-vg-raw-source-rebuild/1",
        "reference_source_commit": REFERENCE_SOURCE_COMMIT,
        "conversion": conversion,
        "pack": pack,
        "canonical": canonical,
        "vocabulary_sha256": vocabulary.sha256(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relationships", required=True)
    parser.add_argument("--image-data", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--vg2coco", required=True)
    parser.add_argument("--psg2coco", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--coco-out", required=True)
    parser.add_argument("--pack-out", required=True)
    parser.add_argument("--canonical-out", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()

    evidence = rebuild_vg_raw_source(
        relationships_path=args.relationships,
        image_data_path=args.image_data,
        image_root=args.image_root,
        registry_path=args.registry,
        vg2coco_path=args.vg2coco,
        psg2coco_path=args.psg2coco,
        vocabulary_path=args.vocabulary,
        coco_output=args.coco_out,
        pack_output=args.pack_out,
        canonical_output=args.canonical_out,
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
