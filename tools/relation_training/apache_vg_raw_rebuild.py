from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from apache_pack_builder import (
    PACK_COMPONENTS,
    pack_coco_sgg_train,
)
from apache_pack_materializer import materialize_pack
from benchmark import RelationVocabulary


EVIDENCE_SCHEMA = "kfcore.apache-vg-raw-rebuild/1"
REFERENCE_SOURCE_COMMIT = "4a07de9d06f2e3f14309753b7907cf1d3a263b08"
RELEASED_MAX_WORDS = 5
RELEASED_DATASET_NAME = "vg_raw"
RELEASED_ANN_SOURCE_LABEL = "runs/datamix/vg_raw_train_coco.json"
RELEASED_IMG_DIR_LABEL = "VG150_coco_format/train"


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


def stable_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normalize_predicate(value: object) -> str:
    text = _normalize_text(value)
    return text if 0 < len(text) <= 60 else ""


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


def _string_set(values: object, name: str) -> set[str]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be an array")
    result: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name}[{index}] must be a non-empty string")
        result.add(value)
    return result


def _int_set(values: object, name: str) -> set[int]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be an array")
    return {_require_int(value, f"{name}[{index}]") for index, value in enumerate(values)}


def _load_mapping(path: Path, name: str) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be an object")
    result: dict[str, int] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if isinstance(value, bool):
            raise ValueError(f"{name}[{key}] must be an integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{key}] must be an integer") from exc
        result[key] = parsed
    return result


class _CocoSGGWriter:
    """Exact relevant behavior of Apache convert_datamix.CocoSGGWriter."""

    def __init__(self) -> None:
        self.images: list[dict[str, object]] = []
        self.annotations: list[dict[str, object]] = []
        self.rel_annotations: list[dict[str, object]] = []
        self.cat_ids: dict[str, int] = {}
        self.pred_ids: dict[str, int] = {}
        self._next_ann = 1

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
        boxes: Sequence[Sequence[float]],
        categories: Sequence[str],
        relations: Sequence[tuple[int, int, str]],
    ) -> None:
        if len(boxes) < 2 or not relations:
            return
        if len(boxes) != len(categories):
            raise ValueError("boxes/categories length mismatch")

        global_ids: list[int] = []
        for box, category in zip(boxes, categories):
            global_ids.append(self._next_ann)
            self.annotations.append(
                {
                    "id": self._next_ann,
                    "image_id": image_id,
                    "bbox": [float(value) for value in box],
                    "category_id": self._cat(category),
                }
            )
            self._next_ann += 1

        kept = 0
        seen: set[tuple[int, int, str]] = set()
        for subject, object_, raw_predicate in relations:
            predicate = _normalize_predicate(raw_predicate)
            key = (subject, object_, predicate)
            if (
                not predicate
                or subject == object_
                or subject < 0
                or object_ < 0
                or subject >= len(global_ids)
                or object_ >= len(global_ids)
                or key in seen
            ):
                continue
            seen.add(key)
            self.rel_annotations.append(
                {
                    "image_id": image_id,
                    "subject_id": global_ids[subject],
                    "object_id": global_ids[object_],
                    "predicate_id": self._pred(predicate),
                }
            )
            kept += 1

        if kept == 0:
            del self.annotations[-len(global_ids):]
            self._next_ann -= len(global_ids)
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
                for name, index in sorted(self.cat_ids.items(), key=lambda item: item[1])
            ],
            "rel_categories": [
                {"id": index, "name": name}
                for name, index in sorted(self.pred_ids.items(), key=lambda item: item[1])
            ],
            "rel_annotations": self.rel_annotations,
        }


def _upstream_json_bytes(payload: object) -> bytes:
    # CocoSGGWriter.write uses json.dump with default separators/no newline.
    return json.dumps(payload).encode("utf-8")


def convert_vg_raw(
    *,
    relationships_path: str | Path,
    image_data_path: str | Path,
    image_root: str | Path,
    registry_path: str | Path,
    vg2coco_path: str | Path,
    psg2coco_path: str | Path,
    max_words: int = RELEASED_MAX_WORDS,
    expected_input_sha256: dict[str, str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if max_words != RELEASED_MAX_WORDS:
        raise ValueError("released VG raw rebuild requires max_words=5")

    relationships_file = Path(relationships_path)
    image_data_file = Path(image_data_path)
    root = Path(image_root)
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
    if not root.is_dir():
        raise FileNotFoundError(root)

    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("registry.json must be an object")
    protected_vg = _string_set(registry.get("protected_vg_ids"), "protected_vg_ids")
    protected_coco_raw = registry.get("banned_coco_ids")
    protected_coco_key = "banned_coco_ids"
    if not protected_coco_raw:
        protected_coco_raw = registry.get("protected_coco_ids")
        protected_coco_key = "protected_coco_ids"
    protected_coco = _int_set(protected_coco_raw, protected_coco_key)

    vg2coco = _load_mapping(vg2coco_file, "vg2coco")
    psg2coco = _load_mapping(psg2coco_file, "psg2coco")

    allowed = {
        filename[:-4]
        for filename in os.listdir(root)
        if filename.endswith(".jpg")
    }
    allowed_sorted = sorted(allowed)

    dims_payload = json.loads(image_data_file.read_text(encoding="utf-8"))
    if not isinstance(dims_payload, list):
        raise ValueError("image_data.json must be an array")
    dims: dict[str, tuple[int, int]] = {}
    for index, image in enumerate(dims_payload):
        if not isinstance(image, dict):
            raise ValueError(f"image_data[{index}] must be an object")
        image_id = str(_require_int(image.get("image_id"), f"image_data[{index}].image_id"))
        width = _require_int(image.get("width"), f"image_data[{index}].width")
        height = _require_int(image.get("height"), f"image_data[{index}].height")
        if width <= 0 or height <= 0:
            raise ValueError(f"image_data[{index}] dimensions must be positive")
        dims[image_id] = (width, height)

    relationships_payload = json.loads(relationships_file.read_text(encoding="utf-8"))
    if not isinstance(relationships_payload, list):
        raise ValueError("relationships.json must be an array")

    input_sha256 = {
        "relationships_json": sha256_file(relationships_file),
        "image_data_json": sha256_file(image_data_file),
        "registry_json": sha256_file(registry_file),
        "vg2coco_json": sha256_file(vg2coco_file),
        "psg2coco_json": sha256_file(psg2coco_file),
    }
    if expected_input_sha256 is not None:
        if set(expected_input_sha256) != set(input_sha256):
            raise ValueError("expected input SHA-256 keys do not match the VG raw contract")
        for name, actual in input_sha256.items():
            expected = expected_input_sha256[name]
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)
            ):
                raise ValueError(f"expected {name} SHA-256 must be lowercase hex")
            if actual != expected:
                raise ValueError(
                    f"{name} SHA-256 drift: expected {expected}, got {actual}"
                )

    writer = _CocoSGGWriter()
    drops: Counter[str] = Counter()

    for record_index, record in enumerate(relationships_payload):
        if not isinstance(record, dict):
            raise ValueError(f"relationships[{record_index}] must be an object")
        image_id_int = _require_int(
            record.get("image_id"),
            f"relationships[{record_index}].image_id",
        )
        image_id = str(image_id_int)

        if image_id not in allowed:
            drops["image_not_on_disk"] += 1
            continue
        if (
            image_id in protected_vg
            or (
                image_id in vg2coco
                and vg2coco[image_id] in protected_coco
            )
        ):
            drops["eval_protected"] += 1
            continue
        dimensions = dims.get(image_id)
        if dimensions is None:
            drops["no_dims"] += 1
            continue

        object_to_box: dict[object, int] = {}
        boxes: list[list[float]] = []
        categories: list[str] = []
        relations: list[tuple[int, int, str]] = []

        def box_index(endpoint: object) -> int | None:
            if not isinstance(endpoint, dict):
                raise ValueError("VG relation endpoint must be an object")
            name = _object_name(endpoint)
            if not name:
                drops["obj_no_name"] += 1
                return None
            if len(name.split()) > max_words:
                drops["obj_name_too_long"] += 1
                return None

            object_id = endpoint.get("object_id")
            if object_id in object_to_box:
                return object_to_box[object_id]

            index = len(boxes)
            object_to_box[object_id] = index
            x = _require_number(endpoint.get("x"), "VG object x")
            y = _require_number(endpoint.get("y"), "VG object y")
            width = max(_require_number(endpoint.get("w"), "VG object w"), 1.0)
            height = max(_require_number(endpoint.get("h"), "VG object h"), 1.0)
            boxes.append([x, y, width, height])
            categories.append(name)
            return index

        raw_relations = record.get("relationships") or []
        if not isinstance(raw_relations, list):
            raise ValueError("VG relationships field must be an array")
        for relation_index, relation in enumerate(raw_relations):
            if not isinstance(relation, dict):
                raise ValueError(
                    f"relationships[{record_index}].relationships[{relation_index}] must be an object"
                )
            predicate = _normalize_text(relation.get("predicate"))
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
        writer.add_image(
            image_id_int,
            f"{image_id}.jpg",
            dimensions[0],
            dimensions[1],
            boxes,
            categories,
            relations,
        )

    kept_ids = {str(image["id"]) for image in writer.images}
    leaked = sorted(
        image_id
        for image_id in kept_ids
        if (
            image_id in protected_vg
            or (
                image_id in vg2coco
                and vg2coco[image_id] in protected_coco
            )
        )
    )
    if leaked:
        raise RuntimeError(
            f"VG raw leakage check failed: {len(leaked)} protected images survived"
        )

    payload = writer.payload()
    raw = _upstream_json_bytes(payload)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "reference_source_commit": REFERENCE_SOURCE_COMMIT,
        "max_words": RELEASED_MAX_WORDS,
        "input_sha256": input_sha256,
        "input_hashes_pinned": expected_input_sha256 is not None,
        "registry_policy": {
            "protected_coco_key": protected_coco_key,
            "protected_coco_count": len(protected_coco),
            "protected_coco_sha256": stable_json_sha256(sorted(protected_coco)),
            "protected_vg_count": len(protected_vg),
            "protected_vg_sha256": ordered_strings_sha256(sorted(protected_vg)),
            "vg2coco_count": len(vg2coco),
            "vg2coco_normalized_sha256": stable_json_sha256(vg2coco),
            "psg2coco_count": len(psg2coco),
            "psg2coco_normalized_sha256": stable_json_sha256(psg2coco),
        },
        "image_stem_allow_set": {
            "count": len(allowed_sorted),
            "sha256": ordered_strings_sha256(allowed_sorted),
        },
        "image_dimension_count": len(dims),
        "relationship_record_count": len(relationships_payload),
        "drops": dict(drops),
        "output": {
            "image_count": len(writer.images),
            "annotation_count": len(writer.annotations),
            "relation_count": len(writer.rel_annotations),
            "category_count": len(writer.cat_ids),
            "predicate_count": len(writer.pred_ids),
            "coco_sgg_sha256": hashlib.sha256(raw).hexdigest(),
            "leakage_count": 0,
        },
    }
    return payload, evidence


def rebuild_vg_raw(
    *,
    relationships_path: str | Path,
    image_data_path: str | Path,
    image_root: str | Path,
    registry_path: str | Path,
    vg2coco_path: str | Path,
    psg2coco_path: str | Path,
    coco_output: str | Path,
    pack_output: str | Path,
    evidence_output: str | Path,
    expected_input_sha256: dict[str, str] | None = None,
    vocabulary_path: str | Path | None = None,
    canonical_output: str | Path | None = None,
) -> dict[str, object]:
    coco_path = Path(coco_output)
    pack_path = Path(pack_output)
    evidence_path = Path(evidence_output)
    if coco_path.exists():
        raise FileExistsError(coco_path)
    if pack_path.exists():
        raise FileExistsError(pack_path)
    if evidence_path.exists():
        raise FileExistsError(evidence_path)

    payload, converter_evidence = convert_vg_raw(
        relationships_path=relationships_path,
        image_data_path=image_data_path,
        image_root=image_root,
        registry_path=registry_path,
        vg2coco_path=vg2coco_path,
        psg2coco_path=psg2coco_path,
        expected_input_sha256=expected_input_sha256,
    )
    coco_path.parent.mkdir(parents=True, exist_ok=True)
    coco_path.write_bytes(_upstream_json_bytes(payload))

    if sha256_file(coco_path) != converter_evidence["output"]["coco_sgg_sha256"]:
        raise RuntimeError("written VG raw COCO-SGG hash differs from converter evidence")

    pack_evidence = pack_coco_sgg_train(
        coco_path,
        pack_path,
        dataset_name=RELEASED_DATASET_NAME,
        ann_source_label=RELEASED_ANN_SOURCE_LABEL,
        img_dir_label=RELEASED_IMG_DIR_LABEL,
    )
    for component in PACK_COMPONENTS:
        if not (pack_path / component).is_file():
            raise RuntimeError(f"VG raw pack is missing {component}")

    canonical_evidence = None
    if vocabulary_path is not None or canonical_output is not None:
        if vocabulary_path is None or canonical_output is None:
            raise ValueError("vocabulary_path and canonical_output must be supplied together")
        vocabulary = RelationVocabulary.load(vocabulary_path)
        canonical_evidence = materialize_pack(
            pack_path,
            vocabulary,
            canonical_output,
        )

    corpus_source_candidate = None
    if converter_evidence["input_hashes_pinned"] and canonical_evidence is not None:
        revision_payload = {
            "input_sha256": converter_evidence["input_sha256"],
            "image_stem_allow_set_sha256": converter_evidence[
                "image_stem_allow_set"
            ]["sha256"],
        }
        revision = "sha256:" + stable_json_sha256(revision_payload)
        corpus_source_candidate = {
            "name": RELEASED_DATASET_NAME,
            "provenance_kind": "deterministic-rebuild",
            "origin": "Visual Genome raw metadata + VG150_coco_format/train",
            "revision": revision,
            "recipe": (
                REFERENCE_SOURCE_COMMIT
                + ":training/convert_vg_raw.py+training/pack_megasg.py"
            ),
            "annotations_sha256": canonical_evidence[
                "canonical_annotations_sha256"
            ],
            "post_exclusion_count": canonical_evidence["images"],
        }

    report = {
        **converter_evidence,
        "coco_sgg_path": str(coco_path.resolve()),
        "pack_path": str(pack_path.resolve()),
        "pack": pack_evidence,
        "canonical_materialization": canonical_evidence,
        "corpus_source_candidate": corpus_source_candidate,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relationships", required=True)
    parser.add_argument("--image-data", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--vg2coco", required=True)
    parser.add_argument("--psg2coco", required=True)
    parser.add_argument("--coco-out", required=True)
    parser.add_argument("--pack-out", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument(
        "--expected-input-hashes",
        help="Optional JSON object pinning all five semantic input SHA-256 values.",
    )
    parser.add_argument(
        "--vocabulary",
        help="Optional released union vocabulary for pack->canonical materialization.",
    )
    parser.add_argument(
        "--canonical-out",
        help="Canonical JSONL output; requires --vocabulary.",
    )
    args = parser.parse_args()

    expected_input_sha256 = None
    if args.expected_input_hashes:
        raw_expected = json.loads(
            Path(args.expected_input_hashes).read_text(encoding="utf-8")
        )
        if not isinstance(raw_expected, dict):
            raise ValueError("expected input hashes must be a JSON object")
        expected_input_sha256 = raw_expected

    report = rebuild_vg_raw(
        relationships_path=args.relationships,
        image_data_path=args.image_data,
        image_root=args.image_root,
        registry_path=args.registry,
        vg2coco_path=args.vg2coco,
        psg2coco_path=args.psg2coco,
        coco_output=args.coco_out,
        pack_output=args.pack_out,
        evidence_output=args.evidence,
        expected_input_sha256=expected_input_sha256,
        vocabulary_path=args.vocabulary,
        canonical_output=args.canonical_out,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
