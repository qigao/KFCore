from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from PIL import Image

from benchmark import DatasetManifest, RelationVocabulary, VOCAB_SCHEMA


BASE_COLUMNS = (
    "ImageID",
    "LabelName1",
    "LabelName2",
    "XMin1",
    "XMax1",
    "YMin1",
    "YMax1",
    "XMin2",
    "XMax2",
    "YMin2",
    "YMax2",
)
RELATION_COLUMN_ALIASES = ("RelationLabel", "RelationshipLabel")
CANONICAL_COLUMNS = BASE_COLUMNS + ("RelationLabel",)


def _relation_column(fieldnames: Sequence[str]) -> str:
    present = [
        name for name in RELATION_COLUMN_ALIASES
        if name in fieldnames
    ]
    if len(present) != 1:
        raise ValueError(
            "Open Images relationship CSV must contain exactly one of "
            "RelationLabel or RelationshipLabel"
        )
    return present[0]


@dataclass(frozen=True, order=True)
class NormalizedEndpoint:
    label_mid: str
    xmin: Decimal
    xmax: Decimal
    ymin: Decimal
    ymax: Decimal

    def validate(self) -> None:
        if not self.label_mid:
            raise ValueError("Open Images endpoint label MID must not be empty")
        values = (self.xmin, self.xmax, self.ymin, self.ymax)
        if any(not value.is_finite() for value in values):
            raise ValueError("Open Images endpoint coordinates must be finite")
        if (
            self.xmin < 0
            or self.ymin < 0
            or self.xmax > 1
            or self.ymax > 1
            or self.xmax <= self.xmin
            or self.ymax <= self.ymin
        ):
            raise ValueError("Open Images endpoint box must be inside [0,1]")

    def pixel_box(
        self,
        *,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        self.validate()
        return (
            float(self.xmin * width),
            float(self.ymin * height),
            float(self.xmax * width),
            float(self.ymax * height),
        )


@dataclass(frozen=True)
class OpenImagesRelation:
    image_id: str
    subject: NormalizedEndpoint
    predicate: str
    object: NormalizedEndpoint


@dataclass(frozen=True)
class SkippedOpenImagesRelation:
    image_id: str
    reason: str

    def __post_init__(self) -> None:
        if self.reason not in {"attribute", "self_relation"}:
            raise ValueError("unsupported Open Images skip reason")


@dataclass(frozen=True)
class ScanSummary:
    relationships: int
    object_relationships: int
    skipped_attributes: int
    skipped_self_relations: int
    images: tuple[str, ...]
    predicates: tuple[str, ...]
    object_mids: tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_class_descriptions(path: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with Path(path).open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.reader(stream)
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(
                    f"class descriptions line {line_number} must have two columns"
                )
            mid, description = row
            if not mid or not description:
                raise ValueError(
                    f"class descriptions line {line_number} is empty"
                )
            previous = result.get(mid)
            if previous is not None and previous != description:
                raise ValueError(
                    f"class description MID {mid} has conflicting names"
                )
            result[mid] = description
    if not result:
        raise ValueError("class descriptions file is empty")
    return result


def _decimal(value: str, field: str, line_number: int) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(
            f"Open Images line {line_number} field {field} is not numeric"
        ) from error
    if not result.is_finite():
        raise ValueError(
            f"Open Images line {line_number} field {field} is not finite"
        )
    return result


def _endpoint(
    row: dict[str, str],
    suffix: str,
    line_number: int,
) -> NormalizedEndpoint:
    endpoint = NormalizedEndpoint(
        label_mid=row[f"LabelName{suffix}"],
        xmin=_decimal(row[f"XMin{suffix}"], f"XMin{suffix}", line_number),
        xmax=_decimal(row[f"XMax{suffix}"], f"XMax{suffix}", line_number),
        ymin=_decimal(row[f"YMin{suffix}"], f"YMin{suffix}", line_number),
        ymax=_decimal(row[f"YMax{suffix}"], f"YMax{suffix}", line_number),
    )
    endpoint.validate()
    return endpoint


def iter_relationship_rows(
    path: str | Path,
) -> Iterator[OpenImagesRelation | SkippedOpenImagesRelation]:
    with Path(path).open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Open Images relationship CSV has no header")
        missing = set(BASE_COLUMNS).difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Open Images relationship CSV is missing columns: {sorted(missing)}"
            )
        relation_column = _relation_column(reader.fieldnames)

        for line_number, row in enumerate(reader, start=2):
            image_id = row["ImageID"]
            predicate = row[relation_column]
            if not image_id or not predicate:
                raise ValueError(
                    f"Open Images line {line_number} has empty image/predicate"
                )

            subject = _endpoint(row, "1", line_number)
            object_ = _endpoint(row, "2", line_number)

            if predicate == "is":
                yield SkippedOpenImagesRelation(
                    image_id=image_id,
                    reason="attribute",
                )
                continue
            if subject == object_:
                yield SkippedOpenImagesRelation(
                    image_id=image_id,
                    reason="self_relation",
                )
                continue

            yield OpenImagesRelation(
                image_id=image_id,
                subject=subject,
                predicate=predicate,
                object=object_,
            )


def scan_relationship_files(
    paths: Sequence[str | Path],
) -> ScanSummary:
    if not paths:
        raise ValueError("at least one Open Images relationship CSV is required")

    total = 0
    object_relationships = 0
    skipped_attributes = 0
    skipped_self_relations = 0
    images: set[str] = set()
    predicates: set[str] = set()
    object_mids: set[str] = set()

    for path in paths:
        for relation in iter_relationship_rows(path):
            total += 1
            if isinstance(relation, SkippedOpenImagesRelation):
                if relation.reason == "attribute":
                    skipped_attributes += 1
                elif relation.reason == "self_relation":
                    skipped_self_relations += 1
                else:
                    raise AssertionError("unreachable Open Images skip reason")
                continue
            object_relationships += 1
            images.add(relation.image_id)
            predicates.add(relation.predicate)
            object_mids.add(relation.subject.label_mid)
            object_mids.add(relation.object.label_mid)

    if object_relationships == 0:
        raise ValueError("Open Images input contains no object relationships")

    return ScanSummary(
        relationships=total,
        object_relationships=object_relationships,
        skipped_attributes=skipped_attributes,
        skipped_self_relations=skipped_self_relations,
        images=tuple(sorted(images)),
        predicates=tuple(sorted(predicates)),
        object_mids=tuple(sorted(object_mids)),
    )


def build_vocabulary(
    summaries: Sequence[ScanSummary],
    class_descriptions: dict[str, str],
) -> RelationVocabulary:
    if not summaries:
        raise ValueError("at least one Open Images scan summary is required")

    predicates: set[str] = set()
    object_mids: set[str] = set()
    for summary in summaries:
        predicates.update(summary.predicates)
        object_mids.update(summary.object_mids)

    missing = sorted(mid for mid in object_mids if mid not in class_descriptions)
    if missing:
        raise ValueError(
            f"Open Images class descriptions are missing MID {missing[0]}"
        )
    objects = sorted({class_descriptions[mid] for mid in object_mids})
    if len(objects) != len(object_mids):
        raise ValueError(
            "Open Images object class descriptions are not one-to-one"
        )
    return RelationVocabulary(
        predicates=tuple(sorted(predicates)),
        object_labels=tuple(objects),
    )


def vocabulary_payload(vocabulary: RelationVocabulary) -> str:
    return (
        json.dumps(
            vocabulary.canonical_payload(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )


def image_id_payload(split: str, image_ids: Iterable[str]) -> str:
    if split not in {"train", "validation", "test"}:
        raise ValueError("Open Images split must be train/validation/test")
    unique = sorted(set(image_ids))
    if not unique:
        raise ValueError("image ID list must not be empty")
    return "".join(f"{split}/{image_id}\n" for image_id in unique)


def _image_path(image_root: Path, image_id: str) -> Path:
    candidates = (
        image_root / f"{image_id}.jpg",
        image_root / f"{image_id}.jpeg",
        image_root / f"{image_id}.png",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected exactly one local image for Open Images ID {image_id}"
        )
    return existing[0]


def convert_relationship_file(
    relationship_csv: str | Path,
    *,
    split: str,
    image_root: str | Path,
    class_descriptions: dict[str, str],
    vocabulary: RelationVocabulary,
) -> tuple[str, dict[str, object]]:
    predicate_index = {
        predicate: index
        for index, predicate in enumerate(vocabulary.predicates)
    }
    allowed_objects = set(vocabulary.object_labels)

    grouped: dict[str, list[OpenImagesRelation]] = {}
    total = 0
    skipped_attributes = 0
    skipped_self_relations = 0
    for relation in iter_relationship_rows(relationship_csv):
        total += 1
        if isinstance(relation, SkippedOpenImagesRelation):
            if relation.reason == "attribute":
                skipped_attributes += 1
            elif relation.reason == "self_relation":
                skipped_self_relations += 1
            else:
                raise AssertionError("unreachable Open Images skip reason")
            continue
        if relation.predicate not in predicate_index:
            raise ValueError(
                f"predicate {relation.predicate} is absent from shared vocabulary"
            )
        grouped.setdefault(relation.image_id, []).append(relation)

    if not grouped:
        raise ValueError("Open Images split contains no object relationships")

    root = Path(image_root)
    records: list[str] = []
    converted_relations = 0
    object_instances = 0

    for image_id in sorted(grouped):
        image_path = _image_path(root, image_id)
        with Image.open(image_path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"image {image_id} has invalid dimensions")

        relations = grouped[image_id]
        endpoints = sorted(
            {
                endpoint
                for relation in relations
                for endpoint in (relation.subject, relation.object)
            }
        )
        endpoint_index = {
            endpoint: index for index, endpoint in enumerate(endpoints)
        }

        boxes: list[list[float]] = []
        labels: list[str] = []
        for endpoint in endpoints:
            description = class_descriptions.get(endpoint.label_mid)
            if description is None:
                raise ValueError(
                    f"Open Images class descriptions are missing MID "
                    f"{endpoint.label_mid}"
                )
            if description not in allowed_objects:
                raise ValueError(
                    f"object label {description} is absent from shared vocabulary"
                )
            boxes.append(
                list(endpoint.pixel_box(width=width, height=height))
            )
            labels.append(description)

        canonical_relations = sorted(
            {
                (
                    endpoint_index[relation.subject],
                    predicate_index[relation.predicate],
                    endpoint_index[relation.object],
                )
                for relation in relations
            }
        )
        if any(subject == object_ for subject, _, object_ in canonical_relations):
            raise ValueError(
                f"Open Images image {image_id} produced a self-relation"
            )

        record = {
            "image": image_path.name,
            "width": width,
            "height": height,
            "boxes_xyxy": boxes,
            "object_labels": labels,
            "relations": [list(value) for value in canonical_relations],
        }
        records.append(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        )
        converted_relations += len(canonical_relations)
        object_instances += len(endpoints)

    output = "".join(records)
    # Validate the exact bytes the caller will write, not a parallel structure.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{split}.jsonl"
        path.write_text(output, encoding="utf-8")
        validated = DatasetManifest.load(path, vocabulary)

    source_path = Path(relationship_csv)
    manifest: dict[str, object] = {
        "schema": "kfcore.openimages-v7-conversion/1",
        "split": split,
        "source_csv": source_path.name,
        "source_sha256": sha256_file(source_path),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "vocabulary_sha256": vocabulary.sha256(),
        "source_rows": total,
        "skipped_attribute_rows": skipped_attributes,
        "skipped_self_relation_rows": skipped_self_relations,
        "images": len(validated.examples),
        "object_instances": object_instances,
        "relations": converted_relations,
    }
    return output, manifest


def select_subset_image_ids(
    relationship_csv: str | Path,
    *,
    max_images: int,
    max_boxes: int,
) -> tuple[str, ...]:
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    if max_boxes < 2:
        raise ValueError("max_boxes must be at least 2")

    endpoints: dict[str, set[NormalizedEndpoint]] = {}
    relation_counts: dict[str, int] = {}
    for relation in iter_relationship_rows(relationship_csv):
        if isinstance(relation, SkippedOpenImagesRelation):
            continue
        endpoints.setdefault(relation.image_id, set()).update(
            (relation.subject, relation.object)
        )
        relation_counts[relation.image_id] = (
            relation_counts.get(relation.image_id, 0) + 1
        )

    eligible = [
        image_id
        for image_id in sorted(endpoints)
        if 2 <= len(endpoints[image_id]) <= max_boxes
        and relation_counts.get(image_id, 0) > 0
    ]
    if len(eligible) < max_images:
        raise ValueError(
            f"only {len(eligible)} eligible Open Images examples; "
            f"requested {max_images}"
        )
    return tuple(eligible[:max_images])


def subset_relationship_csv(
    relationship_csv: str | Path,
    image_ids: Sequence[str],
) -> str:
    selected = set(image_ids)
    if not selected:
        raise ValueError("subset image_ids must not be empty")
    if len(selected) != len(image_ids):
        raise ValueError("subset image_ids must be unique")

    rows: list[tuple[str, ...]] = []
    with Path(relationship_csv).open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Open Images relationship CSV has no header")
        missing = set(BASE_COLUMNS).difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Open Images relationship CSV is missing columns: {sorted(missing)}"
            )
        relation_column = _relation_column(reader.fieldnames)
        for line_number, row in enumerate(reader, start=2):
            if row["ImageID"] not in selected:
                continue
            predicate = row[relation_column]
            if predicate == "is":
                continue
            subject = _endpoint(row, "1", line_number)
            object_ = _endpoint(row, "2", line_number)
            if subject == object_:
                continue
            rows.append(
                tuple(row[column] for column in BASE_COLUMNS)
                + (predicate,)
            )

    present = {row[0] for row in rows}
    missing_ids = sorted(selected - present)
    if missing_ids:
        raise ValueError(
            f"selected Open Images ID has no object relations: {missing_ids[0]}"
        )

    rows.sort()
    import io

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CANONICAL_COLUMNS)
    writer.writerows(rows)
    return buffer.getvalue()


def stable_manifest_json(manifest: dict[str, object]) -> str:
    return (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
