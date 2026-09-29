from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import torch
from torch import Tensor


VOCAB_SCHEMA = "kfcore.relation-vocab/1"


@dataclass(frozen=True)
class RelationVocabulary:
    predicates: tuple[str, ...]
    object_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_names(self.predicates, "predicate")
        if self.object_labels:
            _validate_names(self.object_labels, "object")

    @classmethod
    def load(cls, path: str | Path) -> "RelationVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != VOCAB_SCHEMA:
            raise ValueError("unsupported relation vocabulary schema")
        predicates = payload.get("predicates")
        objects = payload.get("objects", [])
        if not isinstance(predicates, list) or not isinstance(objects, list):
            raise ValueError("vocabulary predicates/objects must be arrays")
        return cls(tuple(predicates), tuple(objects))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": VOCAB_SCHEMA,
            "predicates": list(self.predicates),
            "objects": list(self.object_labels),
        }

    def sha256(self) -> str:
        encoded = (
            json.dumps(
                self.canonical_payload(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RelationExample:
    image: str
    width: int
    height: int
    boxes_xyxy: tuple[tuple[float, float, float, float], ...]
    object_labels: tuple[str, ...]
    relations: tuple[tuple[int, int, int], ...]

    @classmethod
    def from_payload(
        cls,
        payload: object,
        vocabulary: RelationVocabulary,
    ) -> "RelationExample":
        if not isinstance(payload, dict):
            raise ValueError("relation example must be a JSON object")

        image = payload.get("image")
        width = payload.get("width")
        height = payload.get("height")
        boxes_raw = payload.get("boxes_xyxy")
        labels_raw = payload.get("object_labels")
        relations_raw = payload.get("relations")

        if not isinstance(image, str) or not image:
            raise ValueError("image must be a non-empty relative path")
        image_path = PurePosixPath(image.replace("\\", "/"))
        if image_path.is_absolute() or ".." in image_path.parts:
            raise ValueError("image must stay inside the dataset root")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("image width must be a positive integer")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError("image height must be a positive integer")
        if not isinstance(boxes_raw, list):
            raise ValueError("boxes_xyxy must be an array")
        if not isinstance(labels_raw, list):
            raise ValueError("object_labels must be an array")
        if len(labels_raw) != len(boxes_raw):
            raise ValueError("object_labels length must match boxes")
        if not isinstance(relations_raw, list):
            raise ValueError("relations must be an array")

        boxes: list[tuple[float, float, float, float]] = []
        for raw in boxes_raw:
            if not isinstance(raw, list) or len(raw) != 4:
                raise ValueError("every box must contain four coordinates")
            values: list[float] = []
            for value in raw:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("box coordinates must be numeric")
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("box coordinates must be finite")
                values.append(number)
            left, top, right, bottom = values
            if (
                left < 0.0
                or top < 0.0
                or right > float(width)
                or bottom > float(height)
                or right <= left
                or bottom <= top
            ):
                raise ValueError("boxes must be positive and inside the image")
            boxes.append((left, top, right, bottom))

        labels: list[str] = []
        for label in labels_raw:
            if not isinstance(label, str) or not label:
                raise ValueError("object labels must be non-empty strings")
            if vocabulary.object_labels and label not in vocabulary.object_labels:
                raise ValueError(f"unknown object label: {label}")
            labels.append(label)

        relations: list[tuple[int, int, int]] = []
        seen_relations: set[tuple[int, int, int]] = set()
        for raw in relations_raw:
            if (
                not isinstance(raw, list)
                or len(raw) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
            ):
                raise ValueError(
                    "every relation must be [subject_index,predicate_index,object_index]"
                )
            subject, predicate, object_ = raw
            if (
                subject < 0
                or subject >= len(boxes)
                or object_ < 0
                or object_ >= len(boxes)
                or subject == object_
            ):
                raise ValueError("relation object indices are invalid")
            if predicate < 0 or predicate >= len(vocabulary.predicates):
                raise ValueError("relation predicate index is invalid")
            relation = (subject, predicate, object_)
            if relation in seen_relations:
                raise ValueError("duplicate ground-truth relation")
            seen_relations.add(relation)
            relations.append(relation)

        return cls(
            image=image_path.as_posix(),
            width=width,
            height=height,
            boxes_xyxy=tuple(boxes),
            object_labels=tuple(labels),
            relations=tuple(relations),
        )


@dataclass(frozen=True)
class DatasetManifest:
    examples: tuple[RelationExample, ...]
    annotations_sha256: str
    vocabulary_sha256: str

    @classmethod
    def load(
        cls,
        annotations_path: str | Path,
        vocabulary: RelationVocabulary,
    ) -> "DatasetManifest":
        path = Path(annotations_path)
        raw_bytes = path.read_bytes()
        examples: list[RelationExample] = []
        for line_number, raw_line in enumerate(
            raw_bytes.decode("utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                examples.append(
                    RelationExample.from_payload(payload, vocabulary)
                )
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid relation example at line {line_number}: {error}"
                ) from error
        if not examples:
            raise ValueError("relation benchmark annotations are empty")
        return cls(
            examples=tuple(examples),
            annotations_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            vocabulary_sha256=vocabulary.sha256(),
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    top_ks: tuple[int, ...] = (20, 50, 100)
    pair_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.top_ks or any(k <= 0 for k in self.top_ks):
            raise ValueError("top_ks must contain positive integers")
        if tuple(sorted(set(self.top_ks))) != self.top_ks:
            raise ValueError("top_ks must be sorted and unique")
        if not math.isfinite(self.pair_weight):
            raise ValueError("pair_weight must be finite")


@dataclass(frozen=True)
class _PairRecord:
    score: float
    positive: bool


def _validate_names(values: Sequence[str], subject: str) -> None:
    if not values:
        raise ValueError(f"{subject} vocabulary must not be empty")
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{subject} names must be non-empty strings")
        if value in seen:
            raise ValueError(f"duplicate {subject} name: {value}")
        seen.add(value)


def _average_precision(
    records: Iterable[_PairRecord],
    total_positives: int,
) -> float:
    if total_positives <= 0:
        raise ValueError("pair AP requires at least one positive ground-truth pair")
    ordered = sorted(records, key=lambda item: item.score, reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, record in enumerate(ordered, start=1):
        if record.positive:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / total_positives


class RelationBenchmark:
    def __init__(
        self,
        predicate_count: int,
        config: BenchmarkConfig = BenchmarkConfig(),
    ) -> None:
        if predicate_count <= 0:
            raise ValueError("predicate_count must be positive")
        self.predicate_count = predicate_count
        self.config = config
        self.example_count = 0
        self.total_gt_pairs = 0
        self.sampled_gt_pairs = 0
        self.total_gt_triplets = 0
        self.sampled_gt_triplets = 0
        self.predicate_top1_hits = 0
        self.pair_records: list[_PairRecord] = []
        self.triplet_hits = {k: 0 for k in config.top_ks}
        self.predicate_support = [0 for _ in range(predicate_count)]
        self.predicate_hits = {
            k: [0 for _ in range(predicate_count)] for k in config.top_ks
        }

    def add(
        self,
        runtime_outputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        relations: Sequence[tuple[int, int, int]],
    ) -> None:
        pred_logits, pair_logits, sub_idx, obj_idx, valid_mask = runtime_outputs
        if pred_logits.ndim != 3 or pred_logits.shape[0] != 1:
            raise ValueError("benchmark expects pred_logits [1,K,V]")
        if pred_logits.shape[2] != self.predicate_count:
            raise ValueError("prediction vocabulary width does not match benchmark")
        expected_pair_shape = pred_logits.shape[:2]
        for value in (pair_logits, sub_idx, obj_idx, valid_mask):
            if value.shape != expected_pair_shape:
                raise ValueError("runtime pair output shapes do not match")

        gt_triplets = set(relations)
        for subject, predicate, object_ in gt_triplets:
            if subject < 0 or object_ < 0 or subject == object_:
                raise ValueError("ground-truth pair index is invalid")
            if predicate < 0 or predicate >= self.predicate_count:
                raise ValueError("ground-truth predicate index is invalid")
        gt_pairs = {(subject, object_) for subject, _, object_ in gt_triplets}

        self.example_count += 1
        self.total_gt_pairs += len(gt_pairs)
        self.total_gt_triplets += len(gt_triplets)
        for _, predicate, _ in gt_triplets:
            self.predicate_support[predicate] += 1

        pair_scores: dict[tuple[int, int], float] = {}
        triplet_scores: dict[tuple[int, int, int], float] = {}
        top1_predicate: dict[tuple[int, int], int] = {}

        for slot in range(pred_logits.shape[1]):
            if not bool(valid_mask[0, slot]):
                continue
            subject = int(sub_idx[0, slot])
            object_ = int(obj_idx[0, slot])
            if subject < 0 or object_ < 0 or subject == object_:
                raise ValueError("prediction contains invalid pair indices")
            pair = (subject, object_)
            pair_score = float(torch.sigmoid(pair_logits[0, slot]).item())
            pair_scores[pair] = max(pair_scores.get(pair, -math.inf), pair_score)

            predicate_values = pred_logits[0, slot]
            best_predicate = int(torch.argmax(predicate_values).item())
            if pair not in top1_predicate:
                top1_predicate[pair] = best_predicate

            for predicate in range(self.predicate_count):
                score = float(
                    (
                        predicate_values[predicate]
                        + self.config.pair_weight * pair_logits[0, slot]
                    ).item()
                )
                triplet = (subject, predicate, object_)
                previous = triplet_scores.get(triplet)
                if previous is None or score > previous:
                    triplet_scores[triplet] = score

        sampled_pairs = set(pair_scores)
        sampled_gt_pairs = sampled_pairs & gt_pairs
        self.sampled_gt_pairs += len(sampled_gt_pairs)

        for pair, score in pair_scores.items():
            self.pair_records.append(
                _PairRecord(score=score, positive=pair in gt_pairs)
            )

        gt_predicates_by_pair: dict[tuple[int, int], set[int]] = {}
        for subject, predicate, object_ in gt_triplets:
            gt_predicates_by_pair.setdefault(
                (subject, object_), set()
            ).add(predicate)

        for pair in sampled_gt_pairs:
            predicates = gt_predicates_by_pair[pair]
            self.sampled_gt_triplets += len(predicates)
            if top1_predicate.get(pair) in predicates:
                self.predicate_top1_hits += 1

        ranked_triplets = sorted(
            triplet_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for k in self.config.top_ks:
            predicted = {
                triplet for triplet, _ in ranked_triplets[:k]
            }
            matched = predicted & gt_triplets
            self.triplet_hits[k] += len(matched)
            for _, predicate, _ in matched:
                self.predicate_hits[k][predicate] += 1

    def report(
        self,
        *,
        annotations_sha256: str = "",
        vocabulary_sha256: str = "",
        train_predicate_support: Sequence[int] | None = None,
    ) -> dict[str, object]:
        if self.example_count <= 0:
            raise ValueError("benchmark contains no evaluated examples")
        if self.total_gt_pairs <= 0 or self.total_gt_triplets <= 0:
            raise ValueError("benchmark requires positive ground-truth relations")

        sampler_recall = self.sampled_gt_pairs / self.total_gt_pairs
        pair_ap = _average_precision(
            self.pair_records,
            self.total_gt_pairs,
        )
        predicate_accuracy = (
            self.predicate_top1_hits / self.sampled_gt_pairs
            if self.sampled_gt_pairs
            else 0.0
        )

        recall_at_k: dict[str, float] = {}
        mean_recall_at_k: dict[str, float] = {}
        per_predicate: list[dict[str, object]] = []

        supported_predicates = [
            index
            for index, support in enumerate(self.predicate_support)
            if support > 0
        ]
        for k in self.config.top_ks:
            recall_at_k[str(k)] = (
                self.triplet_hits[k] / self.total_gt_triplets
            )
            recalls = [
                self.predicate_hits[k][index]
                / self.predicate_support[index]
                for index in supported_predicates
            ]
            mean_recall_at_k[str(k)] = (
                sum(recalls) / len(recalls) if recalls else 0.0
            )

        for predicate, support in enumerate(self.predicate_support):
            item: dict[str, object] = {
                "predicate_index": predicate,
                "support": support,
                "recall": {},
            }
            recalls = item["recall"]
            assert isinstance(recalls, dict)
            for k in self.config.top_ks:
                recalls[str(k)] = (
                    self.predicate_hits[k][predicate] / support
                    if support
                    else None
                )
            per_predicate.append(item)

        predicate_groups: dict[str, object] | None = None
        if train_predicate_support is not None:
            if len(train_predicate_support) != self.predicate_count:
                raise ValueError(
                    "train predicate support length does not match vocabulary"
                )
            train_support: list[int] = []
            for value in train_predicate_support:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        "train predicate support must contain integers"
                    )
                if value < 0:
                    raise ValueError(
                        "train predicate support must be non-negative"
                    )
                train_support.append(value)

            def group_payload(indices: list[int]) -> dict[str, object]:
                validation_support = sum(
                    self.predicate_support[index]
                    for index in indices
                )
                group_mean: dict[str, float | None] = {}
                for k in self.config.top_ks:
                    recalls = [
                        self.predicate_hits[k][index]
                        / self.predicate_support[index]
                        for index in indices
                    ]
                    group_mean[str(k)] = (
                        sum(recalls) / len(recalls)
                        if recalls
                        else None
                    )
                return {
                    "predicate_indices": indices,
                    "predicate_count": len(indices),
                    "validation_triplet_support": validation_support,
                    "mean_recall_at_k": group_mean,
                }

            seen_indices = [
                index
                for index in supported_predicates
                if train_support[index] > 0
            ]
            zero_shot_indices = [
                index
                for index in supported_predicates
                if train_support[index] == 0
            ]
            predicate_groups = {
                "seen": group_payload(seen_indices),
                "train_zero_support": group_payload(
                    zero_shot_indices
                ),
            }

        report = {
            "schema": "kfcore.relation-benchmark/1",
            "examples": self.example_count,
            "annotations_sha256": annotations_sha256,
            "vocabulary_sha256": vocabulary_sha256,
            "pair_weight": self.config.pair_weight,
            "top_ks": list(self.config.top_ks),
            "ground_truth_pairs": self.total_gt_pairs,
            "ground_truth_triplets": self.total_gt_triplets,
            "sampled_ground_truth_pairs": self.sampled_gt_pairs,
            "sampler_recall": sampler_recall,
            "pair_ap": pair_ap,
            "predicate_top1_accuracy_on_sampled_pairs": predicate_accuracy,
            "recall_at_k": recall_at_k,
            "mean_recall_at_k": mean_recall_at_k,
            "per_predicate": per_predicate,
        }
        if predicate_groups is not None:
            report["predicate_groups"] = predicate_groups
        return report


def stable_report_json(report: dict[str, object]) -> str:
    return json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
