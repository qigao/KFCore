from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Sequence

import torch
from torch import Tensor

from benchmark import RelationExample


DETECTOR_BOX_SCHEMA = "kfcore.detector-boxes/1"


@dataclass(frozen=True)
class DetectorPredictionExample:
    image: str
    width: int
    height: int
    boxes_xyxy: tuple[tuple[float, float, float, float], ...]
    scores: tuple[float, ...]

    @classmethod
    def from_payload(cls, payload: object) -> "DetectorPredictionExample":
        if not isinstance(payload, dict):
            raise ValueError("detector prediction must be a JSON object")
        if payload.get("schema") != DETECTOR_BOX_SCHEMA:
            raise ValueError("unsupported detector prediction schema")

        image = payload.get("image")
        width = payload.get("width")
        height = payload.get("height")
        boxes_raw = payload.get("boxes_xyxy")
        scores_raw = payload.get("scores")

        if not isinstance(image, str) or not image:
            raise ValueError("detector image must be a non-empty relative path")
        image_path = PurePosixPath(image.replace("\\", "/"))
        if image_path.is_absolute() or ".." in image_path.parts:
            raise ValueError("detector image must stay inside the dataset root")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("detector width must be a positive integer")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError("detector height must be a positive integer")
        if not isinstance(boxes_raw, list) or not isinstance(scores_raw, list):
            raise ValueError("detector boxes/scores must be arrays")
        if len(boxes_raw) != len(scores_raw):
            raise ValueError("detector boxes/scores length mismatch")

        boxes: list[tuple[float, float, float, float]] = []
        scores: list[float] = []
        for raw_box, raw_score in zip(boxes_raw, scores_raw):
            if not isinstance(raw_box, list) or len(raw_box) != 4:
                raise ValueError("every detector box must contain four coordinates")
            values: list[float] = []
            for value in raw_box:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("detector box coordinates must be numeric")
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("detector box coordinates must be finite")
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
                raise ValueError("detector boxes must be positive and inside image")
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(float(raw_score))
                or float(raw_score) < 0.0
                or float(raw_score) > 1.0
            ):
                raise ValueError("detector scores must be finite within [0,1]")
            boxes.append((left, top, right, bottom))
            scores.append(float(raw_score))

        return cls(
            image=image_path.as_posix(),
            width=width,
            height=height,
            boxes_xyxy=tuple(boxes),
            scores=tuple(scores),
        )

    def limit(self, max_boxes: int) -> "DetectorPredictionExample":
        if max_boxes <= 0:
            raise ValueError("max_boxes must be positive")
        if len(self.boxes_xyxy) <= max_boxes:
            return self
        order = sorted(
            range(len(self.scores)),
            key=lambda index: (-self.scores[index], index),
        )[:max_boxes]
        return DetectorPredictionExample(
            image=self.image,
            width=self.width,
            height=self.height,
            boxes_xyxy=tuple(self.boxes_xyxy[index] for index in order),
            scores=tuple(self.scores[index] for index in order),
        )


@dataclass(frozen=True)
class DetectorPredictionManifest:
    examples: tuple[DetectorPredictionExample, ...]
    predictions_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "DetectorPredictionManifest":
        source = Path(path)
        raw = source.read_bytes()
        examples: list[DetectorPredictionExample] = []
        seen: set[str] = set()
        for line_number, raw_line in enumerate(
            raw.decode("utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                example = DetectorPredictionExample.from_payload(
                    json.loads(raw_line)
                )
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid detector prediction at line {line_number}: {error}"
                ) from error
            if example.image in seen:
                raise ValueError(
                    f"duplicate detector prediction image: {example.image}"
                )
            seen.add(example.image)
            examples.append(example)
        if not examples:
            raise ValueError("detector prediction manifest is empty")
        return cls(
            examples=tuple(examples),
            predictions_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def by_image(self) -> dict[str, DetectorPredictionExample]:
        return {example.image: example for example in self.examples}


@dataclass(frozen=True)
class DetectorRecoverabilityConfig:
    iou_threshold: float = 0.5
    top_ks: tuple[int, ...] = (20, 50, 100)
    pair_weight: float = 1.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.iou_threshold)
            or self.iou_threshold <= 0.0
            or self.iou_threshold > 1.0
        ):
            raise ValueError("iou_threshold must be finite within (0,1]")
        if not self.top_ks or tuple(sorted(set(self.top_ks))) != self.top_ks:
            raise ValueError("top_ks must be sorted and unique")
        if any(k <= 0 for k in self.top_ks):
            raise ValueError("top_ks must contain positive integers")
        if not math.isfinite(self.pair_weight):
            raise ValueError("pair_weight must be finite")


def _box_iou(
    gt_boxes: Sequence[tuple[float, float, float, float]],
    detector_boxes: Sequence[tuple[float, float, float, float]],
) -> Tensor:
    if not gt_boxes or not detector_boxes:
        return torch.zeros(
            (len(gt_boxes), len(detector_boxes)),
            dtype=torch.float32,
        )
    gt = torch.tensor(gt_boxes, dtype=torch.float32)
    det = torch.tensor(detector_boxes, dtype=torch.float32)

    left = torch.maximum(gt[:, None, 0], det[None, :, 0])
    top = torch.maximum(gt[:, None, 1], det[None, :, 1])
    right = torch.minimum(gt[:, None, 2], det[None, :, 2])
    bottom = torch.minimum(gt[:, None, 3], det[None, :, 3])
    intersection = (
        (right - left).clamp_min(0.0)
        * (bottom - top).clamp_min(0.0)
    )
    gt_area = (
        (gt[:, 2] - gt[:, 0])
        * (gt[:, 3] - gt[:, 1])
    )
    det_area = (
        (det[:, 2] - det[:, 0])
        * (det[:, 3] - det[:, 1])
    )
    return intersection / (
        gt_area[:, None] + det_area[None, :] - intersection
    ).clamp_min(1.0e-6)


def _pair_recoverable(
    subject_candidates: Tensor,
    object_candidates: Tensor,
) -> bool:
    subject = subject_candidates.nonzero(as_tuple=True)[0]
    object_ = object_candidates.nonzero(as_tuple=True)[0]
    if subject.numel() == 0 or object_.numel() == 0:
        return False
    return bool(
        (subject[:, None] != object_[None, :]).any()
    )


def _pair_matches(
    detector_subject: int,
    detector_object: int,
    subject_candidates: Tensor,
    object_candidates: Tensor,
) -> bool:
    if detector_subject == detector_object:
        return False
    if (
        detector_subject < 0
        or detector_subject >= subject_candidates.shape[0]
        or detector_object < 0
        or detector_object >= object_candidates.shape[0]
    ):
        return False
    return bool(
        subject_candidates[detector_subject]
        and object_candidates[detector_object]
    )


class DetectorRecoverabilityBenchmark:
    """Detector ceiling + conditional/end-to-end relation metrics.

    Matching is class-agnostic. A GT object is recoverable if any detector box
    reaches the IoU threshold. A directed GT pair is recoverable if subject and
    object each have a candidate detector box and at least one candidate pair
    uses two distinct detector boxes. This is an upper-bound existence test,
    deliberately independent of an arbitrary one-to-one assignment policy.
    """

    def __init__(
        self,
        predicate_count: int,
        config: DetectorRecoverabilityConfig = DetectorRecoverabilityConfig(),
    ) -> None:
        if predicate_count <= 0:
            raise ValueError("predicate_count must be positive")
        self.predicate_count = int(predicate_count)
        self.config = config

        self.examples = 0
        self.detector_boxes_before_cap = 0
        self.detector_boxes_used = 0
        self.gt_objects = 0
        self.recoverable_gt_objects = 0
        self.gt_pairs = 0
        self.recoverable_gt_pairs = 0
        self.sampled_recoverable_pairs = 0
        self.gt_triplets = 0
        self.recoverable_gt_triplets = 0
        self.sampled_recoverable_triplets = 0

        self.triplet_hits = {k: 0 for k in config.top_ks}
        self.predicate_support = [0 for _ in range(predicate_count)]
        self.recoverable_predicate_support = [
            0 for _ in range(predicate_count)
        ]
        self.predicate_hits = {
            k: [0 for _ in range(predicate_count)]
            for k in config.top_ks
        }

    def add(
        self,
        runtime_outputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        gt: RelationExample,
        detector: DetectorPredictionExample,
        *,
        detector_boxes_before_cap: int | None = None,
    ) -> None:
        if detector.image != gt.image:
            raise ValueError("detector/GT image identity mismatch")
        if detector.width != gt.width or detector.height != gt.height:
            raise ValueError("detector/GT image dimensions mismatch")

        pred_logits, pair_logits, sub_idx, obj_idx, valid_mask = runtime_outputs
        if pred_logits.ndim != 3 or pred_logits.shape[0] != 1:
            raise ValueError("runtime pred_logits must be [1,K,V]")
        if pred_logits.shape[2] != self.predicate_count:
            raise ValueError("runtime predicate width mismatch")
        if pair_logits.shape != pred_logits.shape[:2]:
            raise ValueError("runtime pair logits shape mismatch")
        for value in (sub_idx, obj_idx, valid_mask):
            if value.shape != pair_logits.shape:
                raise ValueError("runtime pair metadata shape mismatch")

        detector_count = len(detector.boxes_xyxy)
        self.examples += 1
        self.detector_boxes_before_cap += (
            detector_count
            if detector_boxes_before_cap is None
            else int(detector_boxes_before_cap)
        )
        self.detector_boxes_used += detector_count
        self.gt_objects += len(gt.boxes_xyxy)

        iou = _box_iou(gt.boxes_xyxy, detector.boxes_xyxy)
        candidates = iou >= self.config.iou_threshold
        object_recoverable = candidates.any(dim=1)
        self.recoverable_gt_objects += int(object_recoverable.sum().item())

        gt_triplets = tuple(gt.relations)
        gt_pairs = sorted(
            {(subject, object_) for subject, _, object_ in gt_triplets}
        )
        pair_recoverable: dict[tuple[int, int], bool] = {}
        for subject, object_ in gt_pairs:
            recoverable = _pair_recoverable(
                candidates[subject],
                candidates[object_],
            )
            pair_recoverable[(subject, object_)] = recoverable

        self.gt_pairs += len(gt_pairs)
        recoverable_pairs = {
            pair
            for pair, recoverable in pair_recoverable.items()
            if recoverable
        }
        self.recoverable_gt_pairs += len(recoverable_pairs)

        sampled_detector_pairs: set[tuple[int, int]] = set()
        ranked_predictions: list[tuple[float, int, int, int]] = []
        for slot in range(pred_logits.shape[1]):
            if not bool(valid_mask[0, slot]):
                continue
            detector_subject = int(sub_idx[0, slot])
            detector_object = int(obj_idx[0, slot])
            if (
                detector_subject < 0
                or detector_subject >= detector_count
                or detector_object < 0
                or detector_object >= detector_count
                or detector_subject == detector_object
            ):
                raise ValueError(
                    "runtime pair index is outside detector boxes"
                )
            sampled_detector_pairs.add(
                (detector_subject, detector_object)
            )
            for predicate in range(self.predicate_count):
                score = float(
                    (
                        pred_logits[0, slot, predicate]
                        + self.config.pair_weight
                        * pair_logits[0, slot]
                    ).item()
                )
                ranked_predictions.append(
                    (
                        score,
                        detector_subject,
                        predicate,
                        detector_object,
                    )
                )

        sampled_pairs: set[tuple[int, int]] = set()
        for subject, object_ in recoverable_pairs:
            for detector_subject, detector_object in sampled_detector_pairs:
                if _pair_matches(
                    detector_subject,
                    detector_object,
                    candidates[subject],
                    candidates[object_],
                ):
                    sampled_pairs.add((subject, object_))
                    break
        self.sampled_recoverable_pairs += len(sampled_pairs)

        recoverable_triplets: set[tuple[int, int, int]] = set()
        sampled_recoverable_triplets: set[tuple[int, int, int]] = set()
        for subject, predicate, object_ in gt_triplets:
            self.gt_triplets += 1
            self.predicate_support[predicate] += 1
            if pair_recoverable[(subject, object_)]:
                recoverable_triplets.add(
                    (subject, predicate, object_)
                )
                self.recoverable_gt_triplets += 1
                self.recoverable_predicate_support[predicate] += 1
                if (subject, object_) in sampled_pairs:
                    sampled_recoverable_triplets.add(
                        (subject, predicate, object_)
                    )
                    self.sampled_recoverable_triplets += 1

        ranked_predictions.sort(
            key=lambda item: item[0],
            reverse=True,
        )
        for k in self.config.top_ks:
            unmatched = set(gt_triplets)
            hits: set[tuple[int, int, int]] = set()
            for _, detector_subject, predicate, detector_object in ranked_predictions[:k]:
                best: tuple[float, tuple[int, int, int]] | None = None
                for triplet in unmatched:
                    subject, gt_predicate, object_ = triplet
                    if gt_predicate != predicate:
                        continue
                    if not _pair_matches(
                        detector_subject,
                        detector_object,
                        candidates[subject],
                        candidates[object_],
                    ):
                        continue
                    quality = float(
                        iou[subject, detector_subject]
                        * iou[object_, detector_object]
                    )
                    if (
                        best is None
                        or quality > best[0]
                        or (
                            quality == best[0]
                            and triplet < best[1]
                        )
                    ):
                        best = (quality, triplet)
                if best is not None:
                    unmatched.remove(best[1])
                    hits.add(best[1])

            self.triplet_hits[k] += len(hits)
            for _, predicate, _ in hits:
                self.predicate_hits[k][predicate] += 1

    def report(
        self,
        *,
        annotations_sha256: str = "",
        detector_predictions_sha256: str = "",
        detector_id: str = "",
        detector_model_sha256: str = "",
        detector_config_sha256: str = "",
    ) -> dict[str, object]:
        if self.examples <= 0:
            raise ValueError("detector benchmark contains no examples")
        if self.gt_pairs <= 0 or self.gt_triplets <= 0:
            raise ValueError("detector benchmark requires GT relations")

        object_ceiling = (
            self.recoverable_gt_objects / self.gt_objects
            if self.gt_objects
            else 0.0
        )
        pair_ceiling = self.recoverable_gt_pairs / self.gt_pairs
        sampler_conditional = (
            self.sampled_recoverable_pairs / self.recoverable_gt_pairs
            if self.recoverable_gt_pairs
            else 0.0
        )
        sampler_end_to_end = (
            self.sampled_recoverable_pairs / self.gt_pairs
        )

        conditional_recall: dict[str, float | None] = {}
        end_to_end_recall: dict[str, float] = {}
        conditional_mean_recall: dict[str, float | None] = {}
        end_to_end_mean_recall: dict[str, float] = {}

        supported = [
            index
            for index, support in enumerate(self.predicate_support)
            if support > 0
        ]
        recoverable_supported = [
            index
            for index, support in enumerate(
                self.recoverable_predicate_support
            )
            if support > 0
        ]

        for k in self.config.top_ks:
            conditional_recall[str(k)] = (
                self.triplet_hits[k] / self.recoverable_gt_triplets
                if self.recoverable_gt_triplets
                else None
            )
            end_to_end_recall[str(k)] = (
                self.triplet_hits[k] / self.gt_triplets
            )
            conditional_values = [
                self.predicate_hits[k][index]
                / self.recoverable_predicate_support[index]
                for index in recoverable_supported
            ]
            conditional_mean_recall[str(k)] = (
                sum(conditional_values) / len(conditional_values)
                if conditional_values
                else None
            )
            end_values = [
                self.predicate_hits[k][index]
                / self.predicate_support[index]
                for index in supported
            ]
            end_to_end_mean_recall[str(k)] = (
                sum(end_values) / len(end_values)
                if end_values
                else 0.0
            )

        max_k = self.config.top_ks[-1]
        detector_miss = (
            self.gt_triplets
            - self.recoverable_gt_triplets
        )
        sampler_miss = (
            self.recoverable_gt_triplets
            - self.sampled_recoverable_triplets
        )
        predicate_miss = (
            self.sampled_recoverable_triplets
            - self.triplet_hits[max_k]
        )
        recovered = self.triplet_hits[max_k]
        if (
            detector_miss
            + sampler_miss
            + predicate_miss
            + recovered
            != self.gt_triplets
        ):
            raise RuntimeError(
                "detector failure decomposition does not sum to GT triplets"
            )

        return {
            "schema": "kfcore.detector-relation-ceiling/1",
            "examples": self.examples,
            "annotations_sha256": annotations_sha256,
            "detector_predictions_sha256": detector_predictions_sha256,
            "detector": {
                "id": detector_id,
                "model_sha256": detector_model_sha256,
                "config_sha256": detector_config_sha256,
                "iou_threshold": self.config.iou_threshold,
                "boxes_before_cap": self.detector_boxes_before_cap,
                "boxes_used": self.detector_boxes_used,
            },
            "ground_truth_objects": self.gt_objects,
            "recoverable_ground_truth_objects": self.recoverable_gt_objects,
            "object_recoverability_ceiling": object_ceiling,
            "ground_truth_directed_pairs": self.gt_pairs,
            "recoverable_directed_pairs": self.recoverable_gt_pairs,
            "directed_pair_recoverability_ceiling": pair_ceiling,
            "sampled_recoverable_pairs": self.sampled_recoverable_pairs,
            "sampler_recall_conditional_on_recoverable_pairs": (
                sampler_conditional
            ),
            "sampler_pair_recall_end_to_end": sampler_end_to_end,
            "ground_truth_triplets": self.gt_triplets,
            "recoverable_ground_truth_triplets": (
                self.recoverable_gt_triplets
            ),
            "sampled_recoverable_triplets": (
                self.sampled_recoverable_triplets
            ),
            "conditional_recall_at_k": conditional_recall,
            "end_to_end_recall_at_k": end_to_end_recall,
            "conditional_mean_recall_at_k": conditional_mean_recall,
            "end_to_end_mean_recall_at_k": end_to_end_mean_recall,
            "failure_decomposition_at_max_k": {
                "max_k": max_k,
                "detector_miss": detector_miss,
                "sampler_miss": sampler_miss,
                "predicate_miss": predicate_miss,
                "recovered": recovered,
            },
        }
