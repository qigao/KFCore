from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from dataset import SequenceExample, load_sequences
from model import GESTURE_CLASSES, HIDDEN_SIZE, NUM_LAYERS, TemporalGestureGru


@dataclass(frozen=True)
class Event:
    gesture: int
    start_ns: int
    end_ns: int


def decode_events(
    gestures: Sequence[int],
    phases: Sequence[int],
    timestamps_ns: Sequence[int],
    confidences: Sequence[float] | None = None,
    minimum_confidence: float = 0.0,
) -> List[Event]:
    result: List[Event] = []
    active_gesture = 0
    active_start = 0
    for index, (gesture, phase, timestamp_ns) in enumerate(
        zip(gestures, phases, timestamps_ns)
    ):
        confidence = 1.0 if confidences is None else float(confidences[index])
        if confidence < minimum_confidence or gesture == 0:
            active_gesture = 0
            continue
        if phase == 0:  # idle
            active_gesture = 0
        elif phase == 1:  # start
            active_gesture = gesture
            active_start = timestamp_ns
        elif phase == 2:  # active
            if active_gesture != gesture:
                active_gesture = 0
        elif phase == 3:  # end
            if active_gesture == gesture:
                result.append(Event(gesture, active_start, timestamp_ns))
            active_gesture = 0
        else:
            raise ValueError(f"invalid phase id: {phase}")
    return result


def temporal_iou(left: Event, right: Event) -> float:
    start = max(left.start_ns, right.start_ns)
    end = min(left.end_ns, right.end_ns)
    intersection = max(0, end - start + 1)
    left_size = max(1, left.end_ns - left.start_ns + 1)
    right_size = max(1, right.end_ns - right.start_ns + 1)
    union = left_size + right_size - intersection
    return float(intersection) / float(union)


def match_events(predicted: Sequence[Event], truth: Sequence[Event], minimum_iou: float):
    candidates: List[Tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predicted):
        for truth_index, expected in enumerate(truth):
            if pred.gesture != expected.gesture:
                continue
            score = temporal_iou(pred, expected)
            if score >= minimum_iou:
                candidates.append((score, pred_index, truth_index))
    candidates.sort(reverse=True)
    used_pred = set()
    used_truth = set()
    matches = []
    for score, pred_index, truth_index in candidates:
        if pred_index in used_pred or truth_index in used_truth:
            continue
        used_pred.add(pred_index)
        used_truth.add(truth_index)
        matches.append((predicted[pred_index], truth[truth_index], score))
    return matches, len(predicted) - len(matches), len(truth) - len(matches)


def load_model(checkpoint: Path) -> TemporalGestureGru:
    payload = torch.load(checkpoint, map_location="cpu")
    expected = {
        "feature_count": 78,
        "hidden_size": 64,
        "num_layers": 2,
        "gesture_count": 8,
        "phase_count": 4,
    }
    if payload.get("config") != expected:
        raise ValueError("checkpoint does not match Temporal Gesture GRU V1")
    model = TemporalGestureGru()
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Temporal Gesture GRU V1 events")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.70)
    parser.add_argument("--event-iou", type=float, default=0.50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not 0.0 <= args.minimum_confidence <= 1.0:
        raise SystemExit("minimum-confidence must be within [0,1]")
    if not 0.0 < args.event_iou <= 1.0:
        raise SystemExit("event-iou must be within (0,1]")

    examples = load_sequences(args.dataset)
    if not examples:
        raise SystemExit("dataset contains no sequences")
    model = load_model(args.checkpoint)

    tp = fp = fn = 0
    matched_latencies_ms: List[float] = []
    total_duration_ns = 0
    confusion = [[0 for _ in GESTURE_CLASSES] for _ in GESTURE_CLASSES]

    with torch.no_grad():
        for example in examples:
            features = torch.tensor(example.features, dtype=torch.float32).unsqueeze(0)
            hidden = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE, dtype=torch.float32)
            gesture_logits, phase_logits, _ = model.forward_sequence(features, hidden)
            gesture_probabilities = torch.softmax(gesture_logits[0], dim=-1)
            predicted_gestures = gesture_probabilities.argmax(dim=-1).tolist()
            predicted_confidences = gesture_probabilities.max(dim=-1).values.tolist()
            predicted_phases = phase_logits[0].argmax(dim=-1).tolist()

            truth_events = decode_events(
                example.gesture_labels, example.phase_labels, example.timestamps_ns
            )
            predicted_events = decode_events(
                predicted_gestures,
                predicted_phases,
                example.timestamps_ns,
                predicted_confidences,
                args.minimum_confidence,
            )
            matches, false_positive, false_negative = match_events(
                predicted_events, truth_events, args.event_iou
            )
            tp += len(matches)
            fp += false_positive
            fn += false_negative
            for predicted, expected, _ in matches:
                matched_latencies_ms.append((predicted.end_ns - expected.end_ns) / 1_000_000.0)

            for expected, predicted in zip(example.gesture_labels, predicted_gestures):
                confusion[expected][predicted] += 1
            if len(example.timestamps_ns) > 1:
                total_duration_ns += example.timestamps_ns[-1] - example.timestamps_ns[0]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    total_minutes = total_duration_ns / 60_000_000_000.0
    false_activations_per_minute = fp / total_minutes if total_minutes > 0.0 else None
    absolute_latencies = [abs(value) for value in matched_latencies_ms]

    report = {
        "event_iou_threshold": args.event_iou,
        "minimum_confidence": args.minimum_confidence,
        "true_positive_events": tp,
        "false_positive_events": fp,
        "false_negative_events": fn,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "false_activations_per_minute": false_activations_per_minute,
        "completion_latency_mean_ms": (
            sum(matched_latencies_ms) / len(matched_latencies_ms)
            if matched_latencies_ms
            else None
        ),
        "completion_latency_p95_abs_ms": percentile(absolute_latencies, 0.95),
        "gesture_classes": list(GESTURE_CLASSES),
        "frame_confusion_matrix": confusion,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
