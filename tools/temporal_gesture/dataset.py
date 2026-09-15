from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from features import EncodedFeatures, FeatureState, encode_record


@dataclass(frozen=True)
class SequenceExample:
    sequence_id: str
    track_id: int
    subject_id: Optional[str]
    features: List[List[float]]
    gesture_labels: List[int]
    phase_labels: List[int]


def _label(record: Dict, key: str, upper_bound: int) -> int:
    value = int(record[key])
    if value < 0 or value >= upper_bound:
        raise ValueError(f"{key} must be within [0,{upper_bound - 1}]")
    return value


def load_sequences(path: Path) -> List[SequenceExample]:
    grouped: Dict[Tuple[str, int], List[Dict]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            try:
                sequence_id = str(record["sequence_id"])
                track_id = int(record["track_id"])
                timestamp_ns = int(record["timestamp_ns"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid sequence identity") from exc
            if track_id < 0 or timestamp_ns < 0:
                raise ValueError(f"{path}:{line_number}: track_id/timestamp must be non-negative")
            grouped.setdefault((sequence_id, track_id), []).append(record)

    result: List[SequenceExample] = []
    for (sequence_id, track_id), records in grouped.items():
        records.sort(key=lambda item: int(item["timestamp_ns"]))
        subject_values = {str(item["subject_id"]) for item in records if "subject_id" in item}
        if len(subject_values) > 1:
            raise ValueError(f"sequence {sequence_id}/{track_id} contains multiple subject_id values")
        subject_id = next(iter(subject_values)) if subject_values else None

        previous: Optional[FeatureState] = None
        features: List[List[float]] = []
        gestures: List[int] = []
        phases: List[int] = []
        for record in records:
            encoded: EncodedFeatures = encode_record(record, previous)
            previous = encoded.next_state
            features.append(encoded.values)
            gestures.append(_label(record, "gesture_label", 8))
            phases.append(_label(record, "phase_label", 4))

        if not features:
            continue
        result.append(
            SequenceExample(
                sequence_id=sequence_id,
                track_id=track_id,
                subject_id=subject_id,
                features=features,
                gesture_labels=gestures,
                phase_labels=phases,
            )
        )
    return result


def subjects(examples: Sequence[SequenceExample]) -> set[str]:
    return {example.subject_id for example in examples if example.subject_id is not None}
