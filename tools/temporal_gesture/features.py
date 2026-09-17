from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

FEATURE_COUNT = 78
PALM_MCP_INDICES = (5, 9, 13, 17)
NANOSECONDS_PER_SECOND = 1_000_000_000.0

POSE_INDEX = {
    "unknown": 0,
    "open": 1,
    "closed": 2,
    "pointer": 3,
}
HANDEDNESS_VALUE = {
    "left": -1.0,
    "unknown": 0.0,
    "right": 1.0,
}


@dataclass(frozen=True)
class FeatureState:
    timestamp_ns: int
    wrist_x_normalized: float
    wrist_y_normalized: float


@dataclass(frozen=True)
class EncodedFeatures:
    values: List[float]
    next_state: FeatureState


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _unit(value: float, name: str) -> float:
    value = _finite(value, name)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be within [0,1]")
    return value


def _landmarks(record: Dict) -> List[List[float]]:
    raw = record.get("landmarks")
    if not isinstance(raw, Sequence) or len(raw) != 21:
        raise ValueError("landmarks must contain exactly 21 xyz triplets")
    result: List[List[float]] = []
    for index, point in enumerate(raw):
        if not isinstance(point, Sequence) or len(point) != 3:
            raise ValueError(f"landmark {index} must contain exactly xyz")
        result.append([
            _finite(point[0], f"landmark[{index}].x"),
            _finite(point[1], f"landmark[{index}].y"),
            _finite(point[2], f"landmark[{index}].z"),
        ])
    return result


def encode_record(record: Dict, previous: Optional[FeatureState] = None) -> EncodedFeatures:
    width = int(record["image_width"])
    height = int(record["image_height"])
    timestamp_ns = int(record["timestamp_ns"])
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if timestamp_ns < 0:
        raise ValueError("timestamp_ns must be non-negative")

    landmarks = _landmarks(record)
    landmark_confidence = _unit(record["landmark_confidence"], "landmark_confidence")
    palm_confidence = _unit(record["palm_confidence"], "palm_confidence")
    handedness = str(record["handedness"]).lower()
    if handedness not in HANDEDNESS_VALUE:
        raise ValueError("handedness must be left, right, or unknown")
    static_pose = str(record["static_pose"]).lower()
    if static_pose not in POSE_INDEX:
        raise ValueError("static_pose must be unknown, open, closed, or pointer")

    wrist = landmarks[0]
    palm_distances = []
    for index in PALM_MCP_INDICES:
        dx = landmarks[index][0] - wrist[0]
        dy = landmarks[index][1] - wrist[1]
        palm_distances.append(math.sqrt(dx * dx + dy * dy))
    scale = sum(palm_distances) / len(palm_distances)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("palm scale must be positive")

    middle_mcp = landmarks[9]
    orientation_x = middle_mcp[0] - wrist[0]
    orientation_y = middle_mcp[1] - wrist[1]
    if orientation_x == 0.0 and orientation_y == 0.0:
        raise ValueError("palm orientation vector must be non-zero")
    orientation = math.atan2(orientation_y, orientation_x)

    wrist_x = wrist[0] / float(width)
    wrist_y = wrist[1] / float(height)
    image_diagonal = math.sqrt(float(width * width + height * height))

    dt = 0.0
    velocity_x = 0.0
    velocity_y = 0.0
    if previous is not None:
        if timestamp_ns <= previous.timestamp_ns:
            raise ValueError("timestamp must increase within a sequence")
        dt = (timestamp_ns - previous.timestamp_ns) / NANOSECONDS_PER_SECOND
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("delta time must be positive and finite")
        velocity_x = (wrist_x - previous.wrist_x_normalized) / dt
        velocity_y = (wrist_y - previous.wrist_y_normalized) / dt

    values: List[float] = []
    mirror = -1.0 if handedness == "left" else 1.0
    for point in landmarks:
        values.extend((
            mirror * (point[0] - wrist[0]) / scale,
            (point[1] - wrist[1]) / scale,
            (point[2] - wrist[2]) / scale,
        ))

    values.extend((
        wrist_x,
        wrist_y,
        velocity_x,
        velocity_y,
        scale / image_diagonal,
        math.sin(orientation),
        math.cos(orientation),
        HANDEDNESS_VALUE[handedness],
        landmark_confidence,
        palm_confidence,
        dt,
    ))

    pose_index = POSE_INDEX[static_pose]
    values.extend(1.0 if index == pose_index else 0.0 for index in range(4))

    if len(values) != FEATURE_COUNT:
        raise RuntimeError(f"feature encoder produced {len(values)} values, expected {FEATURE_COUNT}")
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"encoded feature {index} is not finite")

    return EncodedFeatures(
        values=values,
        next_state=FeatureState(timestamp_ns, wrist_x, wrist_y),
    )
