from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from numbers import Real
from pathlib import Path
from typing import Any


EXPRESSION_COUNT = 52
DEFAULT_FPS = 30.0
ARKIT_BLENDSHAPE_NAMES = (
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{label} must be finite")
    return numeric_value


def _audio_duration_seconds(audio_path: Path) -> float:
    if not audio_path.is_file():
        raise ValueError(f"audio input is not a file: {audio_path}")
    try:
        with wave.open(str(audio_path), "rb") as audio:
            frame_rate = audio.getframerate()
            if frame_rate <= 0:
                raise ValueError(f"audio frame rate must be positive: {frame_rate}")
            return audio.getnframes() / frame_rate
    except (EOFError, wave.Error) as error:
        raise ValueError(f"invalid WAV input {audio_path}: {error}") from error


def validate_a2e_output(
    expression_path: Path, audio_path: Path, fps: float = DEFAULT_FPS
) -> dict[str, Any]:
    if not expression_path.is_file():
        raise ValueError(f"expression output is not a file: {expression_path}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive: {fps}")

    try:
        document = json.loads(expression_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid expression JSON {expression_path}: {error}") from error

    if not isinstance(document, dict):
        raise ValueError("expression output root must be an object")
    names = document.get("names")
    if names != list(ARKIT_BLENDSHAPE_NAMES):
        raise ValueError("expression names must match the official ARKit52 order")
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("expression output must contain at least one frame")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("expression metadata must be an object")
    metadata_fps = _finite_number(metadata.get("fps"), "metadata fps")
    if not math.isclose(metadata_fps, fps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"metadata fps does not match expected fps: {metadata_fps}")
    metadata_frame_count = metadata.get("frame_count")
    if (
        isinstance(metadata_frame_count, bool)
        or not isinstance(metadata_frame_count, int)
        or metadata_frame_count != len(frames)
    ):
        raise ValueError("metadata frame_count does not match frames")
    if metadata.get("blendshape_names") != names:
        raise ValueError("metadata blendshape_names do not match names")

    minimum_weight = math.inf
    maximum_weight = -math.inf
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frame {frame_index} must be an object")
        frame_time = _finite_number(frame.get("time"), f"frame {frame_index} time")
        expected_time = frame_index / fps
        if not math.isclose(frame_time, expected_time, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"frame {frame_index} time does not match {expected_time}"
            )
        rotation = frame.get("rotation")
        if not isinstance(rotation, list) or len(rotation) not in (0, 3):
            raise ValueError(
                f"frame {frame_index} rotation must be empty or contain 3 values"
            )
        for rotation_index, value in enumerate(rotation):
            _finite_number(value, f"frame {frame_index} rotation {rotation_index}")
        weights = frame.get("weights")
        if not isinstance(weights, list) or len(weights) != EXPRESSION_COUNT:
            raise ValueError(f"frame {frame_index} must contain {EXPRESSION_COUNT} weights")
        for weight_index, weight in enumerate(weights):
            numeric_weight = _finite_number(
                weight, f"frame {frame_index} weight {weight_index}"
            )
            if not 0.0 <= numeric_weight <= 1.0:
                raise ValueError(
                    f"frame {frame_index} weight {weight_index} is outside [0, 1]"
                )
            minimum_weight = min(minimum_weight, numeric_weight)
            maximum_weight = max(maximum_weight, numeric_weight)

    duration_seconds = _audio_duration_seconds(audio_path)
    expected_frame_count = math.ceil(duration_seconds * fps)
    if len(frames) != expected_frame_count:
        raise ValueError(
            "expression frame count does not match audio duration: "
            f"actual={len(frames)}, expected={expected_frame_count}"
        )

    return {
        "expression_path": str(expression_path.resolve()),
        "audio_path": str(audio_path.resolve()),
        "duration_seconds": duration_seconds,
        "fps": fps,
        "frame_count": len(frames),
        "expected_frame_count": expected_frame_count,
        "expression_count": EXPRESSION_COUNT,
        "minimum_weight": minimum_weight,
        "maximum_weight": maximum_weight,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a LAM-A2E bsData.json output")
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = validate_a2e_output(args.expression, args.audio, args.fps)
    json.dump(report, fp=sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
