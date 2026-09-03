import json
import math
import wave
from pathlib import Path

import pytest

from tools.lam_python_probe.a2e_output import validate_a2e_output


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


def _write_audio(path: Path, *, sample_rate: int = 16_000, samples: int = 800) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0\0" * samples)


def _write_output(path: Path, frames: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "names": list(ARKIT_BLENDSHAPE_NAMES),
                "metadata": {
                    "fps": 30.0,
                    "frame_count": len(frames),
                    "blendshape_names": list(ARKIT_BLENDSHAPE_NAMES),
                },
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def test_validate_a2e_output_reports_contract_metrics(tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    output = tmp_path / "bsData.json"
    _write_audio(audio)
    frames = [
        {"weights": [0.25] * 52, "time": index / 30.0, "rotation": []}
        for index in range(2)
    ]
    _write_output(output, frames)

    report = validate_a2e_output(output, audio)

    assert report["frame_count"] == 2
    assert report["expected_frame_count"] == 2
    assert report["expression_count"] == 52
    assert report["minimum_weight"] == 0.25
    assert report["maximum_weight"] == 0.25


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([0.5] * 51, "52 weights"),
        ([math.nan] + [0.5] * 51, "finite"),
        ([1.01] + [0.5] * 51, "outside"),
    ],
)
def test_validate_a2e_output_rejects_invalid_weights(
    tmp_path: Path, weights: list[float], message: str
) -> None:
    audio = tmp_path / "input.wav"
    output = tmp_path / "bsData.json"
    _write_audio(audio, samples=1)
    _write_output(output, [{"weights": weights, "time": 0.0, "rotation": []}])

    with pytest.raises(ValueError, match=message):
        validate_a2e_output(output, audio)


def test_validate_a2e_output_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    output = tmp_path / "bsData.json"
    _write_audio(audio, samples=800)
    _write_output(output, [{"weights": [0.5] * 52, "time": 0.0, "rotation": []}])

    with pytest.raises(ValueError, match="frame count"):
        validate_a2e_output(output, audio)


def test_validate_a2e_output_rejects_reordered_names(tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    output = tmp_path / "bsData.json"
    _write_audio(audio, samples=1)
    _write_output(
        output,
        [{"weights": [0.5] * 52, "time": 0.0, "rotation": []}],
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    document["names"][0], document["names"][1] = (
        document["names"][1],
        document["names"][0],
    )
    output.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="official ARKit52 order"):
        validate_a2e_output(output, audio)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fps", 24.0, "metadata fps"),
        ("frame_count", 2, "metadata frame_count"),
        ("blendshape_names", ["wrong"] * 52, "metadata blendshape_names"),
    ],
)
def test_validate_a2e_output_rejects_inconsistent_metadata(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    audio = tmp_path / "input.wav"
    output = tmp_path / "bsData.json"
    _write_audio(audio, samples=1)
    _write_output(
        output,
        [{"weights": [0.5] * 52, "time": 0.0, "rotation": []}],
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    document["metadata"][field] = value
    output.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_a2e_output(output, audio)


@pytest.mark.parametrize(
    ("frame_update", "message"),
    [
        ({"time": 0.25}, "frame 0 time"),
        ({"rotation": [0.0, math.nan, 0.0]}, "rotation 1 must be finite"),
        ({"rotation": [0.0, 0.0]}, "rotation must be empty or contain 3"),
    ],
)
def test_validate_a2e_output_rejects_invalid_frame_metadata(
    tmp_path: Path, frame_update: dict[str, object], message: str
) -> None:
    audio = tmp_path / "input.wav"
    output = tmp_path / "bsData.json"
    _write_audio(audio, samples=1)
    frame: dict[str, object] = {
        "weights": [0.5] * 52,
        "time": 0.0,
        "rotation": [],
    }
    frame.update(frame_update)
    _write_output(output, [frame])

    with pytest.raises(ValueError, match=message):
        validate_a2e_output(output, audio)
