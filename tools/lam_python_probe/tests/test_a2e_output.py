import json
import math
import wave
from pathlib import Path

import pytest

from tools.lam_python_probe.a2e_output import validate_a2e_output


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
                "names": [f"weight_{index}" for index in range(52)],
                "metadata": {"frames_per_second": 30.0},
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
