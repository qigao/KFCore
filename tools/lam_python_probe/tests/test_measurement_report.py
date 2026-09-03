import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = REPOSITORY_ROOT / "docs" / "design" / "lam-python-prototype-results.json"


def test_measurement_report_has_reproducible_commands_and_core_metrics() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["schema_version"] == 1
    assert report["machine"]["gpu"] == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert report["artifacts"]["lam_one_click_archive"]["sha256"]
    assert report["runs"]["a2e"]["command"]
    assert report["runs"]["a2e"]["frame_count"] == 240
    assert report["runs"]["a2e"]["expression_count"] == 52
    assert report["runs"]["lam"]["launch_command"]
    assert report["runs"]["lam"]["render_frame_count"] == 519
    assert report["runs"]["lam"]["cuda_peak_reserved_bytes"] > 0
    assert report["outputs"]["lam_audio_video"]["sha256"]
    assert report["limitations"]

    a2e = report["runs"]["a2e"]
    working_directory = REPOSITORY_ROOT / a2e["working_directory"]
    resolved_command_paths = [
        (working_directory / a2e["command"][index]).resolve()
        for index in (0, 1, 3)
    ]
    assert resolved_command_paths == [
        (
            REPOSITORY_ROOT
            / ".cache"
            / "lam-prototype"
            / "envs"
            / "lam-a2e"
            / "python.exe"
        ).resolve(),
        (working_directory / "inference.py").resolve(),
        (
            working_directory
            / "configs"
            / "lam_audio2exp_config_streaming.py"
        ).resolve(),
    ]
