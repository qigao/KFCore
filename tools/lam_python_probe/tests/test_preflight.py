from pathlib import Path

import pytest

from tools.lam_python_probe.preflight import collect_preflight, write_preflight


def test_collect_preflight_reports_required_fields(tmp_path: Path) -> None:
    lam_root = tmp_path / "LAM"
    a2e_root = tmp_path / "LAM_Audio2Expression"
    lam_root.mkdir()
    a2e_root.mkdir()

    report = collect_preflight(lam_root, a2e_root)

    assert report["schema_version"] == 1
    assert report["python"]["version"]
    assert report["upstreams"]["lam"]["path"] == str(lam_root.resolve())
    assert report["upstreams"]["a2e"]["path"] == str(a2e_root.resolve())


@pytest.mark.parametrize("missing_name", ["LAM", "LAM_Audio2Expression"])
def test_collect_preflight_rejects_missing_upstream_root(
    tmp_path: Path, missing_name: str
) -> None:
    lam_root = tmp_path / "LAM"
    a2e_root = tmp_path / "LAM_Audio2Expression"
    if missing_name != "LAM":
        lam_root.mkdir()
    if missing_name != "LAM_Audio2Expression":
        a2e_root.mkdir()

    with pytest.raises(ValueError, match=missing_name):
        collect_preflight(lam_root, a2e_root)


def test_write_preflight_atomically_writes_valid_json(tmp_path: Path) -> None:
    lam_root = tmp_path / "LAM"
    a2e_root = tmp_path / "LAM_Audio2Expression"
    output = tmp_path / "results" / "preflight.json"
    lam_root.mkdir()
    a2e_root.mkdir()

    write_preflight(lam_root, a2e_root, output)

    assert output.is_file()
    assert output.read_text(encoding="utf-8").startswith("{\n")
    assert not output.with_suffix(".json.tmp").exists()


def test_write_preflight_rejects_directory_output(tmp_path: Path) -> None:
    lam_root = tmp_path / "LAM"
    a2e_root = tmp_path / "LAM_Audio2Expression"
    output = tmp_path / "output"
    lam_root.mkdir()
    a2e_root.mkdir()
    output.mkdir()

    with pytest.raises(ValueError, match="output path is a directory"):
        write_preflight(lam_root, a2e_root, output)
