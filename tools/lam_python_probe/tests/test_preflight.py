from pathlib import Path

import pytest

from tools.lam_python_probe.preflight import collect_preflight, write_preflight


def _create_valid_upstreams(tmp_path: Path) -> tuple[Path, Path]:
    lam_root = tmp_path / "LAM"
    a2e_root = tmp_path / "LAM_Audio2Expression"
    (lam_root / "configs" / "inference").mkdir(parents=True)
    (lam_root / "app_lam.py").write_text("# marker\n", encoding="utf-8")
    (lam_root / "configs" / "inference" / "lam-20k-8gpu.yaml").write_text(
        "# marker\n", encoding="utf-8"
    )
    lam_checkpoint = (
        lam_root
        / "model_zoo"
        / "lam_models"
        / "releases"
        / "lam"
        / "lam-20k"
        / "step_045500"
        / "model.safetensors"
    )
    lam_checkpoint.parent.mkdir(parents=True)
    lam_checkpoint.write_bytes(b"lam-checkpoint")

    (a2e_root / "configs").mkdir(parents=True)
    (a2e_root / "inference.py").write_text("# marker\n", encoding="utf-8")
    (a2e_root / "configs" / "lam_audio2exp_config_streaming.py").write_text(
        "# marker\n", encoding="utf-8"
    )
    a2e_checkpoint = a2e_root / "pretrained_models" / "lam_audio2exp_streaming.tar"
    a2e_checkpoint.parent.mkdir(parents=True)
    a2e_checkpoint.write_bytes(b"a2e-checkpoint")
    return lam_root, a2e_root


def test_collect_preflight_reports_required_fields(tmp_path: Path) -> None:
    lam_root, a2e_root = _create_valid_upstreams(tmp_path)

    report = collect_preflight(lam_root, a2e_root)

    assert report["schema_version"] == 1
    assert report["python"]["version"]
    assert report["upstreams"]["lam"]["path"] == str(lam_root.resolve())
    assert report["upstreams"]["a2e"]["path"] == str(a2e_root.resolve())
    assert report["upstreams"]["lam"]["checkpoint"]["size_bytes"] > 0
    assert report["upstreams"]["a2e"]["checkpoint"]["size_bytes"] > 0


def test_collect_preflight_rejects_empty_upstream_roots(tmp_path: Path) -> None:
    lam_root = tmp_path / "LAM"
    a2e_root = tmp_path / "LAM_Audio2Expression"
    lam_root.mkdir()
    a2e_root.mkdir()

    with pytest.raises(ValueError, match="LAM required file is missing"):
        collect_preflight(lam_root, a2e_root)


def test_collect_preflight_rejects_zero_byte_checkpoint(tmp_path: Path) -> None:
    lam_root, a2e_root = _create_valid_upstreams(tmp_path)
    checkpoint = (
        lam_root
        / "model_zoo"
        / "lam_models"
        / "releases"
        / "lam"
        / "lam-20k"
        / "step_045500"
        / "model.safetensors"
    )
    checkpoint.write_bytes(b"")

    with pytest.raises(ValueError, match="LAM checkpoint is empty"):
        collect_preflight(lam_root, a2e_root)


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
    lam_root, a2e_root = _create_valid_upstreams(tmp_path)
    output = tmp_path / "results" / "preflight.json"

    write_preflight(lam_root, a2e_root, output)

    assert output.is_file()
    assert output.read_text(encoding="utf-8").startswith("{\n")
    assert not output.with_suffix(".json.tmp").exists()


def test_write_preflight_rejects_directory_output(tmp_path: Path) -> None:
    lam_root, a2e_root = _create_valid_upstreams(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(ValueError, match="output path is a directory"):
        write_preflight(lam_root, a2e_root, output)
