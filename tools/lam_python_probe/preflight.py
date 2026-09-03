from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
COMMAND_TIMEOUT_SECONDS = 10
LAM_REQUIRED_FILES = (
    Path("app_lam.py"),
    Path("configs/inference/lam-20k-8gpu.yaml"),
)
LAM_CHECKPOINT = Path(
    "model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors"
)
A2E_REQUIRED_FILES = (
    Path("inference.py"),
    Path("configs/lam_audio2exp_config_streaming.py"),
)
A2E_CHECKPOINT = Path("pretrained_models/lam_audio2exp_streaming.tar")


def _command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}

    return {
        "available": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _upstream(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(root.resolve()), "revision": None}
    if not (root / ".git").exists():
        return result

    revision = _command(["git", "-C", str(root), "rev-parse", "HEAD"])
    if revision.get("available"):
        result["revision"] = revision["stdout"]
    else:
        result["revision_error"] = revision.get("error") or revision.get("stderr")
    return result


def _validated_upstream(
    name: str,
    root: Path,
    required_files: tuple[Path, ...],
    checkpoint_relative: Path,
) -> dict[str, Any]:
    for relative_path in required_files:
        required_path = root / relative_path
        if not required_path.is_file():
            raise ValueError(f"{name} required file is missing: {required_path}")

    checkpoint = root / checkpoint_relative
    if not checkpoint.is_file():
        raise ValueError(f"{name} checkpoint is missing: {checkpoint}")
    checkpoint_size = checkpoint.stat().st_size
    if checkpoint_size <= 0:
        raise ValueError(f"{name} checkpoint is empty: {checkpoint}")

    result = _upstream(root)
    result["checkpoint"] = {
        "path": str(checkpoint.resolve()),
        "size_bytes": checkpoint_size,
    }
    result["required_files"] = [
        str((root / relative_path).resolve()) for relative_path in required_files
    ]
    return result


def _torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as error:
        return {"available": False, "error": str(error)}

    runtime: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if runtime["cuda_available"]:
        runtime["device_count"] = torch.cuda.device_count()
        runtime["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return runtime


def collect_preflight(lam_root: Path, a2e_root: Path) -> dict[str, Any]:
    roots = {"LAM": lam_root, "LAM_Audio2Expression": a2e_root}
    for name, root in roots.items():
        if not root.is_dir():
            raise ValueError(f"{name} root is not a directory: {root}")

    disk = shutil.disk_usage(lam_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "python": {
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "upstreams": {
            "lam": _validated_upstream(
                "LAM", lam_root, LAM_REQUIRED_FILES, LAM_CHECKPOINT
            ),
            "a2e": _validated_upstream(
                "LAM_Audio2Expression",
                a2e_root,
                A2E_REQUIRED_FILES,
                A2E_CHECKPOINT,
            ),
        },
        "torch": _torch_runtime(),
        "nvidia_smi": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
        ),
        "disk": {
            "path": str(lam_root.resolve()),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }


def write_preflight(lam_root: Path, a2e_root: Path, output: Path) -> None:
    if output.is_dir():
        raise ValueError(f"output path is a directory: {output}")

    report = collect_preflight(lam_root, a2e_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect LAM prototype preflight data")
    parser.add_argument("--lam-root", type=Path, required=True)
    parser.add_argument("--a2e-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output is not None:
        write_preflight(args.lam_root, args.a2e_root, args.output)
    else:
        report = collect_preflight(args.lam_root, args.a2e_root)
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
