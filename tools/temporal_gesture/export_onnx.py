from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from model import (
    FEATURE_COUNT,
    GESTURE_COUNT,
    HIDDEN_SIZE,
    NUM_LAYERS,
    PHASE_COUNT,
    StreamingExportWrapper,
    TemporalGestureGru,
    validate_checkpoint_contract,
)

EXPECTED_INPUTS = {
    "features": [1, FEATURE_COUNT],
    "hidden_in": [NUM_LAYERS, 1, HIDDEN_SIZE],
}
EXPECTED_OUTPUTS = {
    "gesture_logits": [1, GESTURE_COUNT],
    "phase_logits": [1, PHASE_COUNT],
    "hidden_out": [NUM_LAYERS, 1, HIDDEN_SIZE],
}


def _shape(value_info) -> list[int]:
    result = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            raise ValueError(f"dynamic tensor dimension found in {value_info.name}")
        result.append(int(dimension.dim_value))
    return result


def _validate_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)
    inputs = {item.name: _shape(item) for item in model.graph.input}
    outputs = {item.name: _shape(item) for item in model.graph.output}
    if inputs != EXPECTED_INPUTS:
        raise ValueError(f"unexpected ONNX inputs: {inputs}")
    if outputs != EXPECTED_OUTPUTS:
        raise ValueError(f"unexpected ONNX outputs: {outputs}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_model(checkpoint: Path | None, contract_smoke: bool) -> TemporalGestureGru:
    model = TemporalGestureGru()
    if contract_smoke:
        torch.manual_seed(20260915)
        model = TemporalGestureGru()
        return model
    if checkpoint is None:
        raise ValueError("checkpoint is required unless --contract-smoke is used")
    payload = torch.load(checkpoint, map_location="cpu")
    validate_checkpoint_contract(payload)
    model.load_state_dict(payload["model_state"], strict=True)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Export KFCore Temporal Gesture GRU V1 to ONNX")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--contract-smoke", action="store_true")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--package-id", default="temporal-gesture-gru-v1")
    parser.add_argument("--version", default="1.0")
    args = parser.parse_args()

    if args.contract_smoke == (args.checkpoint is not None):
        raise SystemExit("choose exactly one of --checkpoint or --contract-smoke")

    model = _load_model(args.checkpoint, args.contract_smoke)
    model.eval()
    wrapper = StreamingExportWrapper(model).eval()

    args.package_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.package_dir / "temporal_gesture.onnx"
    features = torch.zeros(1, FEATURE_COUNT, dtype=torch.float32)
    hidden = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (features, hidden),
            onnx_path,
            input_names=["features", "hidden_in"],
            output_names=["gesture_logits", "phase_logits", "hidden_out"],
            opset_version=17,
            do_constant_folding=True,
        )

    _validate_onnx(onnx_path)
    digest = _sha256(onnx_path)
    variant = "contract-smoke-untrained" if args.contract_smoke else "trained-v1"
    manifest = {
        "schema": "kfcore.model/1",
        "id": args.package_id,
        "version": args.version,
        "model_type": "gesture.temporal-gru",
        "variant": variant,
        "artifacts": [
            {
                "id": "onnx-cpu",
                "format": "onnx",
                "path": onnx_path.name,
                "flavor": "causal-gru-v1",
                "sha256": digest,
                "backend": "onnxruntime",
                "device": "cpu",
            }
        ],
    }
    (args.package_dir / "model.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.contract_smoke:
        (args.package_dir / "CONTRACT_SMOKE_ONLY.txt").write_text(
            "This package contains deterministic untrained random weights.\n"
            "It exists only to validate KFCore ModelPackage/ORT/GRU runtime wiring.\n"
            "It must never be used to claim gesture recognition quality.\n",
            encoding="utf-8",
        )
        print("warning: exported UNTRAINED contract-smoke package")
    print(f"package: {args.package_dir}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
