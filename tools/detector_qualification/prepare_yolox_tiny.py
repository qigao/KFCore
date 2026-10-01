from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


EVIDENCE_SCHEMA = "kfcore.yolox-qualification-artifact/1"
PACKAGE_SCHEMA = "kfcore.model/1"
UPSTREAM_REPOSITORY = "Megvii-BaseDetection/YOLOX"
UPSTREAM_TAG = "0.1.1rc0"
UPSTREAM_COMMIT = "e1052df71842031413f6030723c3607b839c80ce"
UPSTREAM_LICENSE = "Apache-2.0"
UPSTREAM_SOURCE_SHA256 = (
    "427cc366d34e27ff7a03e2899b5e3671"
    "425c262ea2291f88bb942bc1cc70b0f7"
)
QUALIFIED_DECODED_ONNX_SHA256 = (
    "a09bb9dc5b77553181182535842ab1fc"
    "66d29469f291adec01335433e8d4e858"
)
QUALIFIED_PACKAGE_SHA256 = (
    "7130ba963858c2c00155eedbf698e4e1"
    "1508880893fe5384176247b85e6959f9"
)
MODEL_NAME = "yolox-tiny"
INPUT_SIZE = (416, 416)
CLASS_COUNT = 80
STRIDES = (8, 16, 32)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(payload: object) -> str:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _static_shape(value_info: onnx.ValueInfoProto, name: str) -> tuple[int, ...]:
    tensor = value_info.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT:
        raise ValueError(f"{name} must be FP32")
    result: list[int] = []
    for axis, dim in enumerate(tensor.shape.dim):
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            raise ValueError(f"{name} axis {axis} must be static and positive")
        result.append(int(dim.dim_value))
    return tuple(result)


def _decode_constants(
    height: int,
    width: int,
    *,
    strides: tuple[int, ...] = STRIDES,
) -> tuple[np.ndarray, np.ndarray]:
    grids: list[np.ndarray] = []
    stride_values: list[np.ndarray] = []
    for stride in strides:
        if height % stride != 0 or width % stride != 0:
            raise ValueError("input size must be divisible by every YOLOX stride")
        grid_h = height // stride
        grid_w = width // stride
        y, x = np.meshgrid(
            np.arange(grid_h, dtype=np.float32),
            np.arange(grid_w, dtype=np.float32),
            indexing="ij",
        )
        grid = np.stack((x, y), axis=-1).reshape(1, -1, 2)
        grids.append(grid)
        stride_values.append(
            np.full((1, grid_h * grid_w, 1), float(stride), dtype=np.float32)
        )
    return (
        np.concatenate(grids, axis=1),
        np.concatenate(stride_values, axis=1),
    )


def _replace_terminal_output(
    model: onnx.ModelProto,
    old_name: str,
    new_name: str,
) -> None:
    producers = 0
    for node in model.graph.node:
        for index, value in enumerate(node.output):
            if value == old_name:
                node.output[index] = new_name
                producers += 1
    if producers != 1:
        raise ValueError(
            "YOLOX source graph must have exactly one producer for its output"
        )


def append_yolox_decode(
    source: onnx.ModelProto,
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    if len(source.graph.input) != 1 or len(source.graph.output) != 1:
        raise ValueError("YOLOX qualification source must have one input/output")

    input_shape = _static_shape(source.graph.input[0], "YOLOX input")
    if input_shape != (1, 3, INPUT_SIZE[0], INPUT_SIZE[1]):
        raise ValueError(
            "YOLOX-Tiny source input must be [1,3,416,416]"
        )
    output_shape = _static_shape(source.graph.output[0], "YOLOX output")
    grid, stride = _decode_constants(INPUT_SIZE[0], INPUT_SIZE[1])
    expected_candidates = int(grid.shape[1])
    expected_channels = 5 + CLASS_COUNT
    if output_shape != (1, expected_candidates, expected_channels):
        raise ValueError(
            "YOLOX-Tiny source output must be [1,3549,85]"
        )

    model = onnx.ModelProto()
    model.CopyFrom(source)

    final_name = model.graph.output[0].name
    raw_name = final_name + "__kfcore_raw"
    _replace_terminal_output(model, final_name, raw_name)

    prefix = "__kfcore_yolox_decode_"
    initializer_names = {value.name for value in model.graph.initializer}
    node_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
    }
    reserved = {
        prefix + suffix
        for suffix in (
            "grid",
            "stride",
            "starts_xy",
            "ends_xy",
            "starts_wh",
            "ends_wh",
            "starts_tail",
            "ends_tail",
            "axes",
            "steps",
            "xy",
            "wh",
            "tail",
            "xy_grid",
            "xy_decoded",
            "wh_exp",
            "wh_decoded",
        )
    }
    if (initializer_names | node_names) & reserved:
        raise ValueError("YOLOX source graph collides with KFCore decode names")

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(grid, name=prefix + "grid"),
            numpy_helper.from_array(stride, name=prefix + "stride"),
            numpy_helper.from_array(
                np.asarray([0], dtype=np.int64),
                name=prefix + "starts_xy",
            ),
            numpy_helper.from_array(
                np.asarray([2], dtype=np.int64),
                name=prefix + "ends_xy",
            ),
            numpy_helper.from_array(
                np.asarray([2], dtype=np.int64),
                name=prefix + "starts_wh",
            ),
            numpy_helper.from_array(
                np.asarray([4], dtype=np.int64),
                name=prefix + "ends_wh",
            ),
            numpy_helper.from_array(
                np.asarray([4], dtype=np.int64),
                name=prefix + "starts_tail",
            ),
            numpy_helper.from_array(
                np.asarray([expected_channels], dtype=np.int64),
                name=prefix + "ends_tail",
            ),
            numpy_helper.from_array(
                np.asarray([2], dtype=np.int64),
                name=prefix + "axes",
            ),
            numpy_helper.from_array(
                np.asarray([1], dtype=np.int64),
                name=prefix + "steps",
            ),
        ]
    )

    def slice_node(name: str, starts: str, ends: str, output: str):
        return helper.make_node(
            "Slice",
            [
                raw_name,
                prefix + starts,
                prefix + ends,
                prefix + "axes",
                prefix + "steps",
            ],
            [prefix + output],
            name=prefix + name,
        )

    model.graph.node.extend(
        [
            slice_node("slice_xy", "starts_xy", "ends_xy", "xy"),
            slice_node("slice_wh", "starts_wh", "ends_wh", "wh"),
            slice_node("slice_tail", "starts_tail", "ends_tail", "tail"),
            helper.make_node(
                "Add",
                [prefix + "xy", prefix + "grid"],
                [prefix + "xy_grid"],
                name=prefix + "add_grid",
            ),
            helper.make_node(
                "Mul",
                [prefix + "xy_grid", prefix + "stride"],
                [prefix + "xy_decoded"],
                name=prefix + "scale_xy",
            ),
            helper.make_node(
                "Exp",
                [prefix + "wh"],
                [prefix + "wh_exp"],
                name=prefix + "exp_wh",
            ),
            helper.make_node(
                "Mul",
                [prefix + "wh_exp", prefix + "stride"],
                [prefix + "wh_decoded"],
                name=prefix + "scale_wh",
            ),
            helper.make_node(
                "Concat",
                [
                    prefix + "xy_decoded",
                    prefix + "wh_decoded",
                    prefix + "tail",
                ],
                [final_name],
                axis=2,
                name=prefix + "concat",
            ),
        ]
    )

    onnx.checker.check_model(model)
    evidence = {
        "input_shape": list(input_shape),
        "source_output_shape": list(output_shape),
        "decoded_output_shape": list(output_shape),
        "candidate_count": expected_candidates,
        "class_count": CLASS_COUNT,
        "strides": list(STRIDES),
        "decode_contract": (
            "xy=(raw_xy+grid)*stride;"
            "wh=exp(raw_wh)*stride;"
            "tail=objectness+class_confidences"
        ),
    }
    return model, evidence


def prepare(
    source_onnx: str | Path,
    out_dir: str | Path,
    *,
    expected_source_sha256: str = "",
) -> dict[str, Any]:
    source_path = Path(source_onnx)
    root = Path(out_dir)
    if root.exists():
        raise FileExistsError(root)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_sha = sha256_file(source_path)
    if expected_source_sha256 and source_sha != expected_source_sha256:
        raise ValueError(
            "source YOLOX ONNX SHA-256 differs from the pinned value"
        )

    source = onnx.load(source_path)
    decoded, decode = append_yolox_decode(source)

    root.mkdir(parents=True, exist_ok=False)
    decoded_path = root / "yolox_tiny_decoded.onnx"
    onnx.save(decoded, decoded_path)
    decoded_sha = sha256_file(decoded_path)

    package = {
        "schema": PACKAGE_SCHEMA,
        "id": "yolox-tiny-coco-qualification",
        "version": "0.1.1rc0-kfcore-decode1",
        "model_type": "yolo-detection",
        "artifacts": [
            {
                "id": "ort-cpu",
                "format": "onnx",
                "path": decoded_path.name,
                "flavor": "raw-yolox",
                "sha256": decoded_sha,
                "backend": "onnxruntime",
                "device": "cpu",
            }
        ],
    }
    package_path = root / "model.json"
    package_path.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema": EVIDENCE_SCHEMA,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "tag": UPSTREAM_TAG,
            "commit": UPSTREAM_COMMIT,
            "license": UPSTREAM_LICENSE,
            "model": MODEL_NAME,
            "source_asset": source_path.name,
            "source_sha256": source_sha,
            "source_bytes": source_path.stat().st_size,
        },
        "conversion": decode,
        "artifact": {
            "decoded_onnx": decoded_path.name,
            "decoded_onnx_sha256": decoded_sha,
            "decoded_onnx_bytes": decoded_path.stat().st_size,
            "package": package_path.name,
            "package_sha256": sha256_file(package_path),
            "flavor": "raw-yolox",
        },
    }
    evidence_path = root / "evidence.json"
    evidence_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the pinned Apache-2.0 YOLOX-Tiny qualification artifact "
            "for KFCore's raw-yolox detector flavor."
        )
    )
    parser.add_argument("--source-onnx", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    report = prepare(
        args.source_onnx,
        args.out_dir,
        expected_source_sha256=UPSTREAM_SOURCE_SHA256,
    )
    if (
        report["artifact"]["decoded_onnx_sha256"]
        != QUALIFIED_DECODED_ONNX_SHA256
    ):
        raise RuntimeError(
            "decoded YOLOX-Tiny ONNX differs from the qualified SHA-256"
        )
    if (
        report["artifact"]["package_sha256"]
        != QUALIFIED_PACKAGE_SHA256
    ):
        raise RuntimeError(
            "YOLOX-Tiny model package differs from the qualified SHA-256"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
