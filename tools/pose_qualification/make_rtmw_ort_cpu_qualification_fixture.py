#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


INPUT_W = 288
INPUT_H = 384
SPLIT_RATIO = 2.0
KEYPOINTS = 133
SOURCE_W = 640
SOURCE_H = 480
BBOX_PADDING = 1.25
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

CASES = [
    ("normal", 123.25, 66.75, 190.5, 280.25),
    ("left-boundary", -18.5, 52.25, 188.0, 300.0),
    ("wide", 64.0, 141.5, 430.0, 122.0),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_bgr() -> np.ndarray:
    y, x = np.indices((SOURCE_H, SOURCE_W), dtype=np.int32)
    b = 16 + (x * 96) // (SOURCE_W - 1) + (y * 48) // (SOURCE_H - 1)
    g = 24 + (x * 48) // (SOURCE_W - 1) + (y * 96) // (SOURCE_H - 1)
    r = 32 + (x * 80) // (SOURCE_W - 1) + (y * 80) // (SOURCE_H - 1)
    return np.stack([b, g, r], axis=-1).astype(np.uint8)


def geometry(box):
    _, x, y, width, height = box
    center_x = x + width * 0.5
    center_y = y + height * 0.5
    scale_width = width * BBOX_PADDING
    scale_height = height * BBOX_PADDING
    aspect = INPUT_W / INPUT_H
    if scale_width > scale_height * aspect:
        scale_height = scale_width / aspect
    else:
        scale_width = scale_height * aspect
    return center_x, center_y, scale_width, scale_height


def preprocess(image: np.ndarray, box):
    center_x, center_y, scale_width, scale_height = geometry(box)
    dst_to_src = np.asarray(
        [
            [scale_width / INPUT_W, 0.0, center_x - scale_width * 0.5],
            [0.0, scale_height / INPUT_H, center_y - scale_height * 0.5],
        ],
        dtype=np.float32,
    )
    src_to_dst = cv2.invertAffineTransform(dst_to_src)
    crop = cv2.warpAffine(
        image,
        src_to_dst,
        (INPUT_W, INPUT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rgb = crop[:, :, ::-1].astype(np.float32) / np.float32(255.0)
    normalized = (rgb - MEAN) / STD
    nchw = np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)
    return nchw, (center_x, center_y, scale_width, scale_height)


def decode(simcc_x: np.ndarray, simcc_y: np.ndarray, geom):
    x_index = np.argmax(simcc_x, axis=1)
    y_index = np.argmax(simcc_y, axis=1)
    max_x = np.max(simcc_x, axis=1)
    max_y = np.max(simcc_y, axis=1)
    confidence = np.minimum(max_x, max_y).astype(np.float32)

    model_x = x_index.astype(np.float32) / np.float32(SPLIT_RATIO)
    model_y = y_index.astype(np.float32) / np.float32(SPLIT_RATIO)
    invalid = confidence <= 0.0
    model_x[invalid] = -1.0
    model_y[invalid] = -1.0

    center_x, center_y, scale_width, scale_height = geom
    source_x = model_x / np.float32(INPUT_W) * np.float32(scale_width)
    source_x += np.float32(center_x - scale_width * 0.5)
    source_y = model_y / np.float32(INPUT_H) * np.float32(scale_height)
    source_y += np.float32(center_y - scale_height * 0.5)
    source_x[invalid] = -1.0
    source_y[invalid] = -1.0

    return model_x, model_y, source_x, source_y, confidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    model_path = Path(args.onnx).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)

    target_model = out / "end2end.onnx"
    target_model.write_bytes(model_path.read_bytes())
    model_sha = sha256_file(target_model)

    package = {
        "schema": "kfcore.model/1",
        "id": "rtmw-l-384x288-released-qualification",
        "version": "20231122",
        "model_type": "pose.rtmw",
        "semantic_contract": "pose.coco-wholebody-133",
        "semantic_version": "1",
        "artifacts": [
            {
                "id": "ort-cpu",
                "format": "onnx",
                "path": target_model.name,
                "sha256": model_sha,
                "backend": "onnxruntime",
                "device": "cpu",
            }
        ],
    }
    package_path = out / "model.json"
    package_path.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    session = ort.InferenceSession(
        str(target_model),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "input":
        raise RuntimeError("released RTMW reference must expose input")
    if list(inputs[0].shape) != ["batch", 3, INPUT_H, INPUT_W]:
        raise RuntimeError(f"unexpected RTMW input shape: {inputs[0].shape}")
    if [value.name for value in outputs] != ["simcc_x", "simcc_y"]:
        raise RuntimeError("released RTMW reference must expose simcc_x/simcc_y")

    image = synthetic_bgr()
    case_lines = []
    reference_lines = []
    evidence = {
        "schema": "kfcore.rtmw-ort-cpu-fixture/1",
        "model_sha256": model_sha,
        "package_sha256": sha256_file(package_path),
        "onnxruntime_version": ort.__version__,
        "opencv_version": cv2.__version__,
        "input_size": [INPUT_W, INPUT_H],
        "simcc_split_ratio": SPLIT_RATIO,
        "keypoint_count": KEYPOINTS,
        "source_size": [SOURCE_W, SOURCE_H],
        "cases": [],
    }

    for case_index, box in enumerate(CASES):
        label, x, y, width, height = box
        case_lines.append(
            "\t".join(
                [
                    str(case_index),
                    label,
                    format(x, ".9g"),
                    format(y, ".9g"),
                    format(width, ".9g"),
                    format(height, ".9g"),
                ]
            )
        )
        nchw, geom = preprocess(image, box)
        pred_x, pred_y = session.run(
            ["simcc_x", "simcc_y"],
            {"input": nchw},
        )
        pred_x = np.asarray(pred_x[0], dtype=np.float32)
        pred_y = np.asarray(pred_y[0], dtype=np.float32)
        if pred_x.shape != (KEYPOINTS, INPUT_W * 2):
            raise RuntimeError(f"unexpected simcc_x shape: {pred_x.shape}")
        if pred_y.shape != (KEYPOINTS, INPUT_H * 2):
            raise RuntimeError(f"unexpected simcc_y shape: {pred_y.shape}")

        model_x, model_y, source_x, source_y, score = decode(
            pred_x, pred_y, geom
        )

        preprocess_path = out / f"preprocess-{case_index}.f32"
        simcc_x_path = out / f"simcc-x-{case_index}.f32"
        simcc_y_path = out / f"simcc-y-{case_index}.f32"
        nchw.astype("<f4").tofile(preprocess_path)
        pred_x.astype("<f4").tofile(simcc_x_path)
        pred_y.astype("<f4").tofile(simcc_y_path)

        for keypoint in range(KEYPOINTS):
            reference_lines.append(
                "\t".join(
                    [
                        str(case_index),
                        str(keypoint),
                        format(float(model_x[keypoint]), ".9g"),
                        format(float(model_y[keypoint]), ".9g"),
                        format(float(source_x[keypoint]), ".9g"),
                        format(float(source_y[keypoint]), ".9g"),
                        format(float(score[keypoint]), ".9g"),
                    ]
                )
            )

        evidence["cases"].append(
            {
                "index": case_index,
                "label": label,
                "bbox": [x, y, width, height],
                "geometry": list(geom),
                "preprocess_sha256": sha256_file(preprocess_path),
                "simcc_x_sha256": sha256_file(simcc_x_path),
                "simcc_y_sha256": sha256_file(simcc_y_path),
            }
        )

    (out / "cases.tsv").write_text(
        "\n".join(case_lines) + "\n", encoding="utf-8"
    )
    (out / "reference.tsv").write_text(
        "\n".join(reference_lines) + "\n", encoding="utf-8"
    )
    evidence["cases_sha256"] = sha256_file(out / "cases.tsv")
    evidence["reference_sha256"] = sha256_file(out / "reference.tsv")
    (out / "fixture-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
