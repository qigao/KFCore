#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from make_rtmw_ort_cpu_qualification_fixture import (
    INPUT_H,
    INPUT_W,
    KEYPOINTS,
    SPLIT_RATIO,
    VISIBILITY_BETA,
    VISIBILITY_SIGMA_X,
    VISIBILITY_SIGMA_Y,
    decode,
    preprocess,
    sha256_file,
    visibility,
)


EXPECTED_SOURCE_SIZE = (3281, 4096)

NORMAL_CASES = [
    ("full-body", 1000.0, 650.0, 1200.0, 2900.0),
    ("upper-body", 1000.0, 650.0, 1200.0, 1500.0),
    ("left-boundary", -80.0, 650.0, 1800.0, 2900.0),
    ("extreme-wide", 400.0, 1400.0, 2500.0, 900.0),
]

OCCLUDED_CASES = [
    ("occluded-full-body", 1000.0, 650.0, 1200.0, 2900.0),
]


def prepare_source(image_path: Path, occluded: bool) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode source image: {image_path}")

    height, width = image.shape[:2]
    if (width, height) != EXPECTED_SOURCE_SIZE:
        raise RuntimeError(
            f"unexpected source image size: {(width, height)} "
            f"!= {EXPECTED_SOURCE_SIZE}"
        )

    if occluded:
        # Deterministic synthetic occlusion over the upper-right torso/arm.
        cv2.rectangle(
            image,
            (1600, 1250),
            (2150, 2350),
            (0, 0, 0),
            thickness=-1,
        )

    return np.ascontiguousarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--occluded", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.onnx).resolve()
    image_path = Path(args.image).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)

    source_image_sha = sha256_file(image_path)
    source = prepare_source(image_path, args.occluded)
    source_h, source_w = source.shape[:2]

    source_bgr_path = out / "source.bgr"
    source_bgr_path.write_bytes(source.tobytes(order="C"))
    (out / "source.tsv").write_text(
        f"{source_w}\t{source_h}\n", encoding="utf-8"
    )

    target_model = out / "end2end.onnx"
    try:
        os.link(model_path, target_model)
    except OSError:
        shutil.copy2(model_path, target_model)
    model_sha = sha256_file(target_model)

    semantic_config = {
        "schema": "kfcore.pose-semantic/1",
        "codec": "simcc",
        "decode_visibility": True,
        "visibility_beta": VISIBILITY_BETA,
        "visibility_sigma_x": VISIBILITY_SIGMA_X,
        "visibility_sigma_y": VISIBILITY_SIGMA_Y,
    }
    semantic_path = out / "pose.json"
    semantic_path.write_text(
        json.dumps(semantic_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    semantic_sha = sha256_file(semantic_path)

    package = {
        "schema": "kfcore.model/1",
        "id": "rtmw-l-384x288-real-image-qualification",
        "version": "20231122",
        "model_type": "pose.rtmw",
        "semantic_contract": "pose.coco-wholebody-133",
        "semantic_version": "1",
        "semantic_config": semantic_path.name,
        "semantic_config_sha256": semantic_sha,
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

    cases = OCCLUDED_CASES if args.occluded else NORMAL_CASES
    case_lines = []
    reference_lines = []
    evidence = {
        "schema": "kfcore.rtmw-real-image-fixture/1",
        "model_sha256": model_sha,
        "package_sha256": sha256_file(package_path),
        "semantic_config_sha256": semantic_sha,
        "source_url": args.source_url,
        "source_image_sha256": source_image_sha,
        "source_bgr_sha256": sha256_file(source_bgr_path),
        "source_size": [source_w, source_h],
        "occluded": bool(args.occluded),
        "input_size": [INPUT_W, INPUT_H],
        "simcc_split_ratio": SPLIT_RATIO,
        "keypoint_count": KEYPOINTS,
        "visibility": {
            "beta": VISIBILITY_BETA,
            "sigma_x": VISIBILITY_SIGMA_X,
            "sigma_y": VISIBILITY_SIGMA_Y,
        },
        "cases": [],
    }

    for case_index, box in enumerate(cases):
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

        nchw, geom = preprocess(source, box)
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
        decoded_visibility = visibility(pred_x, pred_y)

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
                        format(float(decoded_visibility[keypoint]), ".9g"),
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
