from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
import onnxruntime as ort
from PIL import Image

from prepare_yolox_tiny import (
    INPUT_SIZE,
    QUALIFIED_DECODED_ONNX_SHA256,
    QUALIFIED_PACKAGE_SHA256,
    UPSTREAM_COMMIT,
    UPSTREAM_LICENSE,
    UPSTREAM_REPOSITORY,
    UPSTREAM_SOURCE_SHA256,
    UPSTREAM_TAG,
)


SCHEMA = "kfcore.detector-boxes/1"
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MAX_DETECTIONS = 300
_BORDER_VALUE = 114.0
_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def stable_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_annotation_images(path: str | Path) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid annotation JSON at line {line_number}"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError(
                f"annotation line {line_number} must be an object"
            )
        image = payload.get("image")
        width = payload.get("width")
        height = payload.get("height")
        if (
            not isinstance(image, str)
            or not image
            or not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            raise ValueError(
                f"annotation line {line_number} has invalid image metadata"
            )
        posix = PurePosixPath(image.replace("\\", "/"))
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError("annotation image must stay inside image root")
        name = posix.as_posix()
        if name in seen:
            raise ValueError(f"duplicate annotation image: {name}")
        seen.add(name)
        rows.append((name, width, height))
    if not rows:
        raise ValueError("annotation manifest is empty")
    return rows


def _letterbox_nchw(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("decoded image must be uint8 HWC RGB")
    source_height, source_width, _ = rgb.shape
    destination_width, destination_height = INPUT_SIZE
    scale = min(
        destination_width / float(source_width),
        destination_height / float(source_height),
    )
    output = np.empty(
        (3, destination_height, destination_width),
        dtype=np.float32,
    )

    for y in range(destination_height):
        center_y = y + 0.5
        for x in range(destination_width):
            center_x = x + 0.5
            border = (
                center_x >= source_width * scale
                or center_y >= source_height * scale
            )
            if border:
                value = np.asarray(
                    [_BORDER_VALUE, _BORDER_VALUE, _BORDER_VALUE],
                    dtype=np.float32,
                )
            else:
                source_x = min(
                    max(center_x / scale - 0.5, 0.0),
                    source_width - 1.0,
                )
                source_y = min(
                    max(center_y / scale - 0.5, 0.0),
                    source_height - 1.0,
                )
                x0 = int(math.floor(source_x))
                y0 = int(math.floor(source_y))
                x1 = min(x0 + 1, source_width - 1)
                y1 = min(y0 + 1, source_height - 1)
                fx = source_x - x0
                fy = source_y - y0
                top = (
                    rgb[y0, x0].astype(np.float32) * (1.0 - fx)
                    + rgb[y0, x1].astype(np.float32) * fx
                )
                bottom = (
                    rgb[y1, x0].astype(np.float32) * (1.0 - fx)
                    + rgb[y1, x1].astype(np.float32) * fx
                )
                value = top * (1.0 - fy) + bottom * fy
            output[:, y, x] = (value / 255.0 - _MEAN) / _STD

    return output[None, ...], scale


def _iou(left: tuple[float, float, float, float],
         right: tuple[float, float, float, float]) -> float:
    overlap_width = max(
        0.0,
        min(left[2], right[2]) - max(left[0], right[0]),
    )
    overlap_height = max(
        0.0,
        min(left[3], right[3]) - max(left[1], right[1]),
    )
    intersection = overlap_width * overlap_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _score_summary(
    output: np.ndarray,
    score_threshold: float,
) -> tuple[float, float, int]:
    values = np.asarray(output)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] <= 5:
        raise ValueError("YOLOX output must be [1,A,5+C]")
    objectness = np.asarray(values[0, :, 4], dtype=np.float64)
    class_scores = np.asarray(values[0, :, 5:], dtype=np.float64)
    if not np.isfinite(objectness).all() or not np.isfinite(class_scores).all():
        raise ValueError("YOLOX confidence contains non-finite values")
    best_class = class_scores.max(axis=1)
    fused = objectness * best_class
    return (
        float(objectness.max(initial=0.0)),
        float(fused.max(initial=0.0)),
        int(np.count_nonzero(fused >= score_threshold)),
    )


def _decode(
    output: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    scale: float,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> tuple[list[list[float]], list[float]]:
    values = np.asarray(output)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] <= 5:
        raise ValueError("YOLOX output must be [1,A,5+C]")
    if not np.isfinite(values).all():
        raise ValueError("YOLOX output contains non-finite values")

    candidates: list[
        tuple[float, int, int, tuple[float, float, float, float]]
    ] = []
    for source_index, row in enumerate(values[0]):
        objectness = float(row[4])
        class_scores = np.asarray(row[5:], dtype=np.float64)
        if (
            objectness < 0.0
            or objectness > 1.0
            or np.any(class_scores < 0.0)
            or np.any(class_scores > 1.0)
        ):
            raise ValueError("YOLOX confidence is outside [0,1]")
        best_class = int(np.argmax(class_scores))
        score = objectness * float(class_scores[best_class])
        if score < score_threshold:
            continue

        center_x, center_y, width, height = map(float, row[:4])
        if width <= 0.0 or height <= 0.0:
            raise ValueError("YOLOX box has non-positive extent")
        half_width = width * 0.5
        half_height = height * 0.5
        box = (
            min(max((center_x - half_width) / scale, 0.0), float(source_width)),
            min(max((center_y - half_height) / scale, 0.0), float(source_height)),
            min(max((center_x + half_width) / scale, 0.0), float(source_width)),
            min(max((center_y + half_height) / scale, 0.0), float(source_height)),
        )
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        candidates.append((score, source_index, best_class, box))

    candidates.sort(key=lambda value: (-value[0], value[1]))
    selected: list[
        tuple[float, int, int, tuple[float, float, float, float]]
    ] = []
    for candidate in candidates:
        suppressed = any(
            prior[2] == candidate[2]
            and _iou(prior[3], candidate[3]) > iou_threshold
            for prior in selected
        )
        if suppressed:
            continue
        selected.append(candidate)
        if len(selected) == max_detections:
            break

    boxes = [list(value[3]) for value in selected]
    scores = [value[0] for value in selected]
    return boxes, scores


def run(
    *,
    model_path: str | Path,
    annotations_path: str | Path,
    image_root: str | Path,
    output_path: str | Path,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> dict[str, object]:
    model = Path(model_path)
    if sha256_file(model) != QUALIFIED_DECODED_ONNX_SHA256:
        raise ValueError(
            "detector model is not the qualified YOLOX-Tiny decoded ONNX"
        )
    if (
        not math.isfinite(score_threshold)
        or not 0.0 <= score_threshold <= 1.0
        or not math.isfinite(iou_threshold)
        or not 0.0 <= iou_threshold <= 1.0
        or max_detections <= 0
    ):
        raise ValueError("invalid detector threshold/limit configuration")

    rows = _load_annotation_images(annotations_path)
    root = Path(image_root)
    session = ort.InferenceSession(
        str(model.resolve()),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].shape != [1, 3, INPUT_SIZE[1], INPUT_SIZE[0]]
        or len(outputs) != 1
    ):
        raise ValueError("qualified YOLOX-Tiny ONNX contract drifted")

    lines: list[str] = []
    total_boxes = 0
    max_objectness = 0.0
    max_fused_score = 0.0
    threshold_candidates = 0
    images_with_threshold_candidates = 0
    for image_name, expected_width, expected_height in rows:
        path = root / Path(image_name)
        with Image.open(path) as source:
            rgb_image = source.convert("RGB")
            if rgb_image.size != (expected_width, expected_height):
                raise ValueError(
                    f"image dimensions differ from annotations: {image_name}"
                )
            rgb = np.asarray(rgb_image, dtype=np.uint8)

        tensor, scale = _letterbox_nchw(rgb)
        output = session.run(
            None,
            {inputs[0].name: tensor},
        )[0]
        image_max_objectness, image_max_fused, image_candidates = (
            _score_summary(output, score_threshold)
        )
        max_objectness = max(max_objectness, image_max_objectness)
        max_fused_score = max(max_fused_score, image_max_fused)
        threshold_candidates += image_candidates
        if image_candidates > 0:
            images_with_threshold_candidates += 1
        boxes, scores = _decode(
            output,
            source_width=expected_width,
            source_height=expected_height,
            scale=scale,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        total_boxes += len(boxes)
        payload = {
            "schema": SCHEMA,
            "image": image_name,
            "width": expected_width,
            "height": expected_height,
            "boxes_xyxy": boxes,
            "scores": scores,
        }
        lines.append(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )

    output_file = Path(output_path)
    if output_file.exists():
        raise FileExistsError(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    detector_config = {
        "input_size": list(INPUT_SIZE),
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "max_detections": max_detections,
        "preprocess": (
            "top-left-letterbox-114/rgb/imagenet-mean-std/chw-fp32"
        ),
        "decode": "objectness-times-best-class/class-aware-nms",
    }
    return {
        "schema": "kfcore.yolox-detector-box-manifest-evidence/1",
        "detector_id": "yolox-tiny-coco-0.1.1rc0",
        "detector_model_sha256": QUALIFIED_DECODED_ONNX_SHA256,
        "detector_package_sha256": QUALIFIED_PACKAGE_SHA256,
        "detector_config": detector_config,
        "detector_config_sha256": hashlib.sha256(
            stable_json_bytes(detector_config)
        ).hexdigest(),
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "tag": UPSTREAM_TAG,
            "commit": UPSTREAM_COMMIT,
            "license": UPSTREAM_LICENSE,
            "source_onnx_sha256": UPSTREAM_SOURCE_SHA256,
        },
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "max_detections": max_detections,
        "image_count": len(rows),
        "box_count": total_boxes,
        "raw_score_diagnostics": {
            "max_objectness": max_objectness,
            "max_fused_score": max_fused_score,
            "threshold_candidates_before_nms": threshold_candidates,
            "images_with_threshold_candidates": images_with_threshold_candidates,
        },
        "manifest_sha256": sha256_file(output_file),
        "class_labels_emitted": False,
        "provider": "CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned Apache-2.0 YOLOX-Tiny qualification artifact "
            "and emit class-agnostic kfcore.detector-boxes/1 JSONL."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=DEFAULT_MAX_DETECTIONS,
    )
    parser.add_argument("--evidence-out")
    args = parser.parse_args()

    evidence = run(
        model_path=args.model,
        annotations_path=args.annotations,
        image_root=args.image_root,
        output_path=args.out,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
    )
    if args.evidence_out:
        evidence_path = Path(args.evidence_out)
        if evidence_path.exists():
            raise FileExistsError(evidence_path)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
