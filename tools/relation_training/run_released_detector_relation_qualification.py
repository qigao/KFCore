from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

from benchmark import DatasetManifest, RelationVocabulary
from detector_ceiling import (
    DetectorPredictionManifest,
    DetectorRecoverabilityBenchmark,
    DetectorRecoverabilityConfig,
)
from run_detector_relation_qualification import dataset_image_corpus_sha256
from prepare_released_relsgg import (
    HF_REPOSITORY,
    HF_REVISION,
    ONNX_SHA256,
    PREDICATE_BANK_SHA256,
    TRAINING_GIT_SHA,
    sha256_file,
)


REPORT_SCHEMA = "kfcore.released-detector-box-relation-qualification/1"
DETECTOR_EVIDENCE_SCHEMA = "kfcore.yolox-detector-box-manifest-evidence/1"
IMAGE_SIZE = 448
MAX_BOXES = 32
PAIR_BUDGET = 128
QUERY_DIM = 512


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


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _resize_rgb_like_kfcore(
    rgb: np.ndarray,
    width: int = IMAGE_SIZE,
    height: int = IMAGE_SIZE,
) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("image must be uint8 HWC RGB")
    src_h, src_w, _ = rgb.shape
    if src_h <= 0 or src_w <= 0 or width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    scale_x = np.float32(src_w) / np.float32(width)
    scale_y = np.float32(src_h) / np.float32(height)

    xs = (
        (np.arange(width, dtype=np.float32) + np.float32(0.5))
        * scale_x
        - np.float32(0.5)
    )
    ys = (
        (np.arange(height, dtype=np.float32) + np.float32(0.5))
        * scale_y
        - np.float32(0.5)
    )

    raw_x0 = np.floor(xs).astype(np.int64)
    raw_y0 = np.floor(ys).astype(np.int64)
    x0 = np.clip(raw_x0, 0, src_w - 1)
    y0 = np.clip(raw_y0, 0, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)
    fx = np.clip(
        xs - x0.astype(np.float32),
        np.float32(0.0),
        np.float32(1.0),
    )
    fy = np.clip(
        ys - y0.astype(np.float32),
        np.float32(0.0),
        np.float32(1.0),
    )

    top = (
        rgb[y0[:, None], x0[None, :]].astype(np.float32)
        * (np.float32(1.0) - fx[None, :, None])
        + rgb[y0[:, None], x1[None, :]].astype(np.float32)
        * fx[None, :, None]
    )
    bottom = (
        rgb[y1[:, None], x0[None, :]].astype(np.float32)
        * (np.float32(1.0) - fx[None, :, None])
        + rgb[y1[:, None], x1[None, :]].astype(np.float32)
        * fx[None, :, None]
    )
    resized = (
        top * (np.float32(1.0) - fy[:, None, None])
        + bottom * fy[:, None, None]
    )
    # C++ CpuImageProcessor::resize_bgr stores through uint8_t, which truncates.
    return np.clip(resized, 0.0, 255.0).astype(np.uint8)


def _image_corpus_sha256(
    manifest: DatasetManifest,
    image_root: str | Path,
) -> str:
    root = Path(image_root).resolve()
    digest = hashlib.sha256()
    for name in sorted({example.image for example in manifest.examples}):
        path = (root / Path(name)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _prepare_image(
    image_root: str | Path,
    image_name: str,
    expected_width: int,
    expected_height: int,
) -> np.ndarray:
    path = (Path(image_root).resolve() / Path(image_name)).resolve()
    root = Path(image_root).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        if image.size != (expected_width, expected_height):
            raise ValueError(
                f"decoded image size differs from annotations: {image_name}"
            )
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    resized = _resize_rgb_like_kfcore(rgb)
    nchw = (
        resized.astype(np.float32)
        .transpose(2, 0, 1)[None, ...]
        / np.float32(255.0)
    )
    return np.ascontiguousarray(nchw)


def _prepare_boxes(
    boxes_xyxy: tuple[tuple[float, float, float, float], ...],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes_xyxy) > MAX_BOXES:
        raise ValueError("detector boxes must be capped before ONNX input")
    padded = np.zeros((1, MAX_BOXES, 4), dtype=np.float32)
    for index, (left, top, right, bottom) in enumerate(boxes_xyxy):
        padded[0, index] = (
            ((left + right) * 0.5) / width,
            ((top + bottom) * 0.5) / height,
            (right - left) / width,
            (bottom - top) / height,
        )
    return padded, np.asarray([len(boxes_xyxy)], dtype=np.int64)


def _load_bank(
    path: str | Path,
    vocabulary: RelationVocabulary,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    source = Path(path)
    if sha256_file(source) != PREDICATE_BANK_SHA256:
        raise ValueError("predicate bank SHA-256 differs from released artifact")
    with np.load(source, allow_pickle=True) as bank:
        required = {"names", "W", "alpha"}
        missing = required.difference(bank.files)
        if missing:
            raise ValueError(f"predicate bank is missing fields: {sorted(missing)}")
        names = [str(value) for value in bank["names"]]
        W = np.asarray(bank["W"], dtype=np.float32)
        alpha = np.asarray(bank["alpha"], dtype=np.float32)

    if W.ndim != 2 or W.shape[1] != QUERY_DIM:
        raise ValueError("released predicate W must be [V,512]")
    if alpha.shape != (W.shape[0],) or len(names) != W.shape[0]:
        raise ValueError("released predicate bank arrays disagree")
    if not np.isfinite(W).all() or not np.isfinite(alpha).all():
        raise ValueError("released predicate bank contains non-finite values")

    index = {name: row for row, name in enumerate(names)}
    if len(index) != len(names):
        raise ValueError("released predicate bank contains duplicate names")
    missing_names = [
        name for name in vocabulary.predicates if name not in index
    ]
    if missing_names:
        raise ValueError(
            "qualification vocabulary is outside released predicate bank: "
            + ", ".join(missing_names[:5])
        )

    rows = np.asarray(
        [index[name] for name in vocabulary.predicates],
        dtype=np.int64,
    )
    sliced_W = np.ascontiguousarray(W[rows], dtype=np.float32)
    sliced_alpha = np.ascontiguousarray(alpha[rows], dtype=np.float32)
    return sliced_W, sliced_alpha, {
        "bank_size": len(names),
        "selected_size": len(vocabulary.predicates),
        "selected_names": list(vocabulary.predicates),
    }


def _validate_detector_evidence(
    evidence: dict[str, Any],
    manifest: DetectorPredictionManifest,
) -> tuple[str, str, str]:
    if evidence.get("schema") != DETECTOR_EVIDENCE_SCHEMA:
        raise ValueError("unsupported detector evidence schema")
    if evidence.get("class_labels_emitted") is not False:
        raise ValueError("detector evidence must be class-agnostic")
    if evidence.get("manifest_sha256") != manifest.predictions_sha256:
        raise ValueError("detector evidence manifest hash differs")
    detector_id = evidence.get("detector_id")
    model_sha = evidence.get("detector_model_sha256")
    config_sha = evidence.get("detector_config_sha256")
    for value, name in (
        (detector_id, "detector_id"),
        (model_sha, "detector_model_sha256"),
        (config_sha, "detector_config_sha256"),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"detector evidence {name} is missing")
    return detector_id, model_sha, config_sha


def run(
    *,
    onnx_path: str | Path,
    predicate_bank_path: str | Path,
    metadata_path: str | Path,
    vocabulary_path: str | Path,
    annotations_path: str | Path,
    image_root: str | Path,
    detector_predictions_path: str | Path,
    detector_evidence_path: str | Path,
    iou_threshold: float = 0.5,
    top_ks: tuple[int, ...] = (20, 50, 100),
    pair_weight: float = 1.0,
) -> dict[str, object]:
    onnx_file = Path(onnx_path)
    if sha256_file(onnx_file) != ONNX_SHA256:
        raise ValueError("relation ONNX SHA-256 differs from released artifact")

    metadata = _load_json(metadata_path)
    if (
        metadata.get("img_size") != IMAGE_SIZE
        or metadata.get("max_boxes") != MAX_BOXES
        or metadata.get("final_budget") != PAIR_BUDGET
        or metadata.get("text_dim") != QUERY_DIM
        or metadata.get("vocab_mode") != "input"
    ):
        raise ValueError("released deployment metadata contract drifted")

    vocabulary = RelationVocabulary.load(vocabulary_path)
    manifest = DatasetManifest.load(annotations_path, vocabulary)
    image_corpus_sha256 = _image_corpus_sha256(
        manifest,
        image_root,
    )
    detector_manifest = DetectorPredictionManifest.load(
        detector_predictions_path
    )
    detector_evidence = _load_json(detector_evidence_path)
    detector_id, detector_model_sha, detector_config_sha = (
        _validate_detector_evidence(
            detector_evidence,
            detector_manifest,
        )
    )
    W, alpha, bank_report = _load_bank(
        predicate_bank_path,
        vocabulary,
    )

    session = ort.InferenceSession(
        str(onnx_file.resolve()),
        providers=["CPUExecutionProvider"],
    )
    input_names = {value.name for value in session.get_inputs()}
    output_names = [value.name for value in session.get_outputs()]
    if input_names != {"image", "boxes", "box_counts", "W", "alpha"}:
        raise ValueError("released ONNX input names drifted")
    if output_names != [
        "pred_logits",
        "pair_logits",
        "sub_idx",
        "obj_idx",
        "valid_mask",
    ]:
        raise ValueError("released ONNX output names drifted")

    benchmark_config = DetectorRecoverabilityConfig(
        iou_threshold=iou_threshold,
        top_ks=top_ks,
        pair_weight=pair_weight,
    )
    benchmark = DetectorRecoverabilityBenchmark(
        predicate_count=len(vocabulary.predicates),
        config=benchmark_config,
    )
    detector_by_image = detector_manifest.by_image()
    expected_images = {example.image for example in manifest.examples}
    if set(detector_by_image) != expected_images:
        missing = sorted(expected_images - set(detector_by_image))
        extra = sorted(set(detector_by_image) - expected_images)
        raise ValueError(
            f"detector/GT image set differs: missing={missing[:1]} extra={extra[:1]}"
        )

    for example in manifest.examples:
        raw_detector = detector_by_image[example.image]
        detector = raw_detector.limit(MAX_BOXES)
        if (
            detector.width != example.width
            or detector.height != example.height
        ):
            raise ValueError(
                f"detector/GT dimensions differ: {example.image}"
            )

        image = _prepare_image(
            image_root,
            example.image,
            example.width,
            example.height,
        )
        boxes, box_counts = _prepare_boxes(
            detector.boxes_xyxy,
            detector.width,
            detector.height,
        )
        values = session.run(
            output_names,
            {
                "image": image,
                "boxes": boxes,
                "box_counts": box_counts,
                "W": W,
                "alpha": alpha,
            },
        )
        outputs = tuple(torch.from_numpy(np.asarray(value)) for value in values)
        benchmark.add(
            outputs,  # type: ignore[arg-type]
            example,
            detector,
            detector_boxes_before_cap=len(raw_detector.boxes_xyxy),
        )

    quality = benchmark.report(
        annotations_sha256=manifest.annotations_sha256,
        detector_predictions_sha256=detector_manifest.predictions_sha256,
        detector_id=detector_id,
        detector_model_sha256=detector_model_sha,
        detector_config_sha256=detector_config_sha,
    )

    relation_config = {
        "artifact_kind": "upstream-released-onnx",
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "training_git_sha": TRAINING_GIT_SHA,
        "image_size": IMAGE_SIZE,
        "max_boxes": MAX_BOXES,
        "pair_budget": PAIR_BUDGET,
        "query_dim": QUERY_DIM,
        "pair_weight": pair_weight,
        "iou_threshold": iou_threshold,
        "top_ks": list(top_ks),
        "calibration": metadata.get("calibration"),
        "score_contract": metadata.get("score_contract"),
    }
    return {
        "schema": REPORT_SCHEMA,
        "relation": {
            "artifact_kind": "upstream-released-onnx",
            "onnx_sha256": ONNX_SHA256,
            "predicate_bank_sha256": PREDICATE_BANK_SHA256,
            "metadata_sha256": sha256_file(metadata_path),
            "config_sha256": hashlib.sha256(
                stable_json_bytes(relation_config)
            ).hexdigest(),
            "vocabulary_sha256": vocabulary.sha256(),
            "source_repository": HF_REPOSITORY,
            "source_revision": HF_REVISION,
            "training_git_sha": TRAINING_GIT_SHA,
            "training_reproduction_complete": False,
        },
        "dataset": {
            "annotations_sha256": manifest.annotations_sha256,
            "image_corpus_sha256": image_corpus_sha256,
            "example_count": len(manifest.examples),
        },
        "detector": {
            "id": detector_id,
            "predictions_sha256": detector_manifest.predictions_sha256,
            "model_sha256": detector_model_sha,
            "config_sha256": detector_config_sha,
        },
        "evaluation": {
            "provider": "CPUExecutionProvider",
            "onnxruntime_version": ort.__version__,
            "iou_threshold": iou_threshold,
            "top_ks": list(top_ks),
            "pair_weight": pair_weight,
            "detector_class_labels_enter_relation_inference": False,
            "predicate_bank": bank_report,
        },
        "quality": quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the pinned upstream released RelateAnything ONNX "
            "on class-agnostic detector boxes using #141 ceiling metrics."
        )
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--predicate-bank", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--detector-predictions", required=True)
    parser.add_argument("--detector-evidence", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument(
        "--top-k",
        type=int,
        action="append",
        dest="top_ks",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = run(
        onnx_path=args.onnx,
        predicate_bank_path=args.predicate_bank,
        metadata_path=args.metadata,
        vocabulary_path=args.vocabulary,
        annotations_path=args.annotations,
        image_root=args.image_root,
        detector_predictions_path=args.detector_predictions,
        detector_evidence_path=args.detector_evidence,
        iou_threshold=args.iou_threshold,
        top_ks=tuple(args.top_ks) if args.top_ks else (20, 50, 100),
        pair_weight=args.pair_weight,
    )
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(stable_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
