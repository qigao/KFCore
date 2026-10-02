from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
from PIL import Image

from benchmark import DatasetManifest, RelationVocabulary
from run_detector_relation_qualification import dataset_image_corpus_sha256
from run_released_detector_relation_qualification import _load_bank
from prepare_released_relsgg import ONNX_SHA256, PREDICATE_BANK_SHA256


FIXTURE_SCHEMA = "kfcore.released-scene-behavior-latency-fixture/1"
QUALITY_REPORT_SCHEMA = "kfcore.released-detector-box-relation-qualification/1"
VOCAB_MAGIC = b"KFRELVO1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require_sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def write_vocabulary(
    path: str | Path,
    vocabulary: RelationVocabulary,
    W: np.ndarray,
    alpha: np.ndarray,
) -> None:
    if W.shape != (len(vocabulary.predicates), 512):
        raise ValueError("vocabulary W must be [V,512]")
    if alpha.shape != (len(vocabulary.predicates),):
        raise ValueError("vocabulary alpha must be [V]")
    with Path(path).open("wb") as stream:
        stream.write(VOCAB_MAGIC)
        stream.write(struct.pack("<II", len(vocabulary.predicates), 512))
        for index, name in enumerate(vocabulary.predicates):
            encoded = name.encode("utf-8")
            if not encoded or len(encoded) > 65535:
                raise ValueError("predicate name length is invalid")
            stream.write(struct.pack("<I", len(encoded)))
            stream.write(encoded)
            stream.write(struct.pack("<f", float(alpha[index])))
            stream.write(
                np.asarray(W[index], dtype="<f4").tobytes(order="C")
            )


def prepare(
    *,
    annotations_path: str | Path,
    image_root: str | Path,
    vocabulary_path: str | Path,
    predicate_bank_path: str | Path,
    quality_report_path: str | Path,
    dataset_evidence_path: str | Path,
    out_dir: str | Path,
) -> dict[str, object]:
    root = Path(out_dir)
    if root.exists():
        raise FileExistsError(root)

    vocabulary = RelationVocabulary.load(vocabulary_path)
    manifest = DatasetManifest.load(annotations_path, vocabulary)
    image_sha = dataset_image_corpus_sha256(manifest, image_root)
    W, alpha, bank_report = _load_bank(predicate_bank_path, vocabulary)

    quality_wrapper = load_json(quality_report_path)
    if quality_wrapper.get("schema") != QUALITY_REPORT_SCHEMA:
        raise ValueError("unsupported released quality report schema")
    relation = quality_wrapper.get("relation")
    dataset = quality_wrapper.get("dataset")
    detector = quality_wrapper.get("detector")
    evaluation = quality_wrapper.get("evaluation")
    quality = quality_wrapper.get("quality")
    for value, name in (
        (relation, "relation"),
        (dataset, "dataset"),
        (detector, "detector"),
        (evaluation, "evaluation"),
        (quality, "quality"),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"quality report {name} block is missing")

    if relation.get("onnx_sha256") != ONNX_SHA256:
        raise ValueError("quality relation ONNX differs from released artifact")
    if relation.get("predicate_bank_sha256") != PREDICATE_BANK_SHA256:
        raise ValueError("quality predicate bank differs from released artifact")
    if relation.get("vocabulary_sha256") != vocabulary.sha256():
        raise ValueError("quality vocabulary differs from fixture vocabulary")
    if dataset.get("annotations_sha256") != manifest.annotations_sha256:
        raise ValueError("quality annotations differ from fixture annotations")
    if dataset.get("image_corpus_sha256") != image_sha:
        raise ValueError("quality image corpus differs from fixture pixels")
    if evaluation.get("pair_weight") != 1.0:
        raise ValueError("released latency fixture requires pair_weight=1.0")
    if evaluation.get("top_ks") != [20, 50, 100]:
        raise ValueError("released latency fixture requires top_k 20/50/100")

    detector_evidence = quality_wrapper["detector"]
    detector_model_sha = _require_sha(
        detector_evidence.get("model_sha256"),
        "detector model SHA",
    )
    detector_config_sha = _require_sha(
        detector_evidence.get("config_sha256"),
        "detector config SHA",
    )
    detector_predictions_sha = _require_sha(
        detector_evidence.get("predictions_sha256"),
        "detector predictions SHA",
    )
    relation_config_sha = _require_sha(
        relation.get("config_sha256"),
        "relation config SHA",
    )

    dataset_evidence = load_json(dataset_evidence_path)
    dataset_info = dataset_evidence.get("dataset")
    if not isinstance(dataset_info, dict):
        raise ValueError("HICO evidence dataset block is missing")
    source_parquet = dataset_info.get("source_parquet")
    if not isinstance(source_parquet, dict):
        raise ValueError("HICO source parquet evidence is missing")
    dataset_revision = dataset_info.get("revision")
    if not isinstance(dataset_revision, str) or not dataset_revision:
        raise ValueError("HICO dataset revision is missing")
    parquet_sha = _require_sha(
        source_parquet.get("sha256"),
        "HICO parquet SHA",
    )

    selection = dataset_evidence.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("images") != len(manifest.examples)
    ):
        raise ValueError("HICO evidence selection differs from annotations")

    root.mkdir(parents=True)
    frame_root = root / "frames"
    frame_root.mkdir()
    frames: list[str] = []
    image_base = Path(image_root).resolve()
    for index, example in enumerate(manifest.examples):
        source = (image_base / Path(example.image)).resolve()
        if not source.is_relative_to(image_base) or not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as image:
            if image.size != (example.width, example.height):
                raise ValueError(
                    f"image dimensions differ for {example.image}"
                )
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        raw_name = f"{index:06d}.bgr"
        raw_path = frame_root / raw_name
        raw_path.write_bytes(bgr.tobytes(order="C"))
        frames.append(
            "\t".join(
                (
                    f"frames/{raw_name}",
                    str(example.width),
                    str(example.height),
                    example.image,
                )
            )
        )
    (root / "frames.tsv").write_text(
        "\n".join(frames) + "\n",
        encoding="utf-8",
    )

    write_vocabulary(root / "vocabulary.bin", vocabulary, W, alpha)

    context = {
        "schema": "kfcore.relation-qualification-context/1",
        "relation_model_sha256": ONNX_SHA256,
        "vocabulary_sha256": vocabulary.sha256(),
        "relation_config_sha256": relation_config_sha,
        "detector_model_sha256": detector_model_sha,
        "detector_config_sha256": detector_config_sha,
        "backend": "onnxruntime",
        "device": "cpu",
        "relation_model_type": "relation.open-vocabulary",
        "detector_id": detector_evidence["id"],
        "max_boxes": 32,
        "vocabulary_size": len(vocabulary.predicates),
        "predicate_bank_sha256": PREDICATE_BANK_SHA256,
        "detector_predictions_sha256": detector_predictions_sha,
        "annotations_sha256": manifest.annotations_sha256,
        "image_corpus_sha256": image_sha,
        "dataset": "HICO-DET test",
        "dataset_revision": dataset_revision,
        "dataset_source_parquet_sha256": parquet_sha,
        "quality_input_contract": "detector-boxes",
        "pair_weight": 1.0,
        "top_ks": [20, 50, 100],
        "tracking_sample_policy": "independent-epoch-warm-then-measure",
    }
    (root / "context.json").write_text(
        json.dumps(context, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    evidence = {
        "schema": FIXTURE_SCHEMA,
        "samples": len(manifest.examples),
        "vocabulary_size": len(vocabulary.predicates),
        "predicate_bank": bank_report,
        "files": {
            "frames_tsv_sha256": sha256_file(root / "frames.tsv"),
            "vocabulary_bin_sha256": sha256_file(root / "vocabulary.bin"),
            "context_sha256": sha256_file(root / "context.json"),
            "quality_sha256": sha256_file(root / "quality.json"),
        },
        "dataset": {
            "revision": dataset_revision,
            "source_parquet_sha256": parquet_sha,
            "annotations_sha256": manifest.annotations_sha256,
            "image_corpus_sha256": image_sha,
        },
        "detector": {
            "id": detector_evidence["id"],
            "model_sha256": detector_model_sha,
            "config_sha256": detector_config_sha,
            "predictions_sha256": detector_predictions_sha,
        },
        "relation": {
            "onnx_sha256": ONNX_SHA256,
            "predicate_bank_sha256": PREDICATE_BANK_SHA256,
            "config_sha256": relation_config_sha,
        },
    }
    (root / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--predicate-bank", required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--dataset-evidence", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    report = prepare(
        annotations_path=args.annotations,
        image_root=args.image_root,
        vocabulary_path=args.vocabulary,
        predicate_bank_path=args.predicate_bank,
        quality_report_path=args.quality_report,
        dataset_evidence_path=args.dataset_evidence,
        out_dir=args.out_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
