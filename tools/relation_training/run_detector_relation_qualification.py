from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch

from benchmark import DatasetManifest, RelationVocabulary
from checkpoint import config_from_payload, load_payload
from detector_ceiling import (
    DetectorPredictionManifest,
    DetectorRecoverabilityConfig,
)
from model import HFDinoV3Backbone, KFRelationModel
from training import evaluate_detector_boxes


REPORT_SCHEMA = "kfcore.detector-box-relation-qualification/1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if (
        len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


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


def dataset_image_corpus_sha256(
    manifest: DatasetManifest,
    image_root: str | Path,
) -> str:
    root = Path(image_root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    names = sorted({example.image for example in manifest.examples})
    digest = hashlib.sha256()
    for name in names:
        path = root / Path(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def restore_relation_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[KFRelationModel, dict[str, Any]]:
    payload = load_payload(checkpoint_path)
    config = config_from_payload(payload)
    backbone_name = str(payload["backbone_model"])
    if not backbone_name:
        raise ValueError("relation checkpoint backbone_model must not be empty")

    backbone = HFDinoV3Backbone.from_pretrained(
        backbone_name,
        train_backbone=False,
    )
    embeddings = payload["predicate_embeddings"]
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("relation checkpoint predicate_embeddings must be a tensor")

    model = KFRelationModel(
        backbone,
        embeddings.float(),
        config,
    )
    model.load_state_dict(
        payload["state_dict"],
        strict=True,
    )
    model = model.to(device)
    model.eval()
    return model, payload


def build_report(
    *,
    checkpoint_path: str | Path,
    checkpoint_payload: dict[str, Any],
    vocabulary: RelationVocabulary,
    manifest: DatasetManifest,
    image_corpus_sha256: str,
    detector_manifest: DetectorPredictionManifest,
    detector_id: str,
    detector_model_sha256: str,
    detector_config_sha256: str,
    iou_threshold: float,
    top_ks: tuple[int, ...],
    device: torch.device,
    quality: dict[str, object],
) -> dict[str, object]:
    checkpoint_predicates = checkpoint_payload.get("predicates")
    if (
        not isinstance(checkpoint_predicates, list)
        or checkpoint_predicates != list(vocabulary.predicates)
    ):
        raise ValueError(
            "checkpoint predicate order differs from relation vocabulary"
        )

    config_payload = checkpoint_payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("checkpoint config must be an object")

    config_sha256 = hashlib.sha256(
        stable_json_bytes(config_payload)
    ).hexdigest()

    checkpoint_sha256 = sha256_file(checkpoint_path)
    vocabulary_sha256 = vocabulary.sha256()

    if quality.get("schema") != "kfcore.detector-relation-ceiling/1":
        raise ValueError("unexpected detector relation quality schema")
    if quality.get("annotations_sha256") != manifest.annotations_sha256:
        raise ValueError("quality annotations hash differs from GT manifest")
    if (
        quality.get("detector_predictions_sha256")
        != detector_manifest.predictions_sha256
    ):
        raise ValueError("quality detector predictions hash differs from manifest")

    detector_block = quality.get("detector")
    if not isinstance(detector_block, dict):
        raise ValueError("quality detector provenance is missing")
    if detector_block.get("id") != detector_id:
        raise ValueError("quality detector id differs from requested detector")
    if detector_block.get("model_sha256") != detector_model_sha256:
        raise ValueError("quality detector model hash differs from requested detector")
    if detector_block.get("config_sha256") != detector_config_sha256:
        raise ValueError("quality detector config hash differs from requested detector")

    return {
        "schema": REPORT_SCHEMA,
        "relation": {
            "checkpoint_sha256": checkpoint_sha256,
            "backbone_model": str(checkpoint_payload["backbone_model"]),
            "config_sha256": config_sha256,
            "vocabulary_sha256": vocabulary_sha256,
            "predicate_count": len(vocabulary.predicates),
            "checkpoint_predicate_count": len(checkpoint_predicates),
        },
        "dataset": {
            "annotations_sha256": manifest.annotations_sha256,
            "image_corpus_sha256": image_corpus_sha256,
            "example_count": len(manifest.examples),
        },
        "detector": {
            "id": detector_id,
            "predictions_sha256": detector_manifest.predictions_sha256,
            "model_sha256": detector_model_sha256,
            "config_sha256": detector_config_sha256,
        },
        "evaluation": {
            "device": str(device),
            "iou_threshold": iou_threshold,
            "top_ks": list(top_ks),
            "detector_class_labels_enter_relation_inference": False,
        },
        "quality": quality,
    }


def run_qualification(
    *,
    checkpoint_path: str | Path,
    vocabulary_path: str | Path,
    annotations_path: str | Path,
    image_root: str | Path,
    detector_predictions_path: str | Path,
    detector_id: str,
    detector_model_sha256: str,
    detector_config_sha256: str,
    iou_threshold: float = 0.5,
    top_ks: tuple[int, ...] = (20, 50, 100),
    device_name: str = "cpu",
    model_loader: Callable[..., tuple[KFRelationModel, dict[str, Any]]] = (
        restore_relation_model
    ),
) -> dict[str, object]:
    if not detector_id:
        raise ValueError("detector_id must not be empty")
    detector_model_sha256 = _require_sha256(
        detector_model_sha256,
        "detector_model_sha256",
    )
    detector_config_sha256 = _require_sha256(
        detector_config_sha256,
        "detector_config_sha256",
    )

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA qualification but CUDA is unavailable")

    vocabulary = RelationVocabulary.load(vocabulary_path)
    manifest = DatasetManifest.load(
        annotations_path,
        vocabulary,
    )
    detector_manifest = DetectorPredictionManifest.load(
        detector_predictions_path
    )
    image_hash = dataset_image_corpus_sha256(
        manifest,
        image_root,
    )

    model, payload = model_loader(
        checkpoint_path,
        device=device,
    )
    checkpoint_predicates = payload.get("predicates")
    if checkpoint_predicates != list(vocabulary.predicates):
        raise ValueError(
            "checkpoint predicate order differs from relation vocabulary"
        )

    config = DetectorRecoverabilityConfig(
        iou_threshold=iou_threshold,
        top_ks=top_ks,
    )
    quality = evaluate_detector_boxes(
        model,
        manifest,
        detector_manifest,
        image_root=image_root,
        device=device,
        benchmark_config=config,
        detector_id=detector_id,
        detector_model_sha256=detector_model_sha256,
        detector_config_sha256=detector_config_sha256,
    )

    return build_report(
        checkpoint_path=checkpoint_path,
        checkpoint_payload=payload,
        vocabulary=vocabulary,
        manifest=manifest,
        image_corpus_sha256=image_hash,
        detector_manifest=detector_manifest,
        detector_id=detector_id,
        detector_model_sha256=detector_model_sha256,
        detector_config_sha256=detector_config_sha256,
        iou_threshold=iou_threshold,
        top_ks=top_ks,
        device=device,
        quality=quality,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Apache-reference relation checkpoint on class-agnostic "
            "detector boxes and emit detector/relation qualification evidence."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--detector-predictions", required=True)
    parser.add_argument("--detector-id", required=True)
    parser.add_argument("--detector-model-sha256", required=True)
    parser.add_argument("--detector-config-sha256", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--top-k",
        type=int,
        action="append",
        dest="top_ks",
        help="Repeat for each requested recall K; defaults to 20/50/100.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    top_ks = tuple(args.top_ks) if args.top_ks else (20, 50, 100)
    report = run_qualification(
        checkpoint_path=args.checkpoint,
        vocabulary_path=args.vocabulary,
        annotations_path=args.annotations,
        image_root=args.image_root,
        detector_predictions_path=args.detector_predictions,
        detector_id=args.detector_id,
        detector_model_sha256=args.detector_model_sha256,
        detector_config_sha256=args.detector_config_sha256,
        iou_threshold=args.iou_threshold,
        top_ks=top_ks,
        device_name=args.device,
    )

    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(stable_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
