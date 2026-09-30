from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch

from benchmark import (
    DatasetManifest,
    RelationVocabulary,
)
from checkpoint import (
    config_from_payload,
    load_payload,
)
from detector_ceiling import (
    DetectorPredictionManifest,
    DetectorRecoverabilityConfig,
)
from model import (
    HFDinoV3Backbone,
    KFRelationModel,
)
from training import evaluate_detector_boxes


QUALIFICATION_SCHEMA = "kfcore.detector-relation-qualification/1"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: str, subject: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(
            f"{subject} must be a 64-character SHA-256 hex string"
        )
    return value.lower()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA device requested but torch.cuda.is_available() is false"
        )
    return device


def restore_relation_model(
    checkpoint_path: str | Path,
    *,
    backbone_override: str = "",
    device: torch.device,
) -> tuple[KFRelationModel, dict[str, Any]]:
    payload = load_payload(checkpoint_path)
    config = config_from_payload(payload)
    backbone_name = (
        backbone_override
        if backbone_override
        else str(payload["backbone_model"])
    )
    backbone = HFDinoV3Backbone.from_pretrained(
        backbone_name,
        train_backbone=False,
    )
    model = KFRelationModel(
        backbone,
        payload["predicate_embeddings"].float(),
        config,
    )
    model.load_state_dict(
        payload["state_dict"],
        strict=True,
    )
    model.to(device)
    model.eval()
    return model, payload


def build_qualification_report(
    *,
    checkpoint_path: str | Path,
    payload: dict[str, Any],
    vocabulary: RelationVocabulary,
    metrics: dict[str, object],
    backbone_override: str = "",
) -> dict[str, object]:
    predicates = payload.get("predicates")
    if not isinstance(predicates, list):
        raise ValueError(
            "relation checkpoint predicates must be a list"
        )
    if tuple(str(value) for value in predicates) != vocabulary.predicates:
        raise ValueError(
            "relation checkpoint predicate order does not match vocabulary"
        )

    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            "relation checkpoint config must be an object"
        )
    extra = payload.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError(
            "relation checkpoint extra metadata must be an object"
        )

    backbone_model = (
        backbone_override
        if backbone_override
        else str(payload["backbone_model"])
    )
    return {
        "schema": QUALIFICATION_SCHEMA,
        "relation": {
            "checkpoint_sha256": sha256(checkpoint_path),
            "backbone_model": backbone_model,
            "checkpoint_config": dict(config),
            "predicate_count": len(vocabulary.predicates),
            "vocabulary_sha256": vocabulary.sha256(),
            "checkpoint_extra_keys": sorted(
                str(key) for key in extra
            ),
        },
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the KFCore Apache-reference relation model on detector "
            "boxes and emit detector ceiling plus conditional/end-to-end "
            "relation metrics."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--detector-predictions", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--backbone",
        default="",
        help=(
            "Optional local/HF backbone override. Defaults to checkpoint "
            "backbone provenance."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[20, 50, 100],
    )
    parser.add_argument(
        "--pair-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument("--detector-id", required=True)
    parser.add_argument(
        "--detector-model-sha256",
        required=True,
    )
    parser.add_argument(
        "--detector-config-sha256",
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite qualification report: {output_path}"
        )

    detector_model_sha = validate_sha256(
        args.detector_model_sha256,
        "detector model SHA-256",
    )
    detector_config_sha = validate_sha256(
        args.detector_config_sha256,
        "detector config SHA-256",
    )

    vocabulary = RelationVocabulary.load(
        args.vocabulary
    )
    manifest = DatasetManifest.load(
        args.annotations,
        vocabulary,
    )
    detector_manifest = DetectorPredictionManifest.load(
        args.detector_predictions
    )

    device = resolve_device(args.device)
    model, payload = restore_relation_model(
        args.checkpoint,
        backbone_override=args.backbone,
        device=device,
    )

    checkpoint_predicates = tuple(
        str(value)
        for value in payload["predicates"]
    )
    if checkpoint_predicates != vocabulary.predicates:
        raise ValueError(
            "checkpoint predicate order does not match vocabulary"
        )

    config = DetectorRecoverabilityConfig(
        iou_threshold=float(args.iou_threshold),
        top_ks=tuple(int(value) for value in args.top_k),
        pair_weight=float(args.pair_weight),
    )
    metrics = evaluate_detector_boxes(
        model,
        manifest,
        detector_manifest,
        image_root=args.image_root,
        device=device,
        benchmark_config=config,
        detector_id=args.detector_id,
        detector_model_sha256=detector_model_sha,
        detector_config_sha256=detector_config_sha,
    )

    report = build_qualification_report(
        checkpoint_path=args.checkpoint,
        payload=payload,
        vocabulary=vocabulary,
        metrics=metrics,
        backbone_override=args.backbone,
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
