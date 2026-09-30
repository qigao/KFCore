from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from export_onnx import (
    OUTPUT_NAMES,
    export_dynamic_vocabulary_graph,
)
from model import BackboneAdapter, KFRelationModel, RelationModelConfig


class FixtureBackbone(BackboneAdapter):
    hidden_size = 8
    patch_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        with torch.no_grad():
            values = torch.arange(
                self.proj.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(self.proj.weight)
            values = (values.remainder(17.0) - 8.0) / 64.0
            self.proj.weight.copy_(values)

    def forward_taps(
        self,
        image: Tensor,
        taps: tuple[int, ...] | list[int],
    ) -> list[Tensor]:
        base = self.proj(image)
        return [
            base * (1.0 + 0.05 * float(index))
            for index, _ in enumerate(taps)
        ]


def config() -> RelationModelConfig:
    return RelationModelConfig(
        image_size=8,
        max_boxes=4,
        pair_budget=6,
        hidden_dim=16,
        geometry_dim=8,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        tap_indices=(-3, -2, -1),
        pair_evidence_contract="apache",
        pair_sampler_contract="apache",
        relation_context_contract="apache",
        predicate_head_contract="apache",
        apache_context_dropout=0.0,
        apache_box_token_dropout=0.0,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_float32(path: Path, value: Tensor) -> None:
    array = (
        value.detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
        .numpy()
    )
    path.write_bytes(array.tobytes(order="C"))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def decode(
    outputs: list[np.ndarray],
    regions: list[tuple[float, float, float, float, float]],
    *,
    predicate_count: int,
    threshold: float,
    pair_weight: float,
    calibration_a: float,
    calibration_b: float,
    top_k: int,
) -> list[dict[str, int | float]]:
    pred_logits, pair_logits, sub_idx, obj_idx, valid = outputs
    pred = pred_logits[0]
    pair = pair_logits[0]
    sub = sub_idx[0]
    obj = obj_idx[0]
    valid = valid[0].astype(bool)

    candidates: list[tuple[float, int, dict[str, int | float]]] = []
    insertion = 0
    for slot in range(pair.shape[0]):
        if not valid[slot]:
            continue
        subject = int(sub[slot])
        object_ = int(obj[slot])
        if (
            subject < 0
            or object_ < 0
            or subject >= len(regions)
            or object_ >= len(regions)
            or subject == object_
        ):
            continue
        best_predicate = 0
        best_score = -1.0
        for predicate in range(predicate_count):
            fused = calibration_a * (
                float(pred[slot, predicate])
                + pair_weight * float(pair[slot])
            ) + calibration_b
            score = sigmoid(fused)
            if score > best_score:
                best_score = score
                best_predicate = predicate
        if best_score < threshold:
            continue
        rank = (
            best_score
            * regions[subject][4]
            * regions[object_][4]
        )
        edge = {
            "subject": subject,
            "object": object_,
            "predicate": best_predicate,
            "score": best_score,
        }
        candidates.append((rank, insertion, edge))
        insertion += 1

    candidates.sort(key=lambda value: (-value[0], value[1]))
    return [
        edge
        for _, _, edge in candidates[:top_k]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(20260930)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    cfg = config()
    initial_bank = F.normalize(
        torch.tensor(
            [
                [1.0, 0.2, 0.1, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.1, 0.2, 0.0, 0.0],
                [0.1, 0.0, 1.0, 0.0, 0.2, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    model = KFRelationModel(
        FixtureBackbone(),
        initial_bank,
        cfg,
    ).eval()

    # Deterministic BGR source. C++ converts BGR -> RGB NCHW / 255.
    bgr = np.empty((8, 8, 3), dtype=np.uint8)
    for y in range(8):
        for x in range(8):
            bgr[y, x, 0] = (x * 19 + y * 7 + 11) % 256
            bgr[y, x, 1] = (x * 3 + y * 29 + 23) % 256
            bgr[y, x, 2] = (x * 13 + y * 5 + 47) % 256
    (out / "image.bgr").write_bytes(bgr.tobytes(order="C"))

    regions = [
        (0.0, 0.0, 4.0, 4.0, 0.90),
        (4.0, 0.0, 8.0, 4.0, 0.80),
        (2.0, 4.0, 6.0, 8.0, 0.95),
    ]
    (out / "regions.tsv").write_text(
        "".join(
            "\t".join(str(value) for value in row) + "\n"
            for row in regions
        ),
        encoding="utf-8",
    )

    rgb = bgr[..., ::-1].copy()
    image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    boxes = torch.zeros(
        (1, cfg.max_boxes, 4),
        dtype=torch.float32,
    )
    for index, (left, top, right, bottom, _) in enumerate(regions):
        boxes[0, index] = torch.tensor(
            [
                (left + right) / 16.0,
                (top + bottom) / 16.0,
                (right - left) / 8.0,
                (bottom - top) / 8.0,
            ],
            dtype=torch.float32,
        )
    box_counts = torch.tensor([len(regions)], dtype=torch.int64)

    basis = F.normalize(
        torch.tensor(
            [
                [1.0, 0.2, 0.1, 0.0, 0.0, 0.0],
                [0.1, 1.0, 0.2, 0.1, 0.0, 0.0],
                [0.0, 0.1, 1.0, 0.2, 0.1, 0.0],
                [0.2, 0.0, 0.1, 1.0, 0.2, 0.1],
                [0.1, 0.2, 0.0, 0.1, 1.0, 0.2],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    alpha_all = torch.tensor(
        [0.0, 0.25, 0.50, 0.75, 1.0],
        dtype=torch.float32,
    )
    cases = [
        ("000-v1", 1),
        ("001-v3", 3),
        ("002-v5", 5),
        ("003-v3-repeat", 3),
    ]

    model_path = out / "relation.onnx"
    export_dynamic_vocabulary_graph(
        model,
        model_path,
        image,
        boxes,
        box_counts,
        basis[:3],
        alpha_all[:3],
    )

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    vocab_root = out / "vocabs"
    vocab_root.mkdir()
    reference_cases = []
    threshold = 0.0
    pair_weight = 1.0
    calibration_a = 1.0
    calibration_b = 0.0
    top_k = cfg.pair_budget

    for name, count in cases:
        case_dir = vocab_root / name
        case_dir.mkdir()
        W = basis[:count].contiguous()
        alpha = alpha_all[:count].contiguous()
        names = [f"predicate-{index}" for index in range(count)]
        (case_dir / "names.txt").write_text(
            "\n".join(names) + "\n",
            encoding="utf-8",
        )
        write_float32(case_dir / "W.f32", W)
        write_float32(case_dir / "alpha.f32", alpha)

        actual = session.run(
            OUTPUT_NAMES,
            {
                "image": image.numpy(),
                "boxes": boxes.numpy(),
                "box_counts": box_counts.numpy(),
                "W": W.numpy(),
                "alpha": alpha.numpy(),
            },
        )
        edges = decode(
            actual,
            regions,
            predicate_count=count,
            threshold=threshold,
            pair_weight=pair_weight,
            calibration_a=calibration_a,
            calibration_b=calibration_b,
            top_k=top_k,
        )
        reference_cases.append(
            {
                "case": name,
                "v": count,
                "edges": edges,
                "valid_pair_count": int(actual[4][0].astype(bool).sum()),
            }
        )

    config_lines = {
        "width": 8,
        "height": 8,
        "input_size": cfg.image_size,
        "max_boxes": cfg.max_boxes,
        "max_pairs": cfg.pair_budget,
        "query_dim": int(basis.shape[1]),
        "threshold": threshold,
        "pair_weight": pair_weight,
        "calibration_a": calibration_a,
        "calibration_b": calibration_b,
        "top_k": top_k,
    }
    (out / "config.tsv").write_text(
        "".join(
            f"{key}\t{value}\n"
            for key, value in config_lines.items()
        ),
        encoding="utf-8",
    )

    reference = {
        "schema": "kfcore.gtbox-ort-reference/1",
        "model_sha256": sha256(model_path),
        "image_sha256": sha256(out / "image.bgr"),
        "region_count": len(regions),
        "cases": reference_cases,
    }
    (out / "reference.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(reference, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
