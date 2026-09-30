from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from apache_vocab_head import ApacheVocabHead
from checkpoint import config_from_payload, load_payload


INPUT_NAMES = ["W"]
OUTPUT_NAMES = ["alpha"]


class PredicateRoutingGateExport(nn.Module):
    def __init__(self, head: ApacheVocabHead) -> None:
        super().__init__()
        self.head = head

    def forward(self, W: Tensor) -> Tensor:
        return self.head.routing_alpha(W)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gate_from_relation_checkpoint(
    checkpoint: str | Path,
) -> tuple[PredicateRoutingGateExport, int]:
    payload = load_payload(Path(checkpoint))
    config = config_from_payload(payload)
    if config.predicate_head_contract != "apache":
        raise ValueError(
            "predicate routing gate requires an Apache predicate-head checkpoint"
        )
    embeddings = payload["predicate_embeddings"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(
            "relation checkpoint predicate_embeddings must be [V,D]"
        )
    text_dim = int(embeddings.shape[1])
    head = ApacheVocabHead(
        d_model=config.hidden_dim,
        text_dim=text_dim,
    )
    prefix = "apache_vocab_head."
    state = {
        key[len(prefix):]: value
        for key, value in payload["state_dict"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise ValueError(
            "relation checkpoint is missing apache_vocab_head state"
        )
    head.load_state_dict(state, strict=True)
    head.eval()
    return PredicateRoutingGateExport(head).eval(), text_dim


def export_gate(
    model: PredicateRoutingGateExport,
    output_path: str | Path,
    W: Tensor,
    *,
    opset: int = 18,
) -> Tensor:
    if W.ndim != 2 or W.shape[0] <= 0 or W.shape[1] <= 0:
        raise ValueError("routing-gate W must be [V,D]")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference = model(W)
    torch.onnx.export(
        model,
        (W,),
        str(path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "W": {0: "num_predicates"},
            "alpha": {0: "num_predicates"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return reference


def check_gate_parity(
    output_path: str | Path,
    model: PredicateRoutingGateExport,
    banks: list[Tensor],
    *,
    tolerance: float = 1.0e-6,
) -> float:
    import onnx
    import onnxruntime as ort

    if not banks:
        raise ValueError("routing-gate parity requires at least one bank")
    path = Path(output_path)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    if [item.name for item in session.get_inputs()] != INPUT_NAMES:
        raise RuntimeError("routing-gate ONNX input name drifted")
    if [item.name for item in session.get_outputs()] != OUTPUT_NAMES:
        raise RuntimeError("routing-gate ONNX output name drifted")

    worst = 0.0
    width = None
    for W in banks:
        if W.ndim != 2 or W.shape[0] <= 0:
            raise ValueError("routing-gate parity banks must be [V,D]")
        if width is None:
            width = int(W.shape[1])
        elif int(W.shape[1]) != width:
            raise ValueError("routing-gate parity bank widths must match")
        with torch.inference_mode():
            expected = model(W).detach().cpu().numpy()
        actual = session.run(
            OUTPUT_NAMES,
            {"W": W.detach().cpu().numpy()},
        )[0]
        if actual.shape != (int(W.shape[0]),):
            raise RuntimeError("routing-gate output does not follow runtime V")
        delta = float(np.max(np.abs(expected - actual)))
        worst = max(worst, delta)
        if delta > tolerance:
            raise RuntimeError(
                f"routing-gate ONNX parity failed: max abs delta {delta}"
            )
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the Apache relation checkpoint's predicate routing gate."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.out)
    model, text_dim = load_gate_from_relation_checkpoint(checkpoint)

    W = torch.nn.functional.normalize(
        torch.randn(3, text_dim),
        dim=-1,
    )
    export_gate(
        model,
        output,
        W,
        opset=args.opset,
    )

    metadata = {
        "schema": "kfcore.predicate-routing-gate-onnx/1",
        "model_type": "relation.predicate-routing-gate",
        "checkpoint_sha256": sha256(checkpoint),
        "onnx_sha256": sha256(output),
        "embedding_dim": text_dim,
        "vocabulary_dynamic": True,
        "opset": args.opset,
    }
    if args.check:
        banks = [
            W[:1],
            W,
            torch.nn.functional.normalize(
                torch.randn(17, text_dim),
                dim=-1,
            ),
        ]
        metadata["check_max_abs_delta"] = check_gate_parity(
            output,
            model,
            banks,
        )
        metadata["tested_vocabulary_sizes"] = [1, 3, 17]

    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
