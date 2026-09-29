from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import RelationVocabulary
from make_predicate_embeddings import tensor_sha256


ONTOLOGY_SCHEMA = "kfcore.predicate-semantic-ontology/1"


def semantic_soft_positive_weights(
    embeddings: torch.Tensor,
    *,
    top_k: int = 2,
    temperature: float = 0.1,
    min_weight: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.ndim != 2 or embeddings.shape[0] < 1:
        raise ValueError("semantic embeddings must be [V,D]")
    if embeddings.shape[1] < 1 or not torch.isfinite(embeddings).all():
        raise ValueError("semantic embeddings must be finite with D > 0")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if (
        not math.isfinite(min_weight)
        or min_weight < 0.0
        or min_weight > 1.0
    ):
        raise ValueError("min_weight must be within [0,1]")

    normalized = F.normalize(embeddings.float(), dim=-1)
    cosine = normalized @ normalized.T
    count = int(cosine.shape[0])
    weights = torch.eye(count, dtype=torch.float32)

    if count <= 1 or top_k == 0:
        return weights, cosine

    k = min(top_k, count - 1)
    selected = torch.zeros((count, count), dtype=torch.bool)
    values = torch.zeros((count, count), dtype=torch.float32)
    for index in range(count):
        row = cosine[index].clone()
        row[index] = -torch.inf
        sims, neighbors = torch.topk(
            row,
            k=k,
            largest=True,
            sorted=True,
        )
        for similarity, neighbor in zip(
            sims.tolist(), neighbors.tolist()
        ):
            weight = math.exp(
                (float(similarity) - 1.0) / temperature
            )
            if weight < min_weight:
                continue
            selected[index, neighbor] = True
            values[index, neighbor] = float(
                min(1.0, max(0.0, weight))
            )

    # Text similarity is symmetric. Make the sparse neighborhood symmetric
    # while retaining the stronger directed top-k edge when only one side
    # selected the pair.
    symmetric = torch.maximum(values, values.T)
    weights = torch.maximum(weights, symmetric)
    return weights, cosine


def build_metadata(
    vocabulary: RelationVocabulary,
    weights: torch.Tensor,
    cosine: torch.Tensor,
    *,
    source_tensor_sha256: str,
    top_k: int,
    temperature: float,
    min_weight: float,
) -> dict[str, object]:
    if weights.shape != cosine.shape:
        raise ValueError("ontology weights/cosine shapes must match")
    edges: list[dict[str, object]] = []
    for left in range(weights.shape[0]):
        for right in range(left + 1, weights.shape[1]):
            weight = float(weights[left, right])
            if weight <= 0.0:
                continue
            edges.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "left": vocabulary.predicates[left],
                    "right": vocabulary.predicates[right],
                    "cosine": float(cosine[left, right]),
                    "weight": weight,
                }
            )
    return {
        "schema": ONTOLOGY_SCHEMA,
        "vocabulary_sha256": vocabulary.sha256(),
        "source_tensor_sha256": source_tensor_sha256,
        "weights_tensor_sha256": tensor_sha256(weights),
        "shape": list(weights.shape),
        "top_k": top_k,
        "temperature": temperature,
        "min_weight": min_weight,
        "edge_count": len(edges),
        "edges": edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a supervision-only semantic soft-positive ontology "
            "from raw predicate text embeddings."
        )
    )
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--source-embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--min-weight", type=float, default=0.05)
    args = parser.parse_args()

    out = Path(args.out)
    metadata_path = out.with_suffix(out.suffix + ".json")
    if out.exists() or metadata_path.exists():
        raise FileExistsError("ontology output already exists")

    vocabulary = RelationVocabulary.load(args.vocabulary)
    source = torch.load(
        args.source_embeddings,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(source, torch.Tensor):
        raise ValueError("source embeddings must contain one tensor")
    if source.shape[0] != len(vocabulary.predicates):
        raise ValueError(
            "source embedding rows do not match vocabulary"
        )

    weights, cosine = semantic_soft_positive_weights(
        source,
        top_k=args.top_k,
        temperature=args.temperature,
        min_weight=args.min_weight,
    )
    metadata = build_metadata(
        vocabulary,
        weights,
        cosine,
        source_tensor_sha256=tensor_sha256(source),
        top_k=args.top_k,
        temperature=args.temperature,
        min_weight=args.min_weight,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out)
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
