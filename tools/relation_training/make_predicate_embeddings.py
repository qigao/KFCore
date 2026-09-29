from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmark import RelationVocabulary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic identity predicate prototypes for a "
            "closed-vocabulary Relation-v1 baseline."
        )
    )
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    vocabulary = RelationVocabulary.load(args.vocabulary)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    embeddings = torch.eye(
        len(vocabulary.predicates),
        dtype=torch.float32,
    )
    torch.save(embeddings, out)
    print(
        f"wrote {out} shape={tuple(embeddings.shape)} "
        f"vocabulary_sha256={vocabulary.sha256()}"
    )


if __name__ == "__main__":
    main()
