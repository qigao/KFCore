from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from benchmark import RelationVocabulary


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_TEMPLATES = (
    "{predicate}",
    "a photo of one object {predicate} another object",
    "the relation between two objects is {predicate}",
)


def predicate_text(predicate: str) -> str:
    value = predicate.replace("_", " ").strip()
    if not value:
        raise ValueError("predicate text must not be empty")
    return value


def build_prompts(
    predicates: Sequence[str],
    templates: Sequence[str],
) -> list[str]:
    if not templates:
        raise ValueError("at least one predicate prompt template is required")
    prompts: list[str] = []
    for predicate in predicates:
        text = predicate_text(predicate)
        for template in templates:
            if "{predicate}" not in template:
                raise ValueError(
                    "predicate prompt template must contain {predicate}"
                )
            prompt = template.format(predicate=text).strip()
            if not prompt:
                raise ValueError("predicate prompt must not be empty")
            prompts.append(prompt)
    return prompts


def aggregate_prompt_embeddings(
    prompt_embeddings: torch.Tensor,
    *,
    predicate_count: int,
    template_count: int,
) -> torch.Tensor:
    if predicate_count <= 0 or template_count <= 0:
        raise ValueError("predicate/template counts must be positive")
    if (
        prompt_embeddings.ndim != 2
        or prompt_embeddings.shape[0]
        != predicate_count * template_count
        or prompt_embeddings.shape[1] <= 0
    ):
        raise ValueError("prompt embeddings have an invalid shape")
    if not torch.isfinite(prompt_embeddings).all():
        raise ValueError("prompt embeddings contain non-finite values")

    normalized = F.normalize(prompt_embeddings.float(), dim=-1)
    grouped = normalized.reshape(
        predicate_count,
        template_count,
        normalized.shape[-1],
    )
    averaged = grouped.mean(dim=1)
    result = F.normalize(averaged, dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("predicate embeddings contain non-finite values")
    return result


def identity_embeddings(vocabulary: RelationVocabulary) -> torch.Tensor:
    return torch.eye(
        len(vocabulary.predicates),
        dtype=torch.float32,
    )


def clip_embeddings(
    vocabulary: RelationVocabulary,
    *,
    model_id: str,
    templates: Sequence[str],
) -> torch.Tensor:
    if not model_id:
        raise ValueError("CLIP model ID must not be empty")

    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = CLIPTextModelWithProjection.from_pretrained(model_id)
    model.eval()
    model.requires_grad_(False)

    prompts = build_prompts(vocabulary.predicates, templates)
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        output = model(**inputs)
    text_embeds = output.text_embeds
    if text_embeds is None:
        raise RuntimeError("CLIP text model did not return text embeddings")
    return aggregate_prompt_embeddings(
        text_embeds.detach().cpu(),
        predicate_count=len(vocabulary.predicates),
        template_count=len(templates),
    )


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def stable_metadata_json(payload: dict[str, object]) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic predicate prototypes for Relation-v1."
        )
    )
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode",
        choices=("identity", "clip"),
        default="identity",
    )
    parser.add_argument(
        "--text-model",
        default=DEFAULT_CLIP_MODEL,
    )
    parser.add_argument(
        "--template",
        action="append",
        dest="templates",
        help=(
            "Repeat to override the default CLIP prompt ensemble. "
            "Every template must contain {predicate}."
        ),
    )
    args = parser.parse_args()

    vocabulary = RelationVocabulary.load(args.vocabulary)
    out = Path(args.out)
    metadata_path = out.with_suffix(out.suffix + ".json")
    if out.exists() or metadata_path.exists():
        raise FileExistsError(
            f"predicate embedding output already exists: {out}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    templates = tuple(args.templates or DEFAULT_TEMPLATES)
    if args.mode == "identity":
        embeddings = identity_embeddings(vocabulary)
        metadata = {
            "schema": "kfcore.predicate-embeddings/1",
            "mode": "identity",
            "vocabulary_sha256": vocabulary.sha256(),
            "predicates": list(vocabulary.predicates),
            "shape": list(embeddings.shape),
            "tensor_sha256": tensor_sha256(embeddings),
        }
    else:
        embeddings = clip_embeddings(
            vocabulary,
            model_id=args.text_model,
            templates=templates,
        )
        metadata = {
            "schema": "kfcore.predicate-embeddings/1",
            "mode": "clip",
            "text_model": args.text_model,
            "templates": list(templates),
            "vocabulary_sha256": vocabulary.sha256(),
            "predicates": list(vocabulary.predicates),
            "shape": list(embeddings.shape),
            "tensor_sha256": tensor_sha256(embeddings),
            "min_l2_norm": float(
                torch.linalg.vector_norm(embeddings, dim=-1).min()
            ),
            "max_l2_norm": float(
                torch.linalg.vector_norm(embeddings, dim=-1).max()
            ),
        }

    torch.save(embeddings, out)
    metadata_path.write_text(
        stable_metadata_json(metadata),
        encoding="utf-8",
    )
    print(
        f"wrote {out} shape={tuple(embeddings.shape)} "
        f"mode={args.mode} vocabulary_sha256={vocabulary.sha256()} "
        f"tensor_sha256={tensor_sha256(embeddings)}"
    )
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
