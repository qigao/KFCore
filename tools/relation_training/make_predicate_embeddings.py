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


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def gram_diagnostics(value: torch.Tensor) -> dict[str, float]:
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError("prototype tensor must be [V,D]")
    if not torch.isfinite(value).all():
        raise ValueError("prototype tensor must be finite")

    rows = F.normalize(value.double(), dim=-1)
    gram = rows @ rows.T
    eigenvalues = torch.linalg.eigvalsh(gram)
    count = int(gram.shape[0])
    if count > 1:
        mask = ~torch.eye(count, dtype=torch.bool, device=gram.device)
        off = gram[mask].abs()
        max_off = float(off.max().item())
        mean_off = float(off.mean().item())
    else:
        max_off = 0.0
        mean_off = 0.0
    return {
        "max_abs_off_diagonal": max_off,
        "mean_abs_off_diagonal": mean_off,
        "min_eigenvalue": float(eigenvalues.min().item()),
        "max_eigenvalue": float(eigenvalues.max().item()),
    }


def symmetric_whiten_predicates(
    value: torch.Tensor,
    *,
    relative_epsilon: float = 1.0e-8,
) -> torch.Tensor:
    if relative_epsilon <= 0.0:
        raise ValueError("relative_epsilon must be positive")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError("predicate embeddings must be [V,D]")
    if value.shape[0] > value.shape[1]:
        raise ValueError(
            "row whitening requires embedding dimension >= predicate count"
        )
    if not torch.isfinite(value).all():
        raise ValueError("predicate embeddings must be finite")

    rows = F.normalize(value.double(), dim=-1)
    gram = rows @ rows.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    max_eigenvalue = float(eigenvalues.max().item())
    threshold = max(relative_epsilon * max_eigenvalue, relative_epsilon)
    min_eigenvalue = float(eigenvalues.min().item())
    if min_eigenvalue <= threshold:
        raise ValueError(
            "predicate Gram matrix is numerically rank-deficient: "
            f"min_eigenvalue={min_eigenvalue} threshold={threshold}"
        )

    inverse_sqrt = (
        eigenvectors
        @ torch.diag(eigenvalues.rsqrt())
        @ eigenvectors.T
    )
    whitened = inverse_sqrt @ rows
    whitened = F.normalize(whitened, dim=-1).float()
    if not torch.isfinite(whitened).all():
        raise ValueError("whitened predicate embeddings are non-finite")
    return whitened


def identity_predicate_embeddings(
    predicate_count: int,
    *,
    dimension: int | None = None,
) -> torch.Tensor:
    if predicate_count <= 0:
        raise ValueError("predicate_count must be positive")
    resolved = predicate_count if dimension is None else dimension
    if resolved < predicate_count:
        raise ValueError(
            "identity embedding dimension must be >= predicate_count"
        )
    if resolved <= 0:
        raise ValueError("embedding dimension must be positive")
    embeddings = torch.zeros(
        (predicate_count, resolved),
        dtype=torch.float32,
    )
    embeddings[
        torch.arange(predicate_count),
        torch.arange(predicate_count),
    ] = 1.0
    return embeddings


def predicate_prompts(
    predicates: Sequence[str],
    templates: Sequence[str] = DEFAULT_TEMPLATES,
) -> list[list[str]]:
    if not predicates:
        raise ValueError("predicate list must not be empty")
    if not templates:
        raise ValueError("prompt templates must not be empty")
    result: list[list[str]] = []
    for predicate in predicates:
        text = predicate.replace("_", " ").strip()
        if not text:
            raise ValueError("predicate text must not be empty")
        prompts: list[str] = []
        for template in templates:
            if "{predicate}" not in template:
                raise ValueError(
                    "every prompt template must contain {predicate}"
                )
            prompts.append(template.format(predicate=text))
        result.append(prompts)
    return result


def aggregate_prompt_embeddings(
    encoded: torch.Tensor,
    *,
    predicate_count: int,
    template_count: int,
) -> torch.Tensor:
    if encoded.ndim != 2:
        raise ValueError("encoded prompt embeddings must be [P*T,D]")
    if predicate_count <= 0 or template_count <= 0:
        raise ValueError("predicate/template counts must be positive")
    if encoded.shape[0] != predicate_count * template_count:
        raise ValueError("encoded prompt row count does not match P*T")
    if encoded.shape[1] <= 0 or not torch.isfinite(encoded).all():
        raise ValueError("encoded prompt embeddings must be finite")

    normalized = F.normalize(encoded.float(), dim=-1)
    grouped = normalized.reshape(
        predicate_count,
        template_count,
        encoded.shape[1],
    )
    averaged = grouped.mean(dim=1)
    result = F.normalize(averaged, dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("aggregated predicate embeddings are non-finite")
    return result


def clip_predicate_embeddings(
    predicates: Sequence[str],
    *,
    model_id: str = DEFAULT_CLIP_MODEL,
    templates: Sequence[str] = DEFAULT_TEMPLATES,
    device: str = "cpu",
) -> torch.Tensor:
    if not model_id:
        raise ValueError("CLIP model ID must not be empty")

    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    prompts_by_predicate = predicate_prompts(predicates, templates)
    flat_prompts = [
        prompt
        for prompts in prompts_by_predicate
        for prompt in prompts
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = CLIPTextModelWithProjection.from_pretrained(model_id)
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    encoded_rows: list[torch.Tensor] = []
    batch_size = 64
    with torch.inference_mode():
        for start in range(0, len(flat_prompts), batch_size):
            batch = flat_prompts[start:start + batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {
                key: value.to(device)
                for key, value in tokens.items()
            }
            output = model(**tokens)
            text_embeds = output.text_embeds
            if text_embeds is None:
                raise RuntimeError(
                    "CLIP text model did not return projected embeddings"
                )
            encoded_rows.append(text_embeds.detach().cpu())

    encoded = torch.cat(encoded_rows, dim=0)
    return aggregate_prompt_embeddings(
        encoded,
        predicate_count=len(predicates),
        template_count=len(templates),
    )


def write_outputs(
    out: Path,
    embeddings: torch.Tensor,
    *,
    vocabulary: RelationVocabulary,
    mode: str,
    model_id: str | None,
    templates: Sequence[str],
    source_embeddings: torch.Tensor | None = None,
) -> None:
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    metadata_path = out.with_suffix(out.suffix + ".json")
    if metadata_path.exists():
        raise FileExistsError(
            f"metadata output already exists: {metadata_path}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    if embeddings.ndim != 2:
        raise ValueError("predicate embeddings must be [V,D]")
    if embeddings.shape[0] != len(vocabulary.predicates):
        raise ValueError(
            "predicate embedding rows do not match vocabulary"
        )
    if not torch.isfinite(embeddings).all():
        raise ValueError("predicate embeddings must be finite")
    norms = embeddings.norm(dim=-1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise ValueError("predicate embedding rows must be unit-normalized")

    torch.save(embeddings.cpu(), out)
    metadata = {
        "schema": "kfcore.predicate-prototypes/1",
        "mode": mode,
        "model_id": model_id,
        "templates": list(templates),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "tensor_sha256": tensor_sha256(embeddings),
        "vocabulary_sha256": vocabulary.sha256(),
        "predicates": list(vocabulary.predicates),
        "gram": gram_diagnostics(embeddings),
        "source_gram": (
            gram_diagnostics(source_embeddings)
            if source_embeddings is not None
            else None
        ),
    }
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
    print(
        f"wrote {out} shape={tuple(embeddings.shape)} "
        f"tensor_sha256={metadata['tensor_sha256']}"
    )
    print(f"wrote {metadata_path}")


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
        choices=("identity", "clip", "clip-whitened"),
        default="identity",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=None,
        help=(
            "Identity-only prototype dimension. Defaults to predicate count."
        ),
    )
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    vocabulary = RelationVocabulary.load(args.vocabulary)
    source_embeddings: torch.Tensor | None = None
    if args.mode == "identity":
        embeddings = identity_predicate_embeddings(
            len(vocabulary.predicates),
            dimension=args.dimension,
        )
        model_id = None
        templates: tuple[str, ...] = ()
    else:
        if args.dimension is not None:
            raise ValueError("--dimension is identity-only")
        templates = DEFAULT_TEMPLATES
        raw_clip = clip_predicate_embeddings(
            vocabulary.predicates,
            model_id=args.clip_model,
            templates=templates,
            device=args.device,
        )
        if args.mode == "clip-whitened":
            source_embeddings = raw_clip
            embeddings = symmetric_whiten_predicates(raw_clip)
        else:
            embeddings = raw_clip
        model_id = args.clip_model

    write_outputs(
        Path(args.out),
        embeddings,
        vocabulary=vocabulary,
        mode=args.mode,
        model_id=model_id,
        templates=templates,
        source_embeddings=source_embeddings,
    )


if __name__ == "__main__":
    main()
