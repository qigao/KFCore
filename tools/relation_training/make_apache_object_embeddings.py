from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from apache_text_student import (
    PredicateTextStudent,
    load_apache_checkpoint,
)
from benchmark import RelationVocabulary
from make_predicate_embeddings import tensor_sha256


OBJECT_BANK_SCHEMA = "kfcore.apache-object-embeddings/1"
OBJECT_TEMPLATES = (
    "{p}",
    "a photo of a {p}",
)
RELEASED_TEXT_DIM = 512


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1 << 20),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_strings_sha256(
    values: Sequence[str],
) -> str:
    payload = (
        json.dumps(
            list(values),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tokenize(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
) -> tuple[Tensor, Tensor]:
    encoded = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
    )
    rows = encoded.get("input_ids")
    if not isinstance(rows, list):
        raise ValueError(
            "CLIP tokenizer must return input_ids"
        )
    ids = torch.full(
        (len(rows), max_length),
        PredicateTextStudent.PAD_ID,
        dtype=torch.int64,
    )
    padding_mask = torch.ones(
        (len(rows), max_length),
        dtype=torch.bool,
    )
    for index, row in enumerate(rows):
        if (
            not isinstance(row, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                for value in row
            )
        ):
            raise ValueError(
                "tokenizer input_ids rows must be integer arrays"
            )
        clipped = row[:max_length]
        if clipped:
            ids[index, : len(clipped)] = (
                torch.tensor(
                    clipped,
                    dtype=torch.int64,
                )
            )
            padding_mask[
                index,
                : len(clipped),
            ] = False
    return ids, padding_mask


@torch.no_grad()
def encode_object_bank(
    model: PredicateTextStudent,
    tokenizer: Any,
    object_labels: Sequence[str],
    *,
    templates: Sequence[str] = OBJECT_TEMPLATES,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
) -> Tensor:
    if not object_labels:
        raise ValueError(
            "object label list must not be empty"
        )
    if len(set(object_labels)) != len(object_labels):
        raise ValueError(
            "object labels must be unique"
        )
    if tuple(templates) != OBJECT_TEMPLATES:
        raise ValueError(
            "Apache released object templates must be exact"
        )
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    target = torch.device(device)
    model = model.to(target).eval()
    summed: Tensor | None = None
    for template in templates:
        prompts = [
            template.format(p=label)
            for label in object_labels
        ]
        rows: list[Tensor] = []
        for start in range(
            0,
            len(prompts),
            batch_size,
        ):
            ids, mask = _tokenize(
                tokenizer,
                prompts[start : start + batch_size],
                max_length=model.config.max_length,
            )
            rows.append(
                model(
                    ids.to(target),
                    mask.to(target),
                )
                .float()
                .cpu()
            )
        encoded = torch.cat(
            rows,
            dim=0,
        )
        summed = (
            encoded
            if summed is None
            else summed + encoded
        )

    assert summed is not None
    bank = F.normalize(
        summed,
        dim=-1,
    ).float()
    if (
        bank.ndim != 2
        or bank.shape[0] != len(object_labels)
        or not torch.isfinite(bank).all()
    ):
        raise RuntimeError(
            "object bank generation produced invalid tensor"
        )
    return bank.contiguous()


def _require_sha256(
    value: object,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            char not in "0123456789abcdef"
            for char in value
        )
    ):
        raise ValueError(
            f"{name} must be lowercase SHA-256 hex"
        )
    return value


def validate_object_bank_provenance(
    bank: Tensor,
    *,
    tensor_path: str | Path,
    metadata_path: str | Path,
    object_labels: Sequence[str],
    text_student_path: str | Path,
    tokenizer_contract_path: str | Path,
    required_dim: int = RELEASED_TEXT_DIM,
) -> dict[str, Any]:
    meta = json.loads(
        Path(metadata_path).read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(meta, dict)
        or meta.get("schema") != OBJECT_BANK_SCHEMA
    ):
        raise ValueError(
            "unsupported Apache object-bank provenance schema"
        )
    if meta.get("templates") != list(
        OBJECT_TEMPLATES
    ):
        raise ValueError(
            "Apache object-bank templates differ from release"
        )
    if meta.get("object_labels") != list(
        object_labels
    ):
        raise ValueError(
            "Apache object-bank object-label order differs from vocabulary"
        )
    expected_labels_sha = ordered_strings_sha256(
        object_labels
    )
    if (
        meta.get("object_labels_sha256")
        != expected_labels_sha
    ):
        raise ValueError(
            "Apache object-bank object-label hash mismatch"
        )

    if (
        bank.ndim != 2
        or bank.shape
        != (len(object_labels), required_dim)
        or not torch.isfinite(bank).all()
    ):
        raise ValueError(
            "Apache object bank must be finite [O,512]"
        )
    norms = torch.linalg.vector_norm(
        bank.float(),
        dim=-1,
    )
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise ValueError(
            "Apache object-bank rows must be L2-normalized"
        )

    actual_tensor_sha = tensor_sha256(
        bank.float()
    )
    actual_file_sha = sha256_file(
        tensor_path
    )
    actual_student_sha = sha256_file(
        text_student_path
    )
    actual_tokenizer_contract_sha = sha256_file(
        tokenizer_contract_path
    )
    checks = {
        "tensor_sha256": actual_tensor_sha,
        "tensor_file_sha256": actual_file_sha,
        "text_student_sha256": actual_student_sha,
        "tokenizer_contract_sha256": (
            actual_tokenizer_contract_sha
        ),
    }
    for key, actual in checks.items():
        _require_sha256(
            meta.get(key),
            key,
        )
        if meta[key] != actual:
            raise ValueError(
                f"Apache object-bank {key} mismatch"
            )
    if meta.get("output_dim") != required_dim:
        raise ValueError(
            "Apache object-bank output_dim must be 512"
        )
    if meta.get("row_normalized") is not True:
        raise ValueError(
            "Apache object-bank provenance must record row normalization"
        )

    return dict(meta)


def build_object_bank(
    *,
    checkpoint_path: str | Path,
    tokenizer_dir: str | Path,
    tokenizer_contract_path: str | Path,
    object_labels: Sequence[str],
    output_path: str | Path,
    metadata_path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    output = Path(output_path)
    metadata = Path(metadata_path)
    if output.exists() or metadata.exists():
        raise FileExistsError(
            "object-bank outputs must not already exist"
        )

    contract_path = Path(
        tokenizer_contract_path
    )
    contract = json.loads(
        contract_path.read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(contract, dict)
        or contract.get("schema")
        != "kfcore.clip-tokenizer-golden/1"
    ):
        raise ValueError(
            "unsupported CLIP tokenizer contract"
        )

    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
    )
    model = load_apache_checkpoint(
        str(checkpoint_path),
        device=device,
    )
    bank = encode_object_bank(
        model,
        tokenizer,
        object_labels,
        device=device,
    )
    if bank.shape[1] != RELEASED_TEXT_DIM:
        raise ValueError(
            "released object bank requires 512-D text student output"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        bank,
        output,
    )
    payload = {
        "schema": OBJECT_BANK_SCHEMA,
        "templates": list(
            OBJECT_TEMPLATES
        ),
        "object_labels": list(
            object_labels
        ),
        "object_labels_sha256": (
            ordered_strings_sha256(
                object_labels
            )
        ),
        "output_dim": int(
            bank.shape[1]
        ),
        "row_normalized": True,
        "tensor_sha256": tensor_sha256(
            bank
        ),
        "tensor_file_sha256": sha256_file(
            output
        ),
        "text_student_sha256": sha256_file(
            checkpoint_path
        ),
        "tokenizer_contract_sha256": (
            sha256_file(
                tokenizer_contract_path
            )
        ),
    }
    metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    metadata.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
    )
    parser.add_argument(
        "--tokenizer-contract",
        required=True,
    )
    parser.add_argument(
        "--vocabulary",
        required=True,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--metadata",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    args = parser.parse_args()

    vocabulary = RelationVocabulary.load(
        args.vocabulary
    )
    if not vocabulary.object_labels:
        raise ValueError(
            "vocabulary must contain object labels"
        )
    report = build_object_bank(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        tokenizer_contract_path=(
            args.tokenizer_contract
        ),
        object_labels=(
            vocabulary.object_labels
        ),
        output_path=args.out,
        metadata_path=args.metadata,
        device=args.device,
    )
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
