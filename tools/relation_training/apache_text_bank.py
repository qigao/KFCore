from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import io
from pathlib import Path
from typing import Callable, Sequence
import zipfile

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from apache_object_bank import load_object_text_bank
from apache_text_student import (
    PredicateTextStudent,
    PredicateTextStudentConfig,
    load_apache_checkpoint,
)
from apache_training_recipe import RELEASED_TEXT_DIM


PREDICATE_BANK_SCHEMA = "kfcore.apache-predicate-text-bank/1"
TEXT_BANK_DERIVATION_SCHEMA = "kfcore.apache-text-bank-derivation/1"
TOKENIZER_BUNDLE_SCHEMA = "kfcore.apache-tokenizer-bundle/1"

RELEASED_TEXT_STUDENT_SHA256 = (
    "e0317830b68ea51e6711fc90d4a35954"
    "d0528e5bd78a8d5afd966601ce4ed119"
)
RELEASED_TOKENIZER_ID = "openai/clip-vit-base-patch32"
RELEASED_TRANSFORMERS_VERSION = "5.14.1"

RELEASED_PREDICATE_TEMPLATES = (
    "{p}",
    "one object is {p} another object",
    "a photo of something {p} something",
)
RELEASED_OBJECT_TEMPLATES = (
    "{p}",
    "a photo of a {p}",
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
TOKENIZER_LAYOUTS = (
    ("tokenizer.json",),
    ("vocab.json", "merges.txt"),
)
NPZ_FORMAT = "deterministic-zip-stored-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1 << 20),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_strings_sha256(
    values: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def tensor_sha256(value: Tensor) -> str:
    tensor = (
        value.detach()
        .cpu()
        .contiguous()
    )
    digest = hashlib.sha256()
    digest.update(
        str(tensor.dtype).encode(
            "ascii"
        )
    )
    digest.update(b"\0")
    digest.update(
        ",".join(
            str(int(dim))
            for dim in tensor.shape
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(
        tensor.numpy().tobytes(
            order="C"
        )
    )
    return digest.hexdigest()


def _validate_names(
    values: Sequence[str],
    name: str,
) -> tuple[str, ...]:
    result = tuple(values)
    if (
        not result
        or any(
            not isinstance(value, str)
            or not value
            for value in result
        )
        or len(set(result)) != len(result)
    ):
        raise ValueError(
            f"{name} must be unique non-empty strings"
        )
    return result


def _npy_bytes(
    array: np.ndarray,
) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(array),
        allow_pickle=False,
    )
    return stream.getvalue()


def write_deterministic_npz(
    path: Path,
    arrays: Sequence[
        tuple[str, np.ndarray]
    ],
) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name, array in arrays:
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(
                    1980,
                    1,
                    1,
                    0,
                    0,
                    0,
                ),
            )
            info.compress_type = (
                zipfile.ZIP_STORED
            )
            info.create_system = 3
            info.external_attr = (
                0o100644 << 16
            )
            archive.writestr(
                info,
                _npy_bytes(
                    np.asarray(array)
                ),
            )


def load_predicate_text_bank(
    path: str | Path,
    predicates: Sequence[str],
    *,
    text_dim: int = RELEASED_TEXT_DIM,
) -> tuple[Tensor, dict[str, object]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    names = _validate_names(
        predicates,
        "predicate labels",
    )
    if (
        isinstance(text_dim, bool)
        or not isinstance(text_dim, int)
        or text_dim <= 0
    ):
        raise ValueError(
            "predicate text dimension must be a positive integer"
        )

    with np.load(
        source,
        allow_pickle=False,
    ) as payload:
        required = {
            "predicates",
            "embeddings",
            "templates",
        }
        missing = required - set(
            payload.files
        )
        if missing:
            raise ValueError(
                "predicate bank is missing "
                + ", ".join(
                    sorted(missing)
                )
            )
        raw_names = np.asarray(
            payload["predicates"]
        )
        raw_embeddings = np.asarray(
            payload["embeddings"]
        )
        raw_templates = np.asarray(
            payload["templates"]
        )

    if raw_names.ndim != 1:
        raise ValueError(
            "predicate bank predicates must be [V]"
        )
    bank_names = tuple(
        str(value)
        for value in raw_names.tolist()
    )
    if bank_names != names:
        raise ValueError(
            "predicate bank names/order do not match relation vocabulary"
        )

    if raw_templates.ndim != 1:
        raise ValueError(
            "predicate bank templates must be [T]"
        )
    templates = tuple(
        str(value)
        for value in raw_templates.tolist()
    )
    if (
        templates
        != RELEASED_PREDICATE_TEMPLATES
    ):
        raise ValueError(
            "predicate bank templates differ from released 3-template ensemble"
        )

    array = np.asarray(
        raw_embeddings,
        dtype=np.float32,
    )
    if array.shape != (
        len(names),
        text_dim,
    ):
        raise ValueError(
            f"predicate bank embeddings must be [{len(names)},{text_dim}]"
        )
    if not np.isfinite(array).all():
        raise ValueError(
            "predicate bank embeddings must be finite"
        )
    tensor = torch.from_numpy(
        np.ascontiguousarray(array)
    )
    return tensor, {
        "schema": PREDICATE_BANK_SCHEMA,
        "artifact_sha256": sha256_file(
            source
        ),
        "shape": [
            len(names),
            text_dim,
        ],
        "source_dtype": str(
            raw_embeddings.dtype
        ),
        "runtime_dtype": str(
            tensor.dtype
        ),
        "predicate_count": len(
            names
        ),
        "predicate_order": list(
            names
        ),
        "predicate_order_sha256": (
            ordered_strings_sha256(
                names
            )
        ),
        "templates": list(
            templates
        ),
        "templates_sha256": (
            ordered_strings_sha256(
                templates
            )
        ),
        "text_dim": text_dim,
    }


def tokenizer_bundle_report(
    tokenizer_dir: str | Path,
) -> dict[str, object]:
    root = Path(
        tokenizer_dir
    )
    if not root.is_dir():
        raise FileNotFoundError(root)
    layout_ok = any(
        all(
            (
                root
                / filename
            ).is_file()
            for filename in layout
        )
        for layout in TOKENIZER_LAYOUTS
    )
    if not layout_ok:
        raise ValueError(
            "released tokenizer directory must contain tokenizer.json or vocab.json+merges.txt"
        )

    files = {
        filename: sha256_file(
            root / filename
        )
        for filename in TOKENIZER_FILES
        if (
            root
            / filename
        ).is_file()
    }
    digest = hashlib.sha256()
    for filename in sorted(
        files
    ):
        digest.update(
            filename.encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            files[
                filename
            ].encode("ascii")
        )
        digest.update(b"\0")
    return {
        "schema": TOKENIZER_BUNDLE_SCHEMA,
        "tokenizer_id": (
            RELEASED_TOKENIZER_ID
        ),
        "source_kind": "local-bundle",
        "path": str(
            root.resolve()
        ),
        "files": files,
        "bundle_sha256": (
            digest.hexdigest()
        ),
    }


def load_clip_tokenizer(
    tokenizer_dir: str | Path,
):
    from transformers import (
        CLIPTokenizer,
    )

    return CLIPTokenizer.from_pretrained(
        str(
            Path(
                tokenizer_dir
            )
        ),
        local_files_only=True,
    )


def tokenize_texts(
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
) -> tuple[Tensor, Tensor]:
    values = tuple(
        texts
    )
    encoded = tokenizer(
        list(values),
        truncation=True,
        max_length=max_length,
    )
    if (
        not isinstance(encoded, dict)
        or "input_ids" not in encoded
    ):
        raise ValueError(
            "CLIP tokenizer must return input_ids"
        )
    rows = encoded[
        "input_ids"
    ]
    if (
        not isinstance(rows, list)
        or len(rows) != len(values)
    ):
        raise ValueError(
            "CLIP tokenizer input_ids row count mismatch"
        )
    ids = torch.zeros(
        (
            len(values),
            max_length,
        ),
        dtype=torch.int64,
    )
    padding = torch.ones(
        (
            len(values),
            max_length,
        ),
        dtype=torch.bool,
    )
    for index, row in enumerate(
        rows
    ):
        if (
            not isinstance(row, list)
            or any(
                isinstance(value, bool)
                or not isinstance(
                    value,
                    int,
                )
                for value in row
            )
        ):
            raise ValueError(
                "CLIP tokenizer input_ids must be integer rows"
            )
        clipped = row[
            :max_length
        ]
        if any(
            value < 0
            for value in clipped
        ):
            raise ValueError(
                "CLIP token ids must be non-negative"
            )
        if clipped:
            ids[
                index,
                :len(clipped),
            ] = torch.tensor(
                clipped,
                dtype=torch.int64,
            )
            padding[
                index,
                :len(clipped),
            ] = False
    return ids, padding


@torch.no_grad()
def encode_labels_with_templates(
    model: PredicateTextStudent,
    tokenizer,
    labels: Sequence[str],
    templates: Sequence[str],
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
) -> Tensor:
    names = _validate_names(
        labels,
        "text-bank labels",
    )
    prompts = tuple(
        templates
    )
    if (
        not prompts
        or any(
            not isinstance(template, str)
            or "{p}" not in template
            for template in prompts
        )
    ):
        raise ValueError(
            "text-bank templates must contain {p}"
        )
    if (
        isinstance(batch_size, bool)
        or not isinstance(
            batch_size,
            int,
        )
        or batch_size <= 0
    ):
        raise ValueError(
            "batch_size must be a positive integer"
        )

    target = torch.device(
        device
    )
    model = model.to(
        target
    ).eval()
    summed: Tensor | None = None
    for template in prompts:
        strings = [
            template.format(
                p=name
            )
            for name in names
        ]
        rows: list[Tensor] = []
        for start in range(
            0,
            len(strings),
            batch_size,
        ):
            batch = strings[
                start :
                start + batch_size
            ]
            ids, padding = (
                tokenize_texts(
                    tokenizer,
                    batch,
                    max_length=(
                        model.config.max_length
                    ),
                )
            )
            rows.append(
                model(
                    ids.to(target),
                    padding.to(
                        target
                    ),
                ).detach().cpu()
            )
        encoded = torch.cat(
            rows,
            dim=0,
        )
        summed = (
            encoded
            if summed is None
            else summed
            + encoded
        )

    if summed is None:
        raise RuntimeError(
            "text-bank encoding did not execute"
        )
    result = F.normalize(
        summed.float(),
        dim=-1,
    )
    if not torch.isfinite(
        result
    ).all():
        raise RuntimeError(
            "text-bank embeddings became non-finite"
        )
    return result


def _student_config_report(
    model: PredicateTextStudent,
) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in asdict(
            model.config
        ).items()
    }


def rebuild_text_banks(
    *,
    student_checkpoint: str | Path,
    tokenizer_dir: str | Path,
    predicates: Sequence[str],
    object_labels: Sequence[str],
    predicate_output: str | Path,
    object_output: str | Path,
    evidence_output: str | Path,
    expected_student_sha256: str = (
        RELEASED_TEXT_STUDENT_SHA256
    ),
    require_released_config: bool = True,
    tokenizer_factory: Callable[
        [str | Path],
        object,
    ] = load_clip_tokenizer,
    transformers_version: str | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
) -> dict[str, object]:
    student_path = Path(
        student_checkpoint
    )
    predicate_path = Path(
        predicate_output
    )
    object_path = Path(
        object_output
    )
    evidence_path = Path(
        evidence_output
    )
    for path in (
        predicate_path,
        object_path,
        evidence_path,
    ):
        if path.exists():
            raise FileExistsError(
                path
            )

    student_sha = sha256_file(
        student_path
    )
    if (
        expected_student_sha256
        and student_sha
        != expected_student_sha256
    ):
        raise ValueError(
            "text-student checkpoint SHA-256 does not match expected release identity"
        )

    model = load_apache_checkpoint(
        str(student_path),
        device="cpu",
    )
    if (
        require_released_config
        and model.config
        != PredicateTextStudentConfig()
    ):
        raise ValueError(
            "text-student config differs from released architecture"
        )

    tokenizer_report = (
        tokenizer_bundle_report(
            tokenizer_dir
        )
    )
    tokenizer = tokenizer_factory(
        tokenizer_dir
    )

    predicate_names = _validate_names(
        predicates,
        "predicate labels",
    )
    object_names = _validate_names(
        object_labels,
        "object labels",
    )

    predicate_embeddings = (
        encode_labels_with_templates(
            model,
            tokenizer,
            predicate_names,
            RELEASED_PREDICATE_TEMPLATES,
            device=device,
            batch_size=batch_size,
        )
    )
    object_embeddings = (
        encode_labels_with_templates(
            model,
            tokenizer,
            object_names,
            RELEASED_OBJECT_TEMPLATES,
            device=device,
            batch_size=batch_size,
        )
    )

    predicate_f16 = (
        predicate_embeddings
        .numpy()
        .astype(
            np.float16,
        )
    )
    object_f16 = (
        object_embeddings
        .numpy()
        .astype(
            np.float16,
        )
    )

    write_deterministic_npz(
        predicate_path,
        (
            (
                "embeddings",
                predicate_f16,
            ),
            (
                "predicates",
                np.asarray(
                    predicate_names,
                    dtype=np.str_,
                ),
            ),
            (
                "templates",
                np.asarray(
                    RELEASED_PREDICATE_TEMPLATES,
                    dtype=np.str_,
                ),
            ),
        ),
    )
    write_deterministic_npz(
        object_path,
        (
            (
                "embeddings",
                object_f16,
            ),
            (
                "names",
                np.asarray(
                    object_names,
                    dtype=np.str_,
                ),
            ),
        ),
    )

    output_dim = int(
        model.config.output_dim
    )
    predicate_runtime, (
        predicate_report
    ) = load_predicate_text_bank(
        predicate_path,
        predicate_names,
        text_dim=output_dim,
    )
    object_runtime, (
        object_report
    ) = load_object_text_bank(
        object_path,
        object_names,
        text_dim=output_dim,
    )

    if transformers_version is None:
        try:
            transformers_version = (
                importlib.metadata.version(
                    "transformers"
                )
            )
        except (
            importlib.metadata.PackageNotFoundError
        ):
            transformers_version = (
                "unavailable"
            )

    evidence = {
        "schema": (
            TEXT_BANK_DERIVATION_SCHEMA
        ),
        "student_checkpoint_sha256": (
            student_sha
        ),
        "student_config": (
            _student_config_report(
                model
            )
        ),
        "tokenizer": (
            tokenizer_report
        ),
        "transformers_version": (
            transformers_version
        ),
        "encoding_semantics": (
            "encode-each-template,sum-template-vectors,l2-normalize-once"
        ),
        "predicate_bank": {
            "templates": list(
                RELEASED_PREDICATE_TEMPLATES
            ),
            "label_order_sha256": (
                ordered_strings_sha256(
                    predicate_names
                )
            ),
            "label_count": len(
                predicate_names
            ),
            "shape": list(
                predicate_runtime.shape
            ),
            "tensor_sha256": (
                tensor_sha256(
                    predicate_runtime
                )
            ),
            "artifact_sha256": (
                predicate_report[
                    "artifact_sha256"
                ]
            ),
            "npz_format": (
                NPZ_FORMAT
            ),
        },
        "object_bank": {
            "templates": list(
                RELEASED_OBJECT_TEMPLATES
            ),
            "label_order_sha256": (
                ordered_strings_sha256(
                    object_names
                )
            ),
            "label_count": len(
                object_names
            ),
            "shape": list(
                object_runtime.shape
            ),
            "tensor_sha256": (
                tensor_sha256(
                    object_runtime
                )
            ),
            "artifact_sha256": (
                object_report[
                    "artifact_sha256"
                ]
            ),
            "npz_format": (
                NPZ_FORMAT
            ),
        },
    }
    evidence_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    import json

    evidence_path.write_text(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return evidence
