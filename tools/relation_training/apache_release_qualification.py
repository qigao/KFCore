from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from apache_indoorvg_holdout import (
    DERIVATION_SCHEMA as INDOORVG_DERIVATION_SCHEMA,
    RELEASED_NOTE as INDOORVG_RELEASED_NOTE,
    RELEASED_SOURCE as INDOORVG_RELEASED_SOURCE,
    RELEASED_SPLITS as INDOORVG_RELEASED_SPLITS,
    holdout_bytes as indoorvg_holdout_bytes,
    stable_json_sha256 as indoorvg_stable_json_sha256,
)
from apache_pair_sampler import (
    RELEASED_FINAL_BUDGET,
    RELEASED_GEO_BUDGET,
)
from apache_pair_opportunity import (
    NPZ_FORMAT as PAIR_OPPORTUNITY_NPZ_FORMAT,
    REBUILD_SCHEMA as PAIR_OPPORTUNITY_REBUILD_SCHEMA,
    RELEASED_MIN_SUPPORT as PAIR_OPPORTUNITY_MIN_SUPPORT,
    RELEASED_SCAN_BOX_CAP as PAIR_OPPORTUNITY_SCAN_BOX_CAP,
    RELEASED_SOURCE_NAME as PAIR_OPPORTUNITY_SOURCE_NAME,
)
from apache_released_contract import (
    validate_released_scalar_contract,
    validate_released_structure_report,
)
from apache_objective import (
    SOURCE_ALLOW_SCHEMA,
)
from apache_source_columns import (
    DERIVATION_SCHEMA as SOURCE_COLUMN_DERIVATION_SCHEMA,
    RELEASED_RESTRICTED_SOURCES,
)
from apache_spatial_flags import (
    DERIVATION_ALGORITHM,
    DERIVATION_SCHEMA,
    MAJORITY_COMPARATOR,
    MAJORITY_THRESHOLD,
    SPATIAL_BIT,
)
from apache_text_bank import (
    NPZ_FORMAT as TEXT_BANK_NPZ_FORMAT,
    PREDICATE_BANK_SCHEMA,
    RELEASED_OBJECT_TEMPLATES,
    RELEASED_PREDICATE_TEMPLATES,
    RELEASED_TEXT_STUDENT_SHA256,
    RELEASED_TOKENIZER_ID,
    RELEASED_TRANSFORMERS_VERSION,
    TEXT_BANK_DERIVATION_SCHEMA,
    TOKENIZER_BUNDLE_SCHEMA,
    TOKENIZER_LAYOUTS,
)
from apache_text_student import (
    PredicateTextStudentConfig,
)
from apache_training_recipe import (
    RELEASED_AMP,
    RELEASED_AMP_DTYPE,
    RELEASED_D_MODEL,
    RELEASED_EMA_DECAY,
    RELEASED_IMAGE_SIZE,
    RELEASED_MAX_BOXES,
    RELEASED_PHOTOMETRIC_AUGMENT,
    RELEASED_TEXT_DIM,
)
from apache_mixture import (
    RELEASED_MICRO_BATCH_SIZE,
    RELEASED_MIX_FRACTIONS,
    RELEASED_SAMPLES_PER_EPOCH,
    RELEASED_SEED,
    RELEASED_SOURCE_NAMES,
    RELEASED_WORLD_SIZE,
)


CORPUS_SCHEMA = "kfcore.apache-released-corpus/1"
QUALIFICATION_SCHEMA = "kfcore.apache-release-training-qualification/1"
TRAINING_SCHEMA = "kfcore.relation-training-run/1"
REFERENCE_SOURCE_COMMIT = (
    "4a07de9d06f2e3f14309753b7907cf1d3a263b08"
)
RELEASED_EPOCHS = 12
RELEASED_EFFECTIVE_BATCH_SIZE = (
    RELEASED_MICRO_BATCH_SIZE * RELEASED_WORLD_SIZE
)
PROVENANCE_KINDS = {
    "published-pack",
    "deterministic-rebuild",
    "unresolved",
}


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


def payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        stable_json_bytes(payload)
    ).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise ValueError(
            f"{path}: expected a JSON object"
        )
    return value


def require_sha256(value: object, name: str) -> str:
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


def require_non_empty_string(
    value: object,
    name: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{name} must be a non-empty string"
        )
    return value


def _require_positive_int(
    value: object,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer"
        )
    return value


def _require_non_negative_int(
    value: object,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )
    return value


def _finite_number(
    value: object,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"{name} must be finite"
        )
    return float(value)


def _ordered_names_sha256(
    names: list[str],
) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(
            name.encode("utf-8")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_tokenizer_bundle(
    payload: object,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "text-bank tokenizer evidence must be an object"
        )
    if payload.get("schema") != TOKENIZER_BUNDLE_SCHEMA:
        raise ValueError(
            "unsupported text-bank tokenizer bundle schema"
        )
    if payload.get("tokenizer_id") != RELEASED_TOKENIZER_ID:
        raise ValueError(
            "released text-bank tokenizer id differs from CLIP B/32"
        )
    if payload.get("source_kind") != "local-bundle":
        raise ValueError(
            "released text-bank tokenizer must be a local bundle"
        )
    path = require_non_empty_string(
        payload.get("path"),
        "text-bank tokenizer path",
    )
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(
            "text-bank tokenizer file hashes are missing"
        )
    normalized_files: dict[str, str] = {}
    for name, value in files.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "text-bank tokenizer filenames must be strings"
            )
        normalized_files[name] = require_sha256(
            value,
            f"text-bank tokenizer {name} SHA-256",
        )
    if not any(
        all(
            filename in normalized_files
            for filename in layout
        )
        for layout in TOKENIZER_LAYOUTS
    ):
        raise ValueError(
            "text-bank tokenizer bundle lacks a released CLIP tokenizer layout"
        )
    expected_bundle = hashlib.sha256()
    for filename in sorted(normalized_files):
        expected_bundle.update(
            filename.encode("utf-8")
        )
        expected_bundle.update(b"\0")
        expected_bundle.update(
            normalized_files[
                filename
            ].encode("ascii")
        )
        expected_bundle.update(b"\0")
    bundle_sha = require_sha256(
        payload.get("bundle_sha256"),
        "text-bank tokenizer bundle_sha256",
    )
    if bundle_sha != expected_bundle.hexdigest():
        raise ValueError(
            "text-bank tokenizer bundle hash does not match file hashes"
        )
    return {
        "schema": TOKENIZER_BUNDLE_SCHEMA,
        "tokenizer_id": RELEASED_TOKENIZER_ID,
        "source_kind": "local-bundle",
        "path": path,
        "files": normalized_files,
        "bundle_sha256": bundle_sha,
    }


def _validate_text_bank_entry(
    payload: object,
    *,
    name: str,
    expected_templates: tuple[str, ...],
    expected_artifact_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"{name} text-bank derivation must be an object"
        )
    templates = payload.get("templates")
    if templates != list(expected_templates):
        raise ValueError(
            f"{name} text-bank templates differ from release"
        )
    label_order_sha = require_sha256(
        payload.get("label_order_sha256"),
        f"{name} text-bank label_order_sha256",
    )
    label_count = _require_positive_int(
        payload.get("label_count"),
        f"{name} text-bank label_count",
    )
    shape = payload.get("shape")
    if shape != [
        label_count,
        RELEASED_TEXT_DIM,
    ]:
        raise ValueError(
            f"{name} text-bank shape must be [N,512]"
        )
    tensor_sha = require_sha256(
        payload.get("tensor_sha256"),
        f"{name} text-bank tensor_sha256",
    )
    artifact_sha = require_sha256(
        payload.get("artifact_sha256"),
        f"{name} text-bank artifact_sha256",
    )
    if artifact_sha != expected_artifact_sha256:
        raise ValueError(
            f"{name} text-bank artifact hash does not match corpus"
        )
    if payload.get("npz_format") != TEXT_BANK_NPZ_FORMAT:
        raise ValueError(
            f"{name} text-bank NPZ format differs from released derivation contract"
        )
    return {
        "templates": list(expected_templates),
        "label_order_sha256": label_order_sha,
        "label_count": label_count,
        "shape": list(shape),
        "tensor_sha256": tensor_sha,
        "artifact_sha256": artifact_sha,
        "npz_format": TEXT_BANK_NPZ_FORMAT,
    }


def _validate_text_bank_derivation(
    payload: object,
    *,
    predicate_artifact_sha256: str,
    object_artifact_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "text-bank derivation evidence must be an object"
        )
    if payload.get("schema") != TEXT_BANK_DERIVATION_SCHEMA:
        raise ValueError(
            "unsupported text-bank derivation schema"
        )
    student_sha = require_sha256(
        payload.get(
            "student_checkpoint_sha256"
        ),
        "text-bank student checkpoint SHA-256",
    )
    if student_sha != RELEASED_TEXT_STUDENT_SHA256:
        raise ValueError(
            "text-bank derivation must use the released text-student checkpoint"
        )
    expected_config = asdict(
        PredicateTextStudentConfig()
    )
    if payload.get("student_config") != expected_config:
        raise ValueError(
            "text-bank derivation student config differs from release"
        )
    tokenizer = _validate_tokenizer_bundle(
        payload.get("tokenizer")
    )
    if (
        payload.get("transformers_version")
        != RELEASED_TRANSFORMERS_VERSION
    ):
        raise ValueError(
            "text-bank derivation requires transformers 5.14.1"
        )
    if (
        payload.get("encoding_semantics")
        != "encode-each-template,sum-template-vectors,l2-normalize-once"
    ):
        raise ValueError(
            "text-bank encoding semantics differ from Apache"
        )
    predicate_bank = _validate_text_bank_entry(
        payload.get("predicate_bank"),
        name="predicate",
        expected_templates=(
            RELEASED_PREDICATE_TEMPLATES
        ),
        expected_artifact_sha256=(
            predicate_artifact_sha256
        ),
    )
    object_bank = _validate_text_bank_entry(
        payload.get("object_bank"),
        name="object",
        expected_templates=(
            RELEASED_OBJECT_TEMPLATES
        ),
        expected_artifact_sha256=(
            object_artifact_sha256
        ),
    )
    return {
        "schema": TEXT_BANK_DERIVATION_SCHEMA,
        "student_checkpoint_sha256": student_sha,
        "student_config": expected_config,
        "tokenizer": tokenizer,
        "transformers_version": RELEASED_TRANSFORMERS_VERSION,
        "encoding_semantics": (
            "encode-each-template,sum-template-vectors,l2-normalize-once"
        ),
        "predicate_bank": predicate_bank,
        "object_bank": object_bank,
    }


def _validate_indoorvg_holdout_derivation(
    payload: object,
    *,
    output_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "IndoorVG exclusion derivation evidence must be an object"
        )
    if (
        payload.get("schema")
        != INDOORVG_DERIVATION_SCHEMA
    ):
        raise ValueError(
            "unsupported IndoorVG exclusion derivation schema"
        )
    if (
        payload.get("source")
        != INDOORVG_RELEASED_SOURCE
    ):
        raise ValueError(
            "IndoorVG exclusion derivation source differs from release"
        )
    if (
        payload.get("splits")
        != list(INDOORVG_RELEASED_SPLITS)
    ):
        raise ValueError(
            "IndoorVG exclusion derivation requires splits=['val','test']"
        )

    split_inputs = payload.get(
        "split_inputs"
    )
    if (
        not isinstance(split_inputs, list)
        or len(split_inputs)
        != len(INDOORVG_RELEASED_SPLITS)
    ):
        raise ValueError(
            "IndoorVG exclusion derivation must describe val/test split inputs"
        )

    union_from_splits: set[str] = set()
    normalized_splits: list[
        dict[str, Any]
    ] = []
    for index, expected_split in enumerate(
        INDOORVG_RELEASED_SPLITS
    ):
        entry = split_inputs[
            index
        ]
        if not isinstance(entry, dict):
            raise ValueError(
                f"IndoorVG split input {index} must be an object"
            )
        if entry.get("split") != expected_split:
            raise ValueError(
                "IndoorVG exclusion split order differs from release"
            )
        ids = entry.get("vg_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(
                not isinstance(value, str)
                or not value
                for value in ids
            )
            or len(set(ids)) != len(ids)
            or ids != sorted(ids)
        ):
            raise ValueError(
                f"IndoorVG {expected_split} VG ids must be sorted unique non-empty strings"
            )
        count = _require_positive_int(
            entry.get(
                "vg_id_count"
            ),
            f"IndoorVG {expected_split} vg_id_count",
        )
        if count != len(ids):
            raise ValueError(
                f"IndoorVG {expected_split} count does not match ids"
            )
        digest = require_sha256(
            entry.get(
                "vg_ids_sha256"
            ),
            f"IndoorVG {expected_split} vg_ids_sha256",
        )
        if digest != _ordered_names_sha256(
            ids
        ):
            raise ValueError(
                f"IndoorVG {expected_split} VG-id hash does not match ids"
            )
        union_from_splits.update(
            ids
        )
        normalized_splits.append(
            {
                "split": expected_split,
                "vg_ids": list(ids),
                "vg_id_count": count,
                "vg_ids_sha256": digest,
            }
        )

    vg_ids = payload.get(
        "vg_ids"
    )
    if (
        not isinstance(vg_ids, list)
        or not vg_ids
        or any(
            not isinstance(value, str)
            or not value
            for value in vg_ids
        )
        or len(set(vg_ids)) != len(vg_ids)
        or vg_ids != sorted(vg_ids)
    ):
        raise ValueError(
            "IndoorVG exclusion vg_ids must be sorted unique non-empty strings"
        )
    if vg_ids != sorted(
        union_from_splits
    ):
        raise ValueError(
            "IndoorVG exclusion vg_ids do not equal val/test union"
        )
    vg_count = _require_positive_int(
        payload.get(
            "vg_id_count"
        ),
        "IndoorVG vg_id_count",
    )
    if vg_count != len(vg_ids):
        raise ValueError(
            "IndoorVG vg_id_count does not match vg_ids"
        )
    vg_digest = require_sha256(
        payload.get(
            "vg_ids_sha256"
        ),
        "IndoorVG vg_ids_sha256",
    )
    if vg_digest != _ordered_names_sha256(
        vg_ids
    ):
        raise ValueError(
            "IndoorVG vg_ids hash does not match ids"
        )

    vg2coco_sha = require_sha256(
        payload.get(
            "vg2coco_sha256"
        ),
        "IndoorVG vg2coco_sha256",
    )

    mapped_pairs = payload.get(
        "mapped_pairs"
    )
    if not isinstance(
        mapped_pairs,
        list,
    ):
        raise ValueError(
            "IndoorVG mapped_pairs must be an array"
        )
    normalized_pairs: list[
        dict[str, str]
    ] = []
    mapped_ids: set[str] = set()
    for index, pair in enumerate(
        mapped_pairs
    ):
        if not isinstance(pair, dict):
            raise ValueError(
                f"IndoorVG mapped pair {index} must be an object"
            )
        vg_id = pair.get(
            "vg_id"
        )
        coco_stem = pair.get(
            "coco_stem"
        )
        if (
            not isinstance(vg_id, str)
            or not vg_id
            or vg_id not in union_from_splits
            or vg_id in mapped_ids
        ):
            raise ValueError(
                "IndoorVG mapped pairs require unique held-out VG ids"
            )
        if (
            not isinstance(coco_stem, str)
            or len(coco_stem) != 12
            or not coco_stem.isdigit()
        ):
            raise ValueError(
                "IndoorVG mapped COCO stems must be exactly 12 digits"
            )
        mapped_ids.add(
            vg_id
        )
        normalized_pairs.append(
            {
                "vg_id": vg_id,
                "coco_stem": coco_stem,
            }
        )
    if normalized_pairs != sorted(
        normalized_pairs,
        key=lambda pair: pair[
            "vg_id"
        ],
    ):
        raise ValueError(
            "IndoorVG mapped_pairs must be sorted by VG id"
        )
    mapped_count = _require_non_negative_int(
        payload.get(
            "mapped_vg_count"
        ),
        "IndoorVG mapped_vg_count",
    )
    if mapped_count != len(
        normalized_pairs
    ):
        raise ValueError(
            "IndoorVG mapped_vg_count does not match mapped_pairs"
        )
    mapped_digest = require_sha256(
        payload.get(
            "mapped_pairs_sha256"
        ),
        "IndoorVG mapped_pairs_sha256",
    )
    if mapped_digest != indoorvg_stable_json_sha256(
        normalized_pairs
    ):
        raise ValueError(
            "IndoorVG mapped-pair hash does not match pairs"
        )

    unmapped = payload.get(
        "unmapped_vg_ids"
    )
    if (
        not isinstance(unmapped, list)
        or any(
            not isinstance(value, str)
            or not value
            for value in unmapped
        )
        or len(set(unmapped)) != len(unmapped)
        or unmapped != sorted(unmapped)
    ):
        raise ValueError(
            "IndoorVG unmapped_vg_ids must be sorted unique strings"
        )
    unmapped_count = _require_non_negative_int(
        payload.get(
            "unmapped_vg_count"
        ),
        "IndoorVG unmapped_vg_count",
    )
    if unmapped_count != len(
        unmapped
    ):
        raise ValueError(
            "IndoorVG unmapped_vg_count does not match ids"
        )
    if set(unmapped) & mapped_ids:
        raise ValueError(
            "IndoorVG mapped/unmapped VG ids overlap"
        )
    if (
        mapped_ids
        | set(unmapped)
        != set(vg_ids)
    ):
        raise ValueError(
            "IndoorVG mapped/unmapped VG ids do not partition the holdout"
        )

    coco_stems = payload.get(
        "coco_stems"
    )
    expected_coco_stems = sorted(
        {
            pair[
                "coco_stem"
            ]
            for pair in normalized_pairs
        }
    )
    if (
        not isinstance(coco_stems, list)
        or coco_stems
        != expected_coco_stems
    ):
        raise ValueError(
            "IndoorVG coco_stems do not match mapped pairs"
        )
    coco_count = _require_non_negative_int(
        payload.get(
            "coco_stem_count"
        ),
        "IndoorVG coco_stem_count",
    )
    if coco_count != len(
        coco_stems
    ):
        raise ValueError(
            "IndoorVG coco_stem_count does not match stems"
        )
    coco_digest = require_sha256(
        payload.get(
            "coco_stems_sha256"
        ),
        "IndoorVG coco_stems_sha256",
    )
    if coco_digest != _ordered_names_sha256(
        coco_stems
    ):
        raise ValueError(
            "IndoorVG coco-stem hash does not match stems"
        )

    stems = payload.get(
        "stems"
    )
    expected_stems = sorted(
        set(vg_ids)
        | set(coco_stems)
    )
    if (
        not isinstance(stems, list)
        or stems != expected_stems
    ):
        raise ValueError(
            "IndoorVG final stems do not equal VG/COCO union"
        )
    stem_count = _require_positive_int(
        payload.get(
            "stem_count"
        ),
        "IndoorVG stem_count",
    )
    if stem_count != len(stems):
        raise ValueError(
            "IndoorVG stem_count does not match stems"
        )
    stems_digest = require_sha256(
        payload.get(
            "stems_sha256"
        ),
        "IndoorVG stems_sha256",
    )
    if stems_digest != _ordered_names_sha256(
        stems
    ):
        raise ValueError(
            "IndoorVG final-stem hash does not match stems"
        )

    output_payload = {
        "source": (
            INDOORVG_RELEASED_SOURCE
        ),
        "splits": list(
            INDOORVG_RELEASED_SPLITS
        ),
        "note": (
            INDOORVG_RELEASED_NOTE
        ),
        "vg_ids": list(
            vg_ids
        ),
        "coco_stems": list(
            coco_stems
        ),
        "stems": list(
            stems
        ),
    }
    derived_output_sha = hashlib.sha256(
        indoorvg_holdout_bytes(
            output_payload
        )
    ).hexdigest()
    recorded_output_sha = require_sha256(
        payload.get(
            "output_sha256"
        ),
        "IndoorVG output_sha256",
    )
    if (
        recorded_output_sha
        != derived_output_sha
    ):
        raise ValueError(
            "IndoorVG exclusion output hash does not match reconstructed holdout"
        )
    if (
        recorded_output_sha
        != output_sha256
    ):
        raise ValueError(
            "IndoorVG exclusion output hash does not match corpus"
        )

    return {
        "schema": (
            INDOORVG_DERIVATION_SCHEMA
        ),
        "source": (
            INDOORVG_RELEASED_SOURCE
        ),
        "splits": list(
            INDOORVG_RELEASED_SPLITS
        ),
        "split_inputs": (
            normalized_splits
        ),
        "vg_ids": list(vg_ids),
        "vg_id_count": vg_count,
        "vg_ids_sha256": (
            vg_digest
        ),
        "vg2coco_sha256": (
            vg2coco_sha
        ),
        "mapped_pairs": (
            normalized_pairs
        ),
        "mapped_vg_count": (
            mapped_count
        ),
        "mapped_pairs_sha256": (
            mapped_digest
        ),
        "unmapped_vg_ids": list(
            unmapped
        ),
        "unmapped_vg_count": (
            unmapped_count
        ),
        "coco_stems": list(
            coco_stems
        ),
        "coco_stem_count": (
            coco_count
        ),
        "coco_stems_sha256": (
            coco_digest
        ),
        "stems": list(stems),
        "stem_count": stem_count,
        "stems_sha256": (
            stems_digest
        ),
        "output_sha256": (
            recorded_output_sha
        ),
    }


def _validate_source_column_derivation(
    payload: object,
    *,
    sidecar_sha256: str,
    spatial_derivation: dict[str, Any],
    predicate_order_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "source-column derivation evidence must be an object"
        )
    if (
        payload.get("schema")
        != SOURCE_COLUMN_DERIVATION_SCHEMA
    ):
        raise ValueError(
            "unsupported source-column derivation schema"
        )
    if (
        payload.get("source_order")
        != list(RELEASED_SOURCE_NAMES)
    ):
        raise ValueError(
            "source-column derivation source order differs from release"
        )
    if (
        payload.get("restricted_sources")
        != list(
            RELEASED_RESTRICTED_SOURCES
        )
    ):
        raise ValueError(
            "source-column derivation must restrict hicodet only"
        )

    union = payload.get(
        "union_predicate_order"
    )
    if (
        not isinstance(union, list)
        or not union
        or any(
            not isinstance(name, str)
            or not name
            for name in union
        )
        or len(set(union)) != len(union)
    ):
        raise ValueError(
            "source-column union predicate order must be unique non-empty strings"
        )
    union_count = _require_positive_int(
        payload.get(
            "union_predicate_count"
        ),
        "source-column union_predicate_count",
    )
    if union_count != len(union):
        raise ValueError(
            "source-column union predicate count does not match order"
        )
    union_sha = require_sha256(
        payload.get(
            "union_predicate_order_sha256"
        ),
        "source-column union_predicate_order_sha256",
    )
    if union_sha != _ordered_names_sha256(
        union
    ):
        raise ValueError(
            "source-column union predicate-order hash does not match order"
        )
    if union_sha != predicate_order_sha256:
        raise ValueError(
            "source-column union predicate order does not match named predicate bank"
        )

    spatial_sources = spatial_derivation.get(
        "sources"
    )
    if (
        not isinstance(spatial_sources, list)
        or len(spatial_sources)
        != len(RELEASED_SOURCE_NAMES)
    ):
        raise ValueError(
            "spatial derivation sources unavailable for source-column binding"
        )
    spatial_by_name = {
        source.get("source_name"): source
        for source in spatial_sources
        if isinstance(source, dict)
    }

    sources = payload.get(
        "sources"
    )
    if (
        not isinstance(sources, list)
        or len(sources)
        != len(RELEASED_SOURCE_NAMES)
    ):
        raise ValueError(
            "source-column derivation must describe all released sources"
        )

    union_set = set(union)
    normalized_sources: list[
        dict[str, Any]
    ] = []
    sidecar_sources: list[
        dict[str, object]
    ] = []
    names: list[str] = []

    for index, source in enumerate(
        sources
    ):
        if not isinstance(source, dict):
            raise ValueError(
                f"source-column source {index} must be an object"
            )
        name = require_non_empty_string(
            source.get("source_name"),
            f"source-column source {index} name",
        )
        names.append(name)

        local = source.get(
            "local_predicates"
        )
        if (
            not isinstance(local, list)
            or not local
            or any(
                not isinstance(value, str)
                or not value
                for value in local
            )
            or len(set(local)) != len(local)
        ):
            raise ValueError(
                f"{name} local predicates must be unique non-empty strings"
            )
        local_count = _require_positive_int(
            source.get(
                "local_predicate_count"
            ),
            f"{name} local_predicate_count",
        )
        if local_count != len(local):
            raise ValueError(
                f"{name} local predicate count does not match order"
            )
        local_sha = require_sha256(
            source.get(
                "local_predicate_order_sha256"
            ),
            f"{name} local_predicate_order_sha256",
        )
        if local_sha != _ordered_names_sha256(
            local
        ):
            raise ValueError(
                f"{name} local predicate-order hash does not match order"
            )

        unknown = source.get(
            "unknown_local_predicates"
        )
        expected_unknown = [
            value
            for value in local
            if value not in union_set
        ]
        if unknown != expected_unknown:
            raise ValueError(
                f"{name} unknown local predicate evidence is inconsistent"
            )

        expected_restricted = (
            name
            in RELEASED_RESTRICTED_SOURCES
        )
        if (
            source.get("restricted")
            is not expected_restricted
        ):
            raise ValueError(
                f"{name} restricted flag differs from released rule"
            )

        local_set = set(local)
        expected_allowed = (
            [
                value
                for value in union
                if value in local_set
            ]
            if expected_restricted
            else list(union)
        )
        if not expected_allowed:
            raise ValueError(
                f"restricted source {name} has no union predicates"
            )
        allowed = source.get(
            "allowed_predicates"
        )
        if allowed != expected_allowed:
            raise ValueError(
                f"{name} allowed predicates differ from released rule"
            )
        allowed_count = _require_positive_int(
            source.get(
                "allowed_predicate_count"
            ),
            f"{name} allowed_predicate_count",
        )
        if allowed_count != len(
            expected_allowed
        ):
            raise ValueError(
                f"{name} allowed predicate count is inconsistent"
            )
        allowed_sha = require_sha256(
            source.get(
                "allowed_predicate_order_sha256"
            ),
            f"{name} allowed_predicate_order_sha256",
        )
        if allowed_sha != _ordered_names_sha256(
            expected_allowed
        ):
            raise ValueError(
                f"{name} allowed predicate-order hash is inconsistent"
            )

        meta_sha = require_sha256(
            source.get("meta_sha256"),
            f"{name} source-column meta_sha256",
        )
        spatial_source = (
            spatial_by_name.get(name)
        )
        if (
            not isinstance(
                spatial_source,
                dict,
            )
            or spatial_source.get(
                "meta_sha256"
            )
            != meta_sha
        ):
            raise ValueError(
                f"{name} source-column derivation does not use the same pack meta as spatial derivation"
            )

        pack_split = require_non_empty_string(
            source.get("pack_split"),
            f"{name} source-column pack_split",
        )
        normalized_sources.append(
            {
                "source_name": name,
                "pack_split": pack_split,
                "meta_sha256": meta_sha,
                "local_predicates": list(
                    local
                ),
                "local_predicate_count": (
                    local_count
                ),
                "local_predicate_order_sha256": (
                    local_sha
                ),
                "unknown_local_predicates": list(
                    expected_unknown
                ),
                "restricted": (
                    expected_restricted
                ),
                "allowed_predicates": list(
                    expected_allowed
                ),
                "allowed_predicate_count": (
                    allowed_count
                ),
                "allowed_predicate_order_sha256": (
                    allowed_sha
                ),
            }
        )
        sidecar_sources.append(
            {
                "name": name,
                "predicates": list(
                    expected_allowed
                ),
            }
        )

    if tuple(names) != (
        RELEASED_SOURCE_NAMES
    ):
        raise ValueError(
            "source-column derivation source order differs from release"
        )

    sidecar_payload = {
        "schema": SOURCE_ALLOW_SCHEMA,
        "sources": sidecar_sources,
    }
    derived_sha = hashlib.sha256(
        stable_json_bytes(
            sidecar_payload
        )
    ).hexdigest()
    recorded_sha = require_sha256(
        payload.get(
            "sidecar_sha256"
        ),
        "source-column sidecar_sha256",
    )
    if recorded_sha != derived_sha:
        raise ValueError(
            "source-column derivation sidecar hash does not match reconstructed table"
        )
    if recorded_sha != sidecar_sha256:
        raise ValueError(
            "source-column derivation sidecar hash does not match corpus"
        )

    return {
        "schema": (
            SOURCE_COLUMN_DERIVATION_SCHEMA
        ),
        "source_order": list(
            RELEASED_SOURCE_NAMES
        ),
        "restricted_sources": list(
            RELEASED_RESTRICTED_SOURCES
        ),
        "union_predicate_order": list(
            union
        ),
        "union_predicate_count": (
            union_count
        ),
        "union_predicate_order_sha256": (
            union_sha
        ),
        "sources": normalized_sources,
        "sidecar_sha256": recorded_sha,
    }


def _validate_pair_opportunity_rebuild(
    payload: object,
    *,
    table_sha256: str,
    spatial_derivation: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "pair-opportunity rebuild evidence must be an object"
        )
    if (
        payload.get("schema")
        != PAIR_OPPORTUNITY_REBUILD_SCHEMA
    ):
        raise ValueError(
            "unsupported pair-opportunity rebuild schema"
        )
    if (
        payload.get("source_name")
        != PAIR_OPPORTUNITY_SOURCE_NAME
    ):
        raise ValueError(
            "pair-opportunity rebuild source must be megasg_clean"
        )
    if payload.get("scan_box_cap") != PAIR_OPPORTUNITY_SCAN_BOX_CAP:
        raise ValueError(
            "pair-opportunity rebuild requires scan_box_cap=400"
        )
    if payload.get("min_support") != PAIR_OPPORTUNITY_MIN_SUPPORT:
        raise ValueError(
            "pair-opportunity rebuild requires min_support=50"
        )
    if payload.get("full_scan") is not True:
        raise ValueError(
            "pair-opportunity rebuild requires a full scan"
        )
    if payload.get("extrapolated") is not False:
        raise ValueError(
            "pair-opportunity rebuild must not use extrapolation"
        )
    if (
        payload.get("opportunity_semantics")
        != "ordered-instance-pairs-minus-self-on-diagonal"
    ):
        raise ValueError(
            "pair-opportunity denominator semantics differ from Apache"
        )
    if (
        payload.get("numerator_semantics")
        != "same-pack-relations-after-400-box-cap"
    ):
        raise ValueError(
            "pair-opportunity numerator semantics differ from Apache"
        )
    if (
        payload.get("rate_semantics")
        != "min(1,relations/opportunities)"
    ):
        raise ValueError(
            "pair-opportunity rate semantics differ from Apache"
        )
    if (
        payload.get("npz_format")
        != PAIR_OPPORTUNITY_NPZ_FORMAT
    ):
        raise ValueError(
            "pair-opportunity deterministic NPZ format mismatch"
        )
    output_sha = require_sha256(
        payload.get("output_sha256"),
        "pair-opportunity output_sha256",
    )
    if output_sha != table_sha256:
        raise ValueError(
            "pair-opportunity rebuild output hash does not match corpus table"
        )

    components = payload.get(
        "component_sha256"
    )
    required_components = (
        "meta.json",
        "img_meta.npy",
        "box_cats.npy",
        "rels.npy",
    )
    if not isinstance(components, dict):
        raise ValueError(
            "pair-opportunity component hashes are missing"
        )
    normalized_components: dict[
        str,
        str
    ] = {}
    for name in required_components:
        normalized_components[
            name
        ] = require_sha256(
            components.get(name),
            f"pair-opportunity {name} SHA-256",
        )

    labels = payload.get(
        "object_label_order"
    )
    if (
        not isinstance(labels, list)
        or not labels
        or any(
            not isinstance(name, str)
            or not name
            for name in labels
        )
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(
            "pair-opportunity object-label order must be unique non-empty strings"
        )
    order_sha = require_sha256(
        payload.get(
            "object_label_order_sha256"
        ),
        "pair-opportunity object_label_order_sha256",
    )
    if order_sha != _ordered_names_sha256(
        labels
    ):
        raise ValueError(
            "pair-opportunity object-label order hash does not match labels"
        )
    num_cats = _require_positive_int(
        payload.get("num_cats"),
        "pair-opportunity num_cats",
    )
    if num_cats != len(labels):
        raise ValueError(
            "pair-opportunity num_cats does not match object-label order"
        )

    count_keys = (
        "images_total",
        "images_scanned",
        "boxes_scanned",
        "relations_scanned",
        "relations_dropped_by_box_cap",
        "category_pairs_with_opportunity",
        "trusted_category_pairs",
        "opportunity_sum",
        "relation_sum",
    )
    counts = {
        key: _require_non_negative_int(
            payload.get(key),
            f"pair-opportunity {key}",
        )
        for key in count_keys
    }
    if counts["images_total"] <= 0:
        raise ValueError(
            "pair-opportunity rebuild requires non-empty MegaSG pack"
        )
    if (
        counts["images_scanned"]
        != counts["images_total"]
    ):
        raise ValueError(
            "pair-opportunity released rebuild must scan every packed image"
        )
    pair_capacity = (
        num_cats
        * num_cats
    )
    if (
        counts[
            "category_pairs_with_opportunity"
        ]
        > pair_capacity
        or counts[
            "trusted_category_pairs"
        ]
        > counts[
            "category_pairs_with_opportunity"
        ]
    ):
        raise ValueError(
            "pair-opportunity category-pair summary counts are inconsistent"
        )
    if (
        counts["relation_sum"]
        != counts["relations_scanned"]
    ):
        raise ValueError(
            "pair-opportunity relation summary does not match numerator"
        )

    spatial_sources = spatial_derivation.get(
        "sources"
    )
    if not isinstance(
        spatial_sources,
        list,
    ):
        raise ValueError(
            "spatial derivation sources are unavailable"
        )
    megasg_source = next(
        (
            source
            for source in spatial_sources
            if isinstance(source, dict)
            and source.get(
                "source_name"
            )
            == PAIR_OPPORTUNITY_SOURCE_NAME
        ),
        None,
    )
    if megasg_source is None:
        raise ValueError(
            "spatial derivation does not contain megasg_clean source"
        )
    if (
        normalized_components[
            "meta.json"
        ]
        != megasg_source.get(
            "meta_sha256"
        )
        or normalized_components[
            "rels.npy"
        ]
        != megasg_source.get(
            "rels_sha256"
        )
    ):
        raise ValueError(
            "pair-opportunity rebuild does not use the same MegaSG pack as spatial derivation"
        )

    return {
        "schema": (
            PAIR_OPPORTUNITY_REBUILD_SCHEMA
        ),
        "source_name": (
            PAIR_OPPORTUNITY_SOURCE_NAME
        ),
        "pack_split": (
            require_non_empty_string(
                payload.get(
                    "pack_split"
                ),
                "pair-opportunity pack_split",
            )
        ),
        "component_sha256": (
            normalized_components
        ),
        "object_label_order": list(
            labels
        ),
        "object_label_order_sha256": (
            order_sha
        ),
        "num_cats": num_cats,
        "scan_box_cap": (
            PAIR_OPPORTUNITY_SCAN_BOX_CAP
        ),
        "min_support": (
            PAIR_OPPORTUNITY_MIN_SUPPORT
        ),
        "full_scan": True,
        "extrapolated": False,
        "opportunity_semantics": (
            "ordered-instance-pairs-minus-self-on-diagonal"
        ),
        "numerator_semantics": (
            "same-pack-relations-after-400-box-cap"
        ),
        "rate_semantics": (
            "min(1,relations/opportunities)"
        ),
        **counts,
        "npz_format": (
            PAIR_OPPORTUNITY_NPZ_FORMAT
        ),
        "output_sha256": (
            output_sha
        ),
    }


def _validate_spatial_derivation(
    payload: object,
    *,
    sidecar_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "predicate spatial-flags derivation must be an object"
        )
    if payload.get("schema") != DERIVATION_SCHEMA:
        raise ValueError(
            "unsupported predicate spatial-flags derivation schema"
        )
    if payload.get("algorithm") != DERIVATION_ALGORITHM:
        raise ValueError(
            "predicate spatial-flags derivation algorithm differs from Apache"
        )
    if payload.get("spatial_bit") != SPATIAL_BIT:
        raise ValueError(
            "predicate spatial-flags derivation must use relation flag bit 0"
        )
    threshold = _finite_number(
        payload.get("majority_threshold"),
        "predicate spatial majority threshold",
    )
    if not math.isclose(
        threshold,
        MAJORITY_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "predicate spatial majority threshold must be 0.5"
        )
    if (
        payload.get("majority_comparator")
        != MAJORITY_COMPARATOR
    ):
        raise ValueError(
            "predicate spatial majority comparator must be >="
        )
    derived_sha = require_sha256(
        payload.get("sidecar_sha256"),
        "predicate spatial sidecar_sha256",
    )
    if derived_sha != sidecar_sha256:
        raise ValueError(
            "predicate spatial derivation sidecar hash does not match corpus"
        )

    sources = payload.get("sources")
    if (
        not isinstance(sources, list)
        or len(sources)
        != len(RELEASED_SOURCE_NAMES)
    ):
        raise ValueError(
            "predicate spatial derivation must describe all released sources"
        )
    normalized_sources: list[
        dict[str, Any]
    ] = []
    names: list[str] = []
    for index, source in enumerate(
        sources
    ):
        if not isinstance(source, dict):
            raise ValueError(
                f"predicate spatial source {index} must be an object"
            )
        name = require_non_empty_string(
            source.get("source_name"),
            f"predicate spatial source {index} name",
        )
        names.append(name)
        local_predicate_count = _require_positive_int(
            source.get(
                "local_predicate_count"
            ),
            f"{name} local_predicate_count",
        )
        supported = _require_non_negative_int(
            source.get(
                "supported_predicate_count"
            ),
            f"{name} supported_predicate_count",
        )
        spatial = _require_non_negative_int(
            source.get(
                "local_spatial_majority_count"
            ),
            f"{name} local_spatial_majority_count",
        )
        relations = _require_non_negative_int(
            source.get("relations"),
            f"{name} relations",
        )
        if (
            supported > local_predicate_count
            or spatial > supported
        ):
            raise ValueError(
                f"{name} predicate spatial counts are inconsistent"
            )
        ignored = source.get(
            "ignored_predicates"
        )
        if (
            not isinstance(ignored, list)
            or any(
                not isinstance(value, str)
                or not value
                for value in ignored
            )
        ):
            raise ValueError(
                f"{name} ignored_predicates must be strings"
            )
        normalized_sources.append(
            {
                "source_name": name,
                "pack_split": require_non_empty_string(
                    source.get("pack_split"),
                    f"{name} pack_split",
                ),
                "meta_sha256": require_sha256(
                    source.get("meta_sha256"),
                    f"{name} meta_sha256",
                ),
                "rels_sha256": require_sha256(
                    source.get("rels_sha256"),
                    f"{name} rels_sha256",
                ),
                "relations": relations,
                "local_predicate_count": (
                    local_predicate_count
                ),
                "supported_predicate_count": (
                    supported
                ),
                "local_spatial_majority_count": (
                    spatial
                ),
                "ignored_predicates": list(
                    ignored
                ),
            }
        )
    if tuple(names) != RELEASED_SOURCE_NAMES:
        raise ValueError(
            "predicate spatial derivation source order differs from release"
        )

    union_count = _require_positive_int(
        payload.get(
            "union_predicate_count"
        ),
        "union_predicate_count",
    )
    supported_union = _require_positive_int(
        payload.get(
            "supported_union_predicate_count"
        ),
        "supported_union_predicate_count",
    )
    spatial_union = _require_positive_int(
        payload.get(
            "spatial_union_predicate_count"
        ),
        "spatial_union_predicate_count",
    )
    if (
        supported_union > union_count
        or spatial_union > supported_union
    ):
        raise ValueError(
            "predicate spatial union counts are inconsistent"
        )
    unsupported = payload.get(
        "unsupported_predicates"
    )
    if (
        not isinstance(unsupported, list)
        or any(
            not isinstance(value, str)
            or not value
            for value in unsupported
        )
    ):
        raise ValueError(
            "unsupported_predicates must be strings"
        )

    return {
        "schema": DERIVATION_SCHEMA,
        "algorithm": DERIVATION_ALGORITHM,
        "spatial_bit": SPATIAL_BIT,
        "majority_threshold": (
            MAJORITY_THRESHOLD
        ),
        "majority_comparator": (
            MAJORITY_COMPARATOR
        ),
        "sources": normalized_sources,
        "union_predicate_count": union_count,
        "supported_union_predicate_count": (
            supported_union
        ),
        "spatial_union_predicate_count": (
            spatial_union
        ),
        "unsupported_predicates": list(
            unsupported
        ),
        "sidecar_sha256": derived_sha,
    }


def validate_corpus(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("schema") != CORPUS_SCHEMA:
        raise ValueError(
            "unsupported released corpus schema"
        )
    if (
        payload.get("reference_source_commit")
        != REFERENCE_SOURCE_COMMIT
    ):
        raise ValueError(
            "released corpus uses the wrong Apache source reference"
        )

    sources = payload.get("sources")
    if (
        not isinstance(sources, list)
        or len(sources) != len(RELEASED_SOURCE_NAMES)
    ):
        raise ValueError(
            "released corpus must describe exactly three sources"
        )

    normalized_sources: list[dict[str, Any]] = []
    for index, (source, expected_name) in enumerate(
        zip(sources, RELEASED_SOURCE_NAMES)
    ):
        if not isinstance(source, dict):
            raise ValueError(
                f"source {index} must be an object"
            )
        name = require_non_empty_string(
            source.get("name"),
            f"source {index} name",
        )
        if name != expected_name:
            raise ValueError(
                "released corpus source order/name mismatch"
            )
        provenance_kind = require_non_empty_string(
            source.get("provenance_kind"),
            f"{name} provenance_kind",
        )
        if provenance_kind not in PROVENANCE_KINDS:
            raise ValueError(
                f"unsupported provenance kind for {name}"
            )

        normalized: dict[str, Any] = {
            "name": name,
            "provenance_kind": provenance_kind,
        }

        if provenance_kind == "unresolved":
            normalized["note"] = require_non_empty_string(
                source.get("note"),
                f"{name} unresolved note",
            )
            normalized["resolved"] = False
        else:
            normalized["origin"] = require_non_empty_string(
                source.get("origin"),
                f"{name} origin",
            )
            normalized["revision"] = require_non_empty_string(
                source.get("revision"),
                f"{name} revision",
            )
            if provenance_kind == "deterministic-rebuild":
                normalized["recipe"] = (
                    require_non_empty_string(
                        source.get("recipe"),
                        f"{name} rebuild recipe",
                    )
                )
            normalized["annotations_sha256"] = (
                require_sha256(
                    source.get("annotations_sha256"),
                    f"{name} annotations_sha256",
                )
            )
            normalized["post_exclusion_count"] = (
                _require_positive_int(
                    source.get("post_exclusion_count"),
                    f"{name} post_exclusion_count",
                )
            )
            normalized["resolved"] = True

        normalized_sources.append(normalized)

    normalized_payload = {
        "schema": CORPUS_SCHEMA,
        "reference_source_commit": (
            REFERENCE_SOURCE_COMMIT
        ),
        "sources": normalized_sources,
        "exclude_ids_sha256": require_sha256(
            payload.get("exclude_ids_sha256"),
            "exclude_ids_sha256",
        ),
        "source_column_allow_sha256": require_sha256(
            payload.get("source_column_allow_sha256"),
            "source_column_allow_sha256",
        ),
        "ontology_meta_sha256": require_sha256(
            payload.get("ontology_meta_sha256"),
            "ontology_meta_sha256",
        ),
        "ontology_npz_sha256": require_sha256(
            payload.get("ontology_npz_sha256"),
            "ontology_npz_sha256",
        ),
        "neg_rate_table_sha256": require_sha256(
            payload.get("neg_rate_table_sha256"),
            "neg_rate_table_sha256",
        ),
        "predicate_embeddings_sha256": require_sha256(
            payload.get("predicate_embeddings_sha256"),
            "predicate_embeddings_sha256",
        ),
        "object_embeddings_sha256": require_sha256(
            payload.get("object_embeddings_sha256"),
            "object_embeddings_sha256",
        ),
        "predicate_spatial_flags_sha256": require_sha256(
            payload.get("predicate_spatial_flags_sha256"),
            "predicate_spatial_flags_sha256",
        ),
        "vocabulary_sha256": require_sha256(
            payload.get("vocabulary_sha256"),
            "vocabulary_sha256",
        ),
    }
    normalized_payload[
        "exclude_ids_derivation"
    ] = _validate_indoorvg_holdout_derivation(
        payload.get(
            "exclude_ids_derivation"
        ),
        output_sha256=normalized_payload[
            "exclude_ids_sha256"
        ],
    )
    normalized_payload[
        "predicate_spatial_flags_derivation"
    ] = _validate_spatial_derivation(
        payload.get(
            "predicate_spatial_flags_derivation"
        ),
        sidecar_sha256=normalized_payload[
            "predicate_spatial_flags_sha256"
        ],
    )
    normalized_payload[
        "pair_opportunity_rebuild"
    ] = _validate_pair_opportunity_rebuild(
        payload.get(
            "pair_opportunity_rebuild"
        ),
        table_sha256=normalized_payload[
            "neg_rate_table_sha256"
        ],
        spatial_derivation=normalized_payload[
            "predicate_spatial_flags_derivation"
        ],
    )
    normalized_payload[
        "text_bank_derivation"
    ] = _validate_text_bank_derivation(
        payload.get(
            "text_bank_derivation"
        ),
        predicate_artifact_sha256=normalized_payload[
            "predicate_embeddings_sha256"
        ],
        object_artifact_sha256=normalized_payload[
            "object_embeddings_sha256"
        ],
    )
    normalized_payload[
        "source_column_derivation"
    ] = _validate_source_column_derivation(
        payload.get(
            "source_column_derivation"
        ),
        sidecar_sha256=normalized_payload[
            "source_column_allow_sha256"
        ],
        spatial_derivation=normalized_payload[
            "predicate_spatial_flags_derivation"
        ],
        predicate_order_sha256=normalized_payload[
            "text_bank_derivation"
        ]["predicate_bank"][
            "label_order_sha256"
        ],
    )
    return normalized_payload


def unresolved_sources(
    corpus: dict[str, Any],
) -> tuple[str, ...]:
    normalized = validate_corpus(corpus)
    return tuple(
        source["name"]
        for source in normalized["sources"]
        if not source["resolved"]
    )


def _require_released_fractions(
    values: object,
) -> None:
    if (
        not isinstance(values, list)
        or len(values) != len(RELEASED_MIX_FRACTIONS)
    ):
        raise ValueError(
            "training target fractions are missing"
        )
    for actual, expected in zip(
        values,
        RELEASED_MIX_FRACTIONS,
    ):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "training target fractions differ from released recipe"
            )


def validate_training_run(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("schema") != TRAINING_SCHEMA:
        raise ValueError(
            "unsupported relation training run schema"
        )
    if payload.get("frozen_backbone") is not False:
        raise ValueError(
            "released qualification requires full backbone fine-tuning"
        )

    recipe = payload.get("training_recipe")
    if not isinstance(recipe, dict):
        raise ValueError(
            "training recipe evidence is missing"
        )
    if recipe.get("name") != "apache-reference":
        raise ValueError(
            "training run is not the Apache reference recipe"
        )
    if recipe.get("matches_released_epoch_count") is not True:
        raise ValueError(
            "training run did not execute the released 12 epochs"
        )
    config = recipe.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            "Apache recipe config is missing"
        )
    if config.get("epochs") != RELEASED_EPOCHS:
        raise ValueError(
            "Apache recipe epoch count differs from release"
        )
    if config.get("micro_batch_size") != RELEASED_MICRO_BATCH_SIZE:
        raise ValueError(
            "Apache micro-batch size differs from release"
        )
    if config.get("grad_accum") != RELEASED_WORLD_SIZE:
        raise ValueError(
            "logical rank/gradient accumulation differs from release"
        )
    if config.get("text_dim") != RELEASED_TEXT_DIM:
        raise ValueError(
            "Apache recipe config requires text_dim=512"
        )

    if config.get("image_size") != RELEASED_IMAGE_SIZE:
        raise ValueError(
            "Apache recipe config requires img_size=448"
        )
    if config.get("geo_budget") != RELEASED_GEO_BUDGET:
        raise ValueError(
            "Apache recipe config requires geo_budget=400"
        )
    if config.get("final_budget") != RELEASED_FINAL_BUDGET:
        raise ValueError(
            "Apache recipe config requires final_budget=128"
        )
    if config.get("amp") is not RELEASED_AMP:
        raise ValueError(
            "Apache recipe config requires amp=true"
        )
    if config.get("amp_dtype") != RELEASED_AMP_DTYPE:
        raise ValueError(
            "Apache recipe config requires amp_dtype=bf16"
        )

    config_augment = config.get("augment")
    if (
        isinstance(config_augment, bool)
        or not isinstance(config_augment, (int, float))
        or not math.isclose(
            float(config_augment),
            RELEASED_PHOTOMETRIC_AUGMENT,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "Apache recipe config requires augment=0.3"
        )

    config_ema_decay = config.get("ema_decay")
    if (
        isinstance(config_ema_decay, bool)
        or not isinstance(config_ema_decay, (int, float))
        or not math.isclose(
            float(config_ema_decay),
            RELEASED_EMA_DECAY,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "Apache recipe config requires ema_decay=0.9998"
        )
    if (
        recipe.get("effective_batch_size")
        != RELEASED_EFFECTIVE_BATCH_SIZE
    ):
        raise ValueError(
            "effective batch size differs from release"
        )
    if recipe.get("weight_source") != "ema":
        raise ValueError(
            "released qualification requires EMA checkpoint weights"
        )
    ema = recipe.get("ema")
    if not isinstance(ema, dict):
        raise ValueError(
            "released qualification requires EMA evidence"
        )
    if ema.get("enabled") is not True:
        raise ValueError(
            "released qualification requires EMA enabled"
        )
    decay = ema.get("decay")
    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or not math.isclose(
            float(decay),
            RELEASED_EMA_DECAY,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "released qualification requires ema_decay=0.9998"
        )
    updates = ema.get("updates")
    if (
        isinstance(updates, bool)
        or not isinstance(updates, int)
        or updates <= 0
    ):
        raise ValueError(
            "released qualification requires positive EMA update count"
        )
    require_sha256(
        ema.get("state_sha256"),
        "EMA state_sha256",
    )
    require_sha256(
        ema.get("raw_state_sha256"),
        "raw state_sha256",
    )
    if ema["state_sha256"] == ema["raw_state_sha256"]:
        raise ValueError(
            "released EMA state must differ from raw training state"
        )

    augmentation = recipe.get("augmentation")
    if not isinstance(augmentation, dict):
        raise ValueError(
            "released qualification requires augmentation evidence"
        )
    if augmentation.get("kind") != "brightness-contrast-saturation":
        raise ValueError(
            "released qualification requires photometric augmentation contract"
        )
    strength = augmentation.get("strength")
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isclose(
            float(strength),
            RELEASED_PHOTOMETRIC_AUGMENT,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "released qualification requires augment=0.3"
        )
    if (
        augmentation.get("horizontal_flip") is not False
        or augmentation.get("geometry_transform") is not False
    ):
        raise ValueError(
            "released augmentation must preserve geometry and direction"
        )
    if augmentation.get("rng_source") != "ambient-torch-rng":
        raise ValueError(
            "released augmentation RNG source differs from Apache contract"
        )
    if augmentation.get("rng_equivalence") != "stochastic-distribution":
        raise ValueError(
            "released augmentation must claim stochastic-distribution equivalence"
        )

    precision = recipe.get("precision")
    if not isinstance(precision, dict):
        raise ValueError(
            "released qualification requires precision evidence"
        )
    if precision.get("amp_requested") is not True:
        raise ValueError(
            "released qualification requires AMP enabled"
        )
    if precision.get("amp_dtype") != RELEASED_AMP_DTYPE:
        raise ValueError(
            "released qualification requires BF16 AMP"
        )
    if precision.get("autocast_device_type") != "cuda":
        raise ValueError(
            "released qualification requires CUDA autocast"
        )
    if precision.get("autocast_executed") is not True:
        raise ValueError(
            "released qualification requires executed autocast"
        )
    if precision.get("grad_scaler") is not False:
        raise ValueError(
            "released BF16 contract must not use GradScaler"
        )
    if precision.get("released_cuda_execution") is not True:
        raise ValueError(
            "released qualification requires CUDA BF16 execution"
        )
    device = payload.get("device")
    if (
        not isinstance(device, str)
        or not device.startswith("cuda")
    ):
        raise ValueError(
            "released qualification requires a CUDA training device"
        )

    sampler_budget = recipe.get("sampler_budget")
    if not isinstance(sampler_budget, dict):
        raise ValueError(
            "released qualification requires sampler budget evidence"
        )
    if (
        sampler_budget.get("geo_budget") != RELEASED_GEO_BUDGET
        or sampler_budget.get("final_budget") != RELEASED_FINAL_BUDGET
        or sampler_budget.get("matches_released") is not True
    ):
        raise ValueError(
            "released qualification requires sampler budget 400->128"
        )

    mixture = payload.get("train_mixture")
    if not isinstance(mixture, dict):
        raise ValueError(
            "training mixture evidence is missing"
        )
    if mixture.get("source_names") != list(
        RELEASED_SOURCE_NAMES
    ):
        raise ValueError(
            "training source order differs from release"
        )
    _require_released_fractions(
        mixture.get("target_fractions")
    )
    if mixture.get("seed") != RELEASED_SEED:
        raise ValueError(
            "training mixture seed differs from release"
        )
    if (
        mixture.get("draws_per_epoch")
        != RELEASED_SAMPLES_PER_EPOCH
    ):
        raise ValueError(
            "training draw count differs from release"
        )
    if (
        mixture.get("matches_released_sampling_stream")
        is not True
    ):
        raise ValueError(
            "released sampling stream was not qualified"
        )

    history = payload.get("history")
    if (
        not isinstance(history, list)
        or len(history) != RELEASED_EPOCHS
    ):
        raise ValueError(
            "training history must contain all 12 released epochs"
        )
    expected_epochs = list(
        range(1, RELEASED_EPOCHS + 1)
    )
    actual_epochs = [
        row.get("epoch")
        if isinstance(row, dict)
        else None
        for row in history
    ]
    if actual_epochs != expected_epochs:
        raise ValueError(
            "training history epoch sequence is incomplete"
        )

    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(
            "model config evidence is missing"
        )
    if model_config.get("image_size") != RELEASED_IMAGE_SIZE:
        raise ValueError(
            "Apache released qualification requires image_size=448"
        )
    if model_config.get("max_boxes") != RELEASED_MAX_BOXES:
        raise ValueError(
            "Apache released qualification requires max_boxes=40"
        )
    if model_config.get("pair_budget") != RELEASED_FINAL_BUDGET:
        raise ValueError(
            "Apache released qualification requires pair_budget=128"
        )
    if model_config.get("hidden_dim") != RELEASED_D_MODEL:
        raise ValueError(
            "Apache released qualification requires hidden_dim=512"
        )
    for key in (
        "pair_evidence_contract",
        "pair_sampler_contract",
        "relation_context_contract",
        "predicate_head_contract",
    ):
        if model_config.get(key) != "apache":
            raise ValueError(
                f"{key} is not the Apache contract"
            )

    require_sha256(
        payload.get("checkpoint_sha256"),
        "checkpoint_sha256",
    )
    predicate_shape = payload.get(
        "predicate_embedding_shape"
    )
    if (
        not isinstance(predicate_shape, list)
        or len(predicate_shape) != 2
        or predicate_shape[1] != RELEASED_TEXT_DIM
    ):
        raise ValueError(
            "released qualification requires predicate text_dim=512"
        )
    require_sha256(
        payload.get("predicate_embeddings_sha256"),
        "predicate_embeddings_sha256",
    )
    predicate_bank = payload.get(
        "predicate_bank"
    )
    if not isinstance(predicate_bank, dict):
        raise ValueError(
            "released qualification requires named predicate-bank evidence"
        )
    if (
        predicate_bank.get("schema")
        != PREDICATE_BANK_SCHEMA
    ):
        raise ValueError(
            "unsupported predicate-bank provenance schema"
        )
    if (
        predicate_bank.get("artifact_sha256")
        != payload[
            "predicate_embeddings_sha256"
        ]
    ):
        raise ValueError(
            "predicate-bank artifact hash does not match training input"
        )
    if predicate_bank.get("shape") != predicate_shape:
        raise ValueError(
            "predicate-bank shape does not match training embedding shape"
        )
    if (
        predicate_bank.get("text_dim")
        != RELEASED_TEXT_DIM
    ):
        raise ValueError(
            "predicate-bank text_dim must be 512"
        )
    if (
        predicate_bank.get("templates")
        != list(
            RELEASED_PREDICATE_TEMPLATES
        )
    ):
        raise ValueError(
            "predicate-bank templates differ from release"
        )
    predicate_order = predicate_bank.get(
        "predicate_order"
    )
    if (
        not isinstance(predicate_order, list)
        or len(predicate_order)
        != predicate_shape[0]
        or any(
            not isinstance(name, str)
            or not name
            for name in predicate_order
        )
        or len(set(predicate_order))
        != len(predicate_order)
    ):
        raise ValueError(
            "predicate-bank order must be unique non-empty strings"
        )
    if (
        predicate_bank.get(
            "predicate_order_sha256"
        )
        != _ordered_names_sha256(
            predicate_order
        )
    ):
        raise ValueError(
            "predicate-bank order hash does not match predicate order"
        )
    require_sha256(
        payload.get("vocabulary_sha256"),
        "vocabulary_sha256",
    )

    objective = payload.get(
        "apache_reference_objective"
    )
    if not isinstance(objective, dict):
        raise ValueError(
            "Apache reference objective evidence is missing"
        )
    objective_config = objective.get("config")
    if not isinstance(objective_config, dict):
        raise ValueError(
            "Apache reference objective config is missing"
        )
    scalar_contract = (
        validate_released_scalar_contract(
            recipe=config,
            model=model_config,
            objective=objective_config,
        )
    )
    if recipe.get("scalar_contract") != scalar_contract:
        raise ValueError(
            "released scalar-contract evidence does not match resolved config"
        )
    structure_contract = recipe.get(
        "structure_contract"
    )
    if not isinstance(structure_contract, dict):
        raise ValueError(
            "released structure-contract evidence is missing"
        )
    validate_released_structure_report(
        structure_contract
    )

    assets = objective.get("assets")
    if not isinstance(assets, dict):
        raise ValueError(
            "Apache objective asset evidence is missing"
        )
    for key in (
        "source_column_allow_sha256",
        "ontology_meta_sha256",
        "ontology_npz_sha256",
        "neg_rate_table_sha256",
        "object_embeddings_sha256",
    ):
        require_sha256(
            assets.get(key),
            key,
        )

    object_shape = assets.get(
        "object_embeddings_shape"
    )
    object_order = assets.get(
        "object_label_order"
    )
    if (
        not isinstance(object_shape, list)
        or len(object_shape) != 2
        or object_shape[1] != RELEASED_TEXT_DIM
        or not isinstance(object_shape[0], int)
        or isinstance(object_shape[0], bool)
        or object_shape[0] <= 0
    ):
        raise ValueError(
            "released qualification requires object embeddings [O,512]"
        )
    if (
        not isinstance(object_order, list)
        or len(object_order) != object_shape[0]
        or any(
            not isinstance(name, str) or not name
            for name in object_order
        )
        or len(set(object_order)) != len(object_order)
    ):
        raise ValueError(
            "released qualification requires exact object-label order evidence"
        )
    object_bank = assets.get("object_bank")
    if (
        not isinstance(object_bank, dict)
        or object_bank.get("schema")
        != "kfcore.apache-object-text-bank/1"
        or object_bank.get("artifact_sha256")
        != assets["object_embeddings_sha256"]
        or object_bank.get("shape")
        != object_shape
        or object_bank.get("object_label_order")
        != object_order
        or object_bank.get("text_dim")
        != RELEASED_TEXT_DIM
    ):
        raise ValueError(
            "released qualification requires named object-bank provenance"
        )
    if (
        object_bank.get(
            "object_label_order_sha256"
        )
        != _ordered_names_sha256(
            object_order
        )
    ):
        raise ValueError(
            "object-bank order hash does not match object-label order"
        )

    source_hashes = mixture.get(
        "source_annotation_sha256"
    )
    source_counts = mixture.get("source_counts")
    if (
        not isinstance(source_hashes, list)
        or len(source_hashes)
        != len(RELEASED_SOURCE_NAMES)
    ):
        raise ValueError(
            "source annotation hashes are missing"
        )
    if (
        not isinstance(source_counts, list)
        or len(source_counts)
        != len(RELEASED_SOURCE_NAMES)
    ):
        raise ValueError(
            "source post-exclusion counts are missing"
        )
    for index, value in enumerate(source_hashes):
        require_sha256(
            value,
            f"source {index} annotation SHA-256",
        )
    for index, value in enumerate(source_counts):
        _require_positive_int(
            value,
            f"source {index} count",
        )
    require_sha256(
        mixture.get("exclude_ids_sha256"),
        "training exclude_ids_sha256",
    )

    routing = payload.get(
        "routing_warm_start"
    )
    if not isinstance(routing, dict):
        raise ValueError(
            "released qualification requires routing warm-start evidence"
        )
    if (
        routing.get("schema")
        != "kfcore.apache-routing-warm-start/1"
    ):
        raise ValueError(
            "unsupported routing warm-start schema"
        )
    require_sha256(
        routing.get(
            "spatial_flags_sha256"
        ),
        "routing spatial_flags_sha256",
    )
    spatial_count = _require_positive_int(
        routing.get("spatial_count"),
        "routing spatial_count",
    )
    semantic_count = _require_positive_int(
        routing.get("semantic_count"),
        "routing semantic_count",
    )
    if (
        spatial_count
        + semantic_count
        != predicate_shape[0]
    ):
        raise ValueError(
            "routing spatial/semantic counts do not match predicate vocabulary"
        )
    for key in (
        "target_alpha_mean",
        "target_alpha_spatial_mean",
        "target_alpha_semantic_mean",
        "mse_before",
        "mse_after",
        "reported_final_mse",
    ):
        value = _finite_number(
            routing.get(key),
            f"routing {key}",
        )
        if key.startswith("target_alpha") and not (
            0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"routing {key} must be within [0,1]"
            )
        if key.startswith("mse_") or key == "reported_final_mse":
            if value < 0.0:
                raise ValueError(
                    f"routing {key} must be non-negative"
                )
    if not (
        float(routing["mse_after"])
        < float(routing["mse_before"])
    ):
        raise ValueError(
            "routing warm start must improve MSE"
        )

    return dict(payload)


def qualify_training_run(
    corpus: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    corpus = validate_corpus(corpus)
    training = validate_training_run(training)

    missing = tuple(
        source["name"]
        for source in corpus["sources"]
        if not source["resolved"]
    )
    if missing:
        raise ValueError(
            "released corpus provenance is unresolved for: "
            + ", ".join(missing)
        )

    mixture = training["train_mixture"]
    expected_hashes = [
        source["annotations_sha256"]
        for source in corpus["sources"]
    ]
    expected_counts = [
        source["post_exclusion_count"]
        for source in corpus["sources"]
    ]
    if (
        mixture["source_annotation_sha256"]
        != expected_hashes
    ):
        raise ValueError(
            "training source annotation hashes do not match corpus manifest"
        )
    if mixture["source_counts"] != expected_counts:
        raise ValueError(
            "training source counts do not match corpus manifest"
        )
    if (
        mixture["exclude_ids_sha256"]
        != corpus["exclude_ids_sha256"]
    ):
        raise ValueError(
            "training exclusion artifact does not match corpus manifest"
        )

    objective_assets = training[
        "apache_reference_objective"
    ]["assets"]
    asset_pairs = (
        (
            "source_column_allow_sha256",
            "source_column_allow_sha256",
        ),
        (
            "ontology_meta_sha256",
            "ontology_meta_sha256",
        ),
        (
            "ontology_npz_sha256",
            "ontology_npz_sha256",
        ),
        (
            "neg_rate_table_sha256",
            "neg_rate_table_sha256",
        ),
        (
            "object_embeddings_sha256",
            "object_embeddings_sha256",
        ),
    )
    for training_key, corpus_key in asset_pairs:
        if (
            objective_assets[training_key]
            != corpus[corpus_key]
        ):
            raise ValueError(
                f"{training_key} does not match corpus manifest"
            )

    pair_rebuild = corpus[
        "pair_opportunity_rebuild"
    ]
    if (
        pair_rebuild[
            "object_label_order"
        ]
        != objective_assets[
            "object_label_order"
        ]
    ):
        raise ValueError(
            "pair-opportunity category order does not match training object vocabulary"
        )
    if (
        pair_rebuild["num_cats"]
        != objective_assets[
            "object_embeddings_shape"
        ][0]
    ):
        raise ValueError(
            "pair-opportunity category count does not match object text bank"
        )

    if (
        training["predicate_embeddings_sha256"]
        != corpus["predicate_embeddings_sha256"]
    ):
        raise ValueError(
            "predicate embedding asset does not match corpus manifest"
        )
    if (
        training["vocabulary_sha256"]
        != corpus["vocabulary_sha256"]
    ):
        raise ValueError(
            "vocabulary asset does not match corpus manifest"
        )
    text_derivation = corpus[
        "text_bank_derivation"
    ]
    predicate_bank = training[
        "predicate_bank"
    ]
    object_bank = objective_assets[
        "object_bank"
    ]
    if (
        predicate_bank[
            "predicate_order_sha256"
        ]
        != text_derivation[
            "predicate_bank"
        ]["label_order_sha256"]
    ):
        raise ValueError(
            "predicate-bank order does not match text-bank derivation"
        )
    if (
        object_bank[
            "object_label_order_sha256"
        ]
        != text_derivation[
            "object_bank"
        ]["label_order_sha256"]
    ):
        raise ValueError(
            "object-bank order does not match text-bank derivation"
        )
    if (
        text_derivation[
            "predicate_bank"
        ]["label_count"]
        != training[
            "predicate_embedding_shape"
        ][0]
    ):
        raise ValueError(
            "predicate-bank derivation count does not match training vocabulary"
        )
    if (
        text_derivation[
            "object_bank"
        ]["label_count"]
        != objective_assets[
            "object_embeddings_shape"
        ][0]
    ):
        raise ValueError(
            "object-bank derivation count does not match training vocabulary"
        )

    routing = training[
        "routing_warm_start"
    ]
    if (
        routing["spatial_flags_sha256"]
        != corpus[
            "predicate_spatial_flags_sha256"
        ]
    ):
        raise ValueError(
            "predicate spatial-flags hash does not match corpus manifest"
        )
    derivation = corpus[
        "predicate_spatial_flags_derivation"
    ]
    predicate_count = training[
        "predicate_embedding_shape"
    ][0]
    if (
        derivation["union_predicate_count"]
        != predicate_count
    ):
        raise ValueError(
            "predicate spatial derivation vocabulary size does not match training"
        )
    if (
        derivation[
            "spatial_union_predicate_count"
        ]
        != routing["spatial_count"]
    ):
        raise ValueError(
            "predicate spatial derivation count does not match routing evidence"
        )
    if (
        predicate_count
        - derivation[
            "spatial_union_predicate_count"
        ]
        != routing["semantic_count"]
    ):
        raise ValueError(
            "predicate semantic derivation count does not match routing evidence"
        )

    return {
        "schema": QUALIFICATION_SCHEMA,
        "qualified": True,
        "reference_source_commit": (
            REFERENCE_SOURCE_COMMIT
        ),
        "corpus_sha256": payload_sha256(corpus),
        "training_sha256": payload_sha256(training),
        "checkpoint_sha256": training[
            "checkpoint_sha256"
        ],
        "source_names": list(
            RELEASED_SOURCE_NAMES
        ),
        "source_counts": expected_counts,
        "source_annotation_sha256": (
            expected_hashes
        ),
        "exclude_ids_sha256": corpus[
            "exclude_ids_sha256"
        ],
        "exclude_ids_derivation": corpus[
            "exclude_ids_derivation"
        ],
        "epochs": RELEASED_EPOCHS,
        "draws_per_epoch": (
            RELEASED_SAMPLES_PER_EPOCH
        ),
        "effective_batch_size": (
            RELEASED_EFFECTIVE_BATCH_SIZE
        ),
        "weight_source": "ema",
        "ema_decay": RELEASED_EMA_DECAY,
        "ema_updates": training[
            "training_recipe"
        ]["ema"]["updates"],
        "ema_state_sha256": training[
            "training_recipe"
        ]["ema"]["state_sha256"],
        "augmentation": training[
            "training_recipe"
        ]["augmentation"],
        "text_dim": RELEASED_TEXT_DIM,
        "image_size": RELEASED_IMAGE_SIZE,
        "geo_budget": RELEASED_GEO_BUDGET,
        "final_budget": RELEASED_FINAL_BUDGET,
        "object_embeddings_sha256": corpus[
            "object_embeddings_sha256"
        ],
        "object_embeddings_shape": training[
            "apache_reference_objective"
        ]["assets"]["object_embeddings_shape"],
        "predicate_spatial_flags_sha256": corpus[
            "predicate_spatial_flags_sha256"
        ],
        "predicate_spatial_flags_derivation": corpus[
            "predicate_spatial_flags_derivation"
        ],
        "pair_opportunity_rebuild": corpus[
            "pair_opportunity_rebuild"
        ],
        "text_bank_derivation": corpus[
            "text_bank_derivation"
        ],
        "source_column_derivation": corpus[
            "source_column_derivation"
        ],
        "precision": training[
            "training_recipe"
        ]["precision"],
        "scalar_contract": training[
            "training_recipe"
        ]["scalar_contract"],
        "structure_contract": training[
            "training_recipe"
        ]["structure_contract"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    corpus = load_json(args.corpus)
    training = load_json(args.training)
    result = qualify_training_run(
        corpus,
        training,
    )
    Path(args.out).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
