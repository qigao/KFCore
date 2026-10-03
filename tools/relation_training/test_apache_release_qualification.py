from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import unittest

from apache_indoorvg_holdout import (
    DERIVATION_SCHEMA as INDOORVG_DERIVATION_SCHEMA,
    RELEASED_NOTE as INDOORVG_RELEASED_NOTE,
    RELEASED_SOURCE as INDOORVG_RELEASED_SOURCE,
    RELEASED_SPLITS as INDOORVG_RELEASED_SPLITS,
    holdout_bytes as indoorvg_holdout_bytes,
    stable_json_sha256 as indoorvg_stable_json_sha256,
)
from apache_objective import (
    SOURCE_ALLOW_SCHEMA,
)
from apache_source_columns import (
    DERIVATION_SCHEMA as SOURCE_COLUMN_DERIVATION_SCHEMA,
    RELEASED_RESTRICTED_SOURCES,
    stable_json_bytes as source_column_json_bytes,
)
from apache_pair_opportunity import (
    NPZ_FORMAT as PAIR_OPPORTUNITY_NPZ_FORMAT,
    REBUILD_SCHEMA as PAIR_OPPORTUNITY_REBUILD_SCHEMA,
    RELEASED_MIN_SUPPORT as PAIR_OPPORTUNITY_MIN_SUPPORT,
    RELEASED_SCAN_BOX_CAP as PAIR_OPPORTUNITY_SCAN_BOX_CAP,
    RELEASED_SOURCE_NAME as PAIR_OPPORTUNITY_SOURCE_NAME,
)
from apache_mixture import (
    RELEASED_MIX_FRACTIONS,
    RELEASED_SAMPLES_PER_EPOCH,
    RELEASED_SEED,
    RELEASED_SOURCE_NAMES,
)
from apache_released_contract import (
    RELEASED_MODEL_SCALARS,
    RELEASED_OBJECTIVE_SCALARS,
    RELEASED_RECIPE_SCALARS,
    RELEASED_STRUCTURE,
    validate_released_scalar_contract,
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
)
from apache_text_student import (
    PredicateTextStudentConfig,
)
from apache_release_qualification import (
    CORPUS_SCHEMA,
    QUALIFICATION_SCHEMA,
    REFERENCE_SOURCE_COMMIT,
    qualify_training_run,
    unresolved_sources,
    validate_corpus,
    validate_training_run,
)


def h(char: str) -> str:
    return char * 64


def names_h(names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def tokenizer_bundle_h(
    files: dict[str, str],
) -> str:
    digest = hashlib.sha256()
    for filename in sorted(files):
        digest.update(
            filename.encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            files[filename].encode(
                "ascii"
            )
        )
        digest.update(b"\0")
    return digest.hexdigest()


def corpus() -> dict:
    hashes = ("1", "2", "3")
    counts = (463_657, 40_000, 97)
    sources = []
    for index, name in enumerate(
        RELEASED_SOURCE_NAMES
    ):
        sources.append(
            {
                "name": name,
                "provenance_kind": (
                    "published-pack"
                    if index == 0
                    else "deterministic-rebuild"
                ),
                "origin": (
                    "hf://datasets/maelic/RA-4M"
                    if index == 0
                    else "Maelic/RelateAnything"
                ),
                "revision": (
                    "ee58e2f0cd6eab60044c90a5c5309ec51584f641"
                    if index == 0
                    else REFERENCE_SOURCE_COMMIT
                ),
                "recipe": (
                    None
                    if index == 0
                    else (
                        "training/convert_vg_raw.py"
                        if index == 1
                        else "training/convert_hicodet.py"
                    )
                ),
                "annotations_sha256": h(hashes[index]),
                "post_exclusion_count": counts[index],
            }
        )
        if sources[-1]["recipe"] is None:
            del sources[-1]["recipe"]

    union_predicates = (
        "above",
        "over",
        "below",
    )
    source_column_sources = [
        {
            "name": "megasg_clean",
            "predicates": list(
                union_predicates
            ),
        },
        {
            "name": "vg_raw",
            "predicates": list(
                union_predicates
            ),
        },
        {
            "name": "hicodet",
            "predicates": [
                "above",
            ],
        },
    ]
    source_column_payload = {
        "schema": SOURCE_ALLOW_SCHEMA,
        "sources": source_column_sources,
    }
    source_column_sha = hashlib.sha256(
        source_column_json_bytes(
            source_column_payload
        )
    ).hexdigest()
    local_predicates = {
        "megasg_clean": [
            "above",
            "over",
            "below",
        ],
        "vg_raw": [
            "below",
            "above",
            "over",
        ],
        "hicodet": [
            "above",
            "unknown-local",
        ],
    }
    source_column_derivation = {
        "schema": SOURCE_COLUMN_DERIVATION_SCHEMA,
        "source_order": list(
            RELEASED_SOURCE_NAMES
        ),
        "restricted_sources": list(
            RELEASED_RESTRICTED_SOURCES
        ),
        "union_predicate_order": list(
            union_predicates
        ),
        "union_predicate_count": len(
            union_predicates
        ),
        "union_predicate_order_sha256": names_h(
            union_predicates
        ),
        "sources": [
            {
                "source_name": name,
                "pack_split": f"/packs/{name}/train",
                "meta_sha256": h(
                    str(index + 4)
                ),
                "local_predicates": (
                    local_predicates[name]
                ),
                "local_predicate_count": len(
                    local_predicates[
                        name
                    ]
                ),
                "local_predicate_order_sha256": names_h(
                    tuple(
                        local_predicates[
                            name
                        ]
                    )
                ),
                "unknown_local_predicates": (
                    [
                        value
                        for value in local_predicates[
                            name
                        ]
                        if value
                        not in union_predicates
                    ]
                ),
                "restricted": (
                    name
                    in RELEASED_RESTRICTED_SOURCES
                ),
                "allowed_predicates": (
                    source_column_sources[
                        index
                    ]["predicates"]
                ),
                "allowed_predicate_count": len(
                    source_column_sources[
                        index
                    ]["predicates"]
                ),
                "allowed_predicate_order_sha256": names_h(
                    tuple(
                        source_column_sources[
                            index
                        ]["predicates"]
                    )
                ),
            }
            for index, name in enumerate(
                RELEASED_SOURCE_NAMES
            )
        ],
        "sidecar_sha256": (
            source_column_sha
        ),
    }

    indoorvg_vg_ids = [
        "100",
        "200",
    ]
    indoorvg_coco_stems = [
        "000000000001",
    ]
    indoorvg_stems = sorted(
        set(indoorvg_vg_ids)
        | set(indoorvg_coco_stems)
    )
    indoorvg_mapped_pairs = [
        {
            "vg_id": "100",
            "coco_stem": "000000000001",
        },
    ]
    indoorvg_payload = {
        "source": INDOORVG_RELEASED_SOURCE,
        "splits": list(
            INDOORVG_RELEASED_SPLITS
        ),
        "note": INDOORVG_RELEASED_NOTE,
        "vg_ids": indoorvg_vg_ids,
        "coco_stems": indoorvg_coco_stems,
        "stems": indoorvg_stems,
    }
    exclude_ids_sha = hashlib.sha256(
        indoorvg_holdout_bytes(
            indoorvg_payload
        )
    ).hexdigest()
    exclude_ids_derivation = {
        "schema": INDOORVG_DERIVATION_SCHEMA,
        "source": INDOORVG_RELEASED_SOURCE,
        "splits": list(
            INDOORVG_RELEASED_SPLITS
        ),
        "split_inputs": [
            {
                "split": "val",
                "vg_ids": ["100"],
                "vg_id_count": 1,
                "vg_ids_sha256": names_h(
                    ("100",)
                ),
            },
            {
                "split": "test",
                "vg_ids": ["200"],
                "vg_id_count": 1,
                "vg_ids_sha256": names_h(
                    ("200",)
                ),
            },
        ],
        "vg_ids": indoorvg_vg_ids,
        "vg_id_count": 2,
        "vg_ids_sha256": names_h(
            tuple(
                indoorvg_vg_ids
            )
        ),
        "vg2coco_sha256": h("d"),
        "mapped_pairs": indoorvg_mapped_pairs,
        "mapped_vg_count": 1,
        "mapped_pairs_sha256": (
            indoorvg_stable_json_sha256(
                indoorvg_mapped_pairs
            )
        ),
        "unmapped_vg_ids": [
            "200",
        ],
        "unmapped_vg_count": 1,
        "coco_stems": indoorvg_coco_stems,
        "coco_stem_count": 1,
        "coco_stems_sha256": names_h(
            tuple(
                indoorvg_coco_stems
            )
        ),
        "stems": indoorvg_stems,
        "stem_count": len(
            indoorvg_stems
        ),
        "stems_sha256": names_h(
            tuple(
                indoorvg_stems
            )
        ),
        "output_sha256": (
            exclude_ids_sha
        ),
    }

    return {
        "schema": CORPUS_SCHEMA,
        "reference_source_commit": (
            REFERENCE_SOURCE_COMMIT
        ),
        "sources": sources,
        "exclude_ids_sha256": exclude_ids_sha,
        "exclude_ids_derivation": (
            exclude_ids_derivation
        ),
        "source_column_allow_sha256": (
            source_column_sha
        ),
        "source_column_derivation": (
            source_column_derivation
        ),
        "ontology_meta_sha256": h("6"),
        "ontology_npz_sha256": h("7"),
        "neg_rate_table_sha256": h("8"),
        "predicate_embeddings_sha256": h("9"),
        "object_embeddings_sha256": h("e"),
        "predicate_spatial_flags_sha256": h("0"),
        "predicate_spatial_flags_derivation": {
            "schema": DERIVATION_SCHEMA,
            "algorithm": DERIVATION_ALGORITHM,
            "spatial_bit": SPATIAL_BIT,
            "majority_threshold": MAJORITY_THRESHOLD,
            "majority_comparator": MAJORITY_COMPARATOR,
            "sources": [
                {
                    "source_name": name,
                    "pack_split": f"/packs/{name}/train",
                    "meta_sha256": h(str(index + 4)),
                    "rels_sha256": h(str(index + 7)),
                    "relations": 100 + index,
                    "local_predicate_count": 3,
                    "supported_predicate_count": 3,
                    "local_spatial_majority_count": 1,
                    "ignored_predicates": [],
                }
                for index, name in enumerate(
                    RELEASED_SOURCE_NAMES
                )
            ],
            "union_predicate_count": 3,
            "supported_union_predicate_count": 3,
            "spatial_union_predicate_count": 1,
            "unsupported_predicates": [],
            "sidecar_sha256": h("0"),
        },
        "text_bank_derivation": {
            "schema": TEXT_BANK_DERIVATION_SCHEMA,
            "student_checkpoint_sha256": (
                RELEASED_TEXT_STUDENT_SHA256
            ),
            "student_config": asdict(
                PredicateTextStudentConfig()
            ),
            "tokenizer": {
                "schema": TOKENIZER_BUNDLE_SCHEMA,
                "tokenizer_id": RELEASED_TOKENIZER_ID,
                "source_kind": "local-bundle",
                "path": "/release/tokenizer",
                "files": {
                    "tokenizer.json": h("a"),
                    "tokenizer_config.json": h("b"),
                },
                "bundle_sha256": tokenizer_bundle_h(
                    {
                        "tokenizer.json": h("a"),
                        "tokenizer_config.json": h("b"),
                    }
                ),
            },
            "transformers_version": (
                RELEASED_TRANSFORMERS_VERSION
            ),
            "encoding_semantics": (
                "encode-each-template,sum-template-vectors,l2-normalize-once"
            ),
            "predicate_bank": {
                "templates": list(
                    RELEASED_PREDICATE_TEMPLATES
                ),
                "label_order_sha256": names_h(
                    (
                        "above",
                        "over",
                        "below",
                    )
                ),
                "label_count": 3,
                "shape": [3, 512],
                "tensor_sha256": h("c"),
                "artifact_sha256": h("9"),
                "npz_format": TEXT_BANK_NPZ_FORMAT,
            },
            "object_bank": {
                "templates": list(
                    RELEASED_OBJECT_TEMPLATES
                ),
                "label_order_sha256": names_h(
                    (
                        "person",
                        "horse",
                    )
                ),
                "label_count": 2,
                "shape": [2, 512],
                "tensor_sha256": h("d"),
                "artifact_sha256": h("e"),
                "npz_format": TEXT_BANK_NPZ_FORMAT,
            },
        },
        "pair_opportunity_rebuild": {
            "schema": PAIR_OPPORTUNITY_REBUILD_SCHEMA,
            "source_name": PAIR_OPPORTUNITY_SOURCE_NAME,
            "pack_split": "/packs/megasg_clean/train",
            "component_sha256": {
                "meta.json": h("4"),
                "img_meta.npy": h("b"),
                "box_cats.npy": h("c"),
                "rels.npy": h("7"),
            },
            "object_label_order": [
                "person",
                "horse",
            ],
            "object_label_order_sha256": names_h(
                (
                    "person",
                    "horse",
                )
            ),
            "num_cats": 2,
            "scan_box_cap": PAIR_OPPORTUNITY_SCAN_BOX_CAP,
            "min_support": PAIR_OPPORTUNITY_MIN_SUPPORT,
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
            "images_total": 100,
            "images_scanned": 100,
            "boxes_scanned": 200,
            "relations_scanned": 50,
            "relations_dropped_by_box_cap": 0,
            "category_pairs_with_opportunity": 4,
            "trusted_category_pairs": 2,
            "opportunity_sum": 1000,
            "relation_sum": 50,
            "npz_format": PAIR_OPPORTUNITY_NPZ_FORMAT,
            "output_sha256": h("8"),
        },
        "vocabulary_sha256": h("a"),
    }


def training() -> dict:
    c = corpus()
    recipe_scalars = dict(
        RELEASED_RECIPE_SCALARS
    )
    model_scalars = dict(
        RELEASED_MODEL_SCALARS
    )
    objective_scalars = dict(
        RELEASED_OBJECTIVE_SCALARS
    )
    scalar_contract = (
        validate_released_scalar_contract(
            recipe=recipe_scalars,
            model=model_scalars,
            objective=objective_scalars,
        )
    )
    structure_contract = {
        "schema": "kfcore.apache-released-structure/1",
        "matches_released": True,
        **RELEASED_STRUCTURE,
        "box_pe_num_freqs": 16,
        "box_pe_max_octave": 7.0,
    }
    return {
        "schema": "kfcore.relation-training-run/1",
        "backbone": "hf_hub:timm/vit_small_patch16_dinov3.lvd1689m",
        "frozen_backbone": False,
        "device": "cuda:0",
        "training_recipe": {
            "name": "apache-reference",
            "released_epoch_target": 12,
            "matches_released_epoch_count": True,
            "config": {
                "epochs": 12,
                "micro_batch_size": 32,
                "grad_accum": 4,
                "ema_decay": 0.9998,
                "augment": 0.3,
                "text_dim": 512,
                "image_size": 448,
                "geo_budget": 400,
                "final_budget": 128,
                "amp": True,
                "amp_dtype": "bf16",
                **recipe_scalars,
            },
            "effective_batch_size": 128,
            "weight_source": "ema",
            "ema": {
                "enabled": True,
                "decay": 0.9998,
                "updates": 47_220,
                "effective_decay": 0.9998,
                "state_sha256": h("c"),
                "raw_state_sha256": h("d"),
                "weights_source": "ema",
            },
            "augmentation": {
                "kind": "brightness-contrast-saturation",
                "strength": 0.3,
                "horizontal_flip": False,
                "geometry_transform": False,
                "rng_source": "ambient-torch-rng",
                "rng_equivalence": "stochastic-distribution",
                "worker_trajectory_equivalence": False,
            },
            "precision": {
                "amp_requested": True,
                "amp_dtype": "bf16",
                "autocast_device_type": "cuda",
                "autocast_executed": True,
                "grad_scaler": False,
                "released_cuda_execution": True,
            },
            "scalar_contract": scalar_contract,
            "structure_contract": structure_contract,
            "sampler_budget": {
                "geo_budget": 400,
                "final_budget": 128,
                "matches_released": True,
            },
        },
        "train_mixture": {
            "source_names": list(
                RELEASED_SOURCE_NAMES
            ),
            "target_fractions": list(
                RELEASED_MIX_FRACTIONS
            ),
            "seed": RELEASED_SEED,
            "draws_per_epoch": (
                RELEASED_SAMPLES_PER_EPOCH
            ),
            "matches_released_sampling_stream": True,
            "source_annotation_sha256": [
                source["annotations_sha256"]
                for source in c["sources"]
            ],
            "source_counts": [
                source["post_exclusion_count"]
                for source in c["sources"]
            ],
            "exclude_ids_sha256": c[
                "exclude_ids_sha256"
            ],
        },
        "history": [
            {"epoch": epoch, "loss": 1.0}
            for epoch in range(1, 13)
        ],
        "routing_warm_start": {
            "schema": "kfcore.apache-routing-warm-start/1",
            "spatial_flags_sha256": c[
                "predicate_spatial_flags_sha256"
            ],
            "spatial_count": 1,
            "semantic_count": 2,
            "target_alpha_mean": 0.4,
            "target_alpha_spatial_mean": 0.8,
            "target_alpha_semantic_mean": 0.2,
            "mse_before": 0.20,
            "mse_after": 0.05,
            "reported_final_mse": 0.06,
        },
        "model_config": {
            "image_size": 448,
            "max_boxes": 40,
            "pair_budget": 128,
            "hidden_dim": 512,
            "pair_evidence_contract": "apache",
            "pair_sampler_contract": "apache",
            "relation_context_contract": "apache",
            "predicate_head_contract": "apache",
            **model_scalars,
        },
        "checkpoint_sha256": h("b"),
        "predicate_embedding_shape": [3, 512],
        "predicate_embeddings_sha256": c[
            "predicate_embeddings_sha256"
        ],
        "predicate_bank": {
            "schema": PREDICATE_BANK_SCHEMA,
            "artifact_sha256": c[
                "predicate_embeddings_sha256"
            ],
            "shape": [3, 512],
            "source_dtype": "float16",
            "runtime_dtype": "torch.float32",
            "predicate_count": 3,
            "predicate_order": [
                "above",
                "over",
                "below",
            ],
            "predicate_order_sha256": names_h(
                (
                    "above",
                    "over",
                    "below",
                )
            ),
            "templates": list(
                RELEASED_PREDICATE_TEMPLATES
            ),
            "templates_sha256": names_h(
                RELEASED_PREDICATE_TEMPLATES
            ),
            "text_dim": 512,
        },
        "vocabulary_sha256": c[
            "vocabulary_sha256"
        ],
        "apache_reference_objective": {
            "config": objective_scalars,
            "assets": {
                "source_column_allow_sha256": c[
                    "source_column_allow_sha256"
                ],
                "ontology_meta_sha256": c[
                    "ontology_meta_sha256"
                ],
                "ontology_npz_sha256": c[
                    "ontology_npz_sha256"
                ],
                "neg_rate_table_sha256": c[
                    "neg_rate_table_sha256"
                ],
                "object_embeddings_sha256": c[
                    "object_embeddings_sha256"
                ],
                "object_embeddings_shape": [
                    2,
                    512,
                ],
                "object_label_order": [
                    "person",
                    "horse",
                ],
                "object_bank": {
                    "schema": "kfcore.apache-object-text-bank/1",
                    "artifact_sha256": c[
                        "object_embeddings_sha256"
                    ],
                    "shape": [2, 512],
                    "source_dtype": "float16",
                    "runtime_dtype": "torch.float32",
                    "object_label_count": 2,
                    "object_label_order": [
                        "person",
                        "horse",
                    ],
                    "object_label_order_sha256": names_h(
                        (
                            "person",
                            "horse",
                        )
                    ),
                    "text_dim": 512,
                },
            }
        },
    }


class ApacheReleaseQualificationTest(
    unittest.TestCase
):
    def test_resolved_corpus_and_training_qualify(self):
        result = qualify_training_run(
            corpus(),
            training(),
        )
        self.assertEqual(
            result["schema"],
            QUALIFICATION_SCHEMA,
        )
        self.assertTrue(result["qualified"])
        self.assertEqual(
            result["epochs"],
            12,
        )
        self.assertEqual(
            result["draws_per_epoch"],
            RELEASED_SAMPLES_PER_EPOCH,
        )
        self.assertEqual(
            result["effective_batch_size"],
            128,
        )
        self.assertEqual(
            result["weight_source"],
            "ema",
        )
        self.assertEqual(
            result["ema_decay"],
            0.9998,
        )
        self.assertEqual(
            result["ema_state_sha256"],
            h("c"),
        )
        self.assertEqual(
            result["augmentation"]["strength"],
            0.3,
        )
        self.assertFalse(
            result["augmentation"][
                "worker_trajectory_equivalence"
            ]
        )
        self.assertEqual(
            result["text_dim"],
            512,
        )
        self.assertEqual(
            result["object_embeddings_sha256"],
            h("e"),
        )
        self.assertEqual(
            result["object_embeddings_shape"],
            [2, 512],
        )
        self.assertEqual(
            result["pair_opportunity_rebuild"][
                "output_sha256"
            ],
            h("8"),
        )
        self.assertEqual(
            result["pair_opportunity_rebuild"][
                "min_support"
            ],
            50,
        )
        self.assertEqual(
            result["text_bank_derivation"][
                "student_checkpoint_sha256"
            ],
            RELEASED_TEXT_STUDENT_SHA256,
        )
        self.assertEqual(
            result["text_bank_derivation"][
                "predicate_bank"
            ]["templates"],
            list(
                RELEASED_PREDICATE_TEMPLATES
            ),
        )
        self.assertEqual(
            result["exclude_ids_derivation"][
                "output_sha256"
            ],
            corpus()["exclude_ids_sha256"],
        )
        self.assertEqual(
            result["exclude_ids_derivation"][
                "splits"
            ],
            ["val", "test"],
        )
        self.assertEqual(
            result["source_column_derivation"][
                "restricted_sources"
            ],
            ["hicodet"],
        )
        self.assertEqual(
            result["source_column_derivation"][
                "sources"
            ][2]["allowed_predicates"],
            ["above"],
        )
        self.assertEqual(
            result["predicate_spatial_flags_sha256"],
            h("0"),
        )
        self.assertEqual(
            result["predicate_spatial_flags_derivation"][
                "majority_comparator"
            ],
            ">=",
        )
        self.assertEqual(
            result["precision"]["amp_dtype"],
            "bf16",
        )
        self.assertTrue(
            result["precision"][
                "released_cuda_execution"
            ]
        )
        self.assertEqual(
            result["image_size"],
            448,
        )
        self.assertEqual(
            result["geo_budget"],
            400,
        )
        self.assertEqual(
            result["final_budget"],
            128,
        )

    def test_unresolved_source_is_representable_but_blocks_qualification(self):
        value = corpus()
        value["sources"][1] = {
            "name": "vg_raw",
            "provenance_kind": "unresolved",
            "note": "released public pack hash not available",
        }
        self.assertEqual(
            unresolved_sources(value),
            ("vg_raw",),
        )
        with self.assertRaisesRegex(
            ValueError,
            "unresolved for: vg_raw",
        ):
            qualify_training_run(
                value,
                training(),
            )

    def test_rebuild_source_requires_recipe(self):
        value = corpus()
        del value["sources"][1]["recipe"]
        with self.assertRaisesRegex(
            ValueError,
            "rebuild recipe",
        ):
            validate_corpus(value)

    def test_training_source_hash_drift_is_rejected(self):
        run = training()
        run["train_mixture"][
            "source_annotation_sha256"
        ][2] = h("c")
        with self.assertRaisesRegex(
            ValueError,
            "annotation hashes",
        ):
            qualify_training_run(
                corpus(),
                run,
            )

    def test_training_source_count_drift_is_rejected(self):
        run = training()
        run["train_mixture"]["source_counts"][1] += 1
        with self.assertRaisesRegex(
            ValueError,
            "source counts",
        ):
            qualify_training_run(
                corpus(),
                run,
            )

    def test_released_hidden_dim_must_be_512(self):
        run = training()
        run["model_config"]["hidden_dim"] = 256
        with self.assertRaisesRegex(
            ValueError,
            "hidden_dim=512",
        ):
            validate_training_run(run)

    def test_released_max_boxes_must_be_40(self):
        run = training()
        run["model_config"]["max_boxes"] = 32
        with self.assertRaisesRegex(
            ValueError,
            "max_boxes=40",
        ):
            validate_training_run(run)

    def test_released_reference_requires_cuda_bf16_amp(self):
        run = training()
        run["training_recipe"]["config"][
            "amp"
        ] = False
        with self.assertRaisesRegex(
            ValueError,
            "amp=true",
        ):
            validate_training_run(run)

        run = training()
        run["training_recipe"]["config"][
            "amp_dtype"
        ] = "fp16"
        with self.assertRaisesRegex(
            ValueError,
            "amp_dtype=bf16",
        ):
            validate_training_run(run)

        run = training()
        run["training_recipe"]["precision"][
            "autocast_executed"
        ] = False
        with self.assertRaisesRegex(
            ValueError,
            "executed autocast",
        ):
            validate_training_run(run)

        run = training()
        run["training_recipe"]["precision"][
            "grad_scaler"
        ] = True
        with self.assertRaisesRegex(
            ValueError,
            "must not use GradScaler",
        ):
            validate_training_run(run)

        run = training()
        run["device"] = "cpu"
        run["training_recipe"]["precision"][
            "autocast_device_type"
        ] = "cpu"
        run["training_recipe"]["precision"][
            "released_cuda_execution"
        ] = False
        with self.assertRaisesRegex(
            ValueError,
            "CUDA autocast|CUDA BF16|CUDA training device",
        ):
            validate_training_run(run)

    def test_released_reference_requires_448_400_128_shape(self):
        run = training()
        run["training_recipe"]["config"][
            "image_size"
        ] = 224
        with self.assertRaisesRegex(
            ValueError,
            "img_size=448",
        ):
            validate_training_run(run)

        run = training()
        run["training_recipe"]["config"][
            "geo_budget"
        ] = 399
        with self.assertRaisesRegex(
            ValueError,
            "geo_budget=400",
        ):
            validate_training_run(run)

        run = training()
        run["training_recipe"]["config"][
            "final_budget"
        ] = 64
        with self.assertRaisesRegex(
            ValueError,
            "final_budget=128",
        ):
            validate_training_run(run)

        run = training()
        run["model_config"]["image_size"] = 224
        with self.assertRaisesRegex(
            ValueError,
            "image_size=448",
        ):
            validate_training_run(run)

        run = training()
        run["model_config"]["pair_budget"] = 64
        with self.assertRaisesRegex(
            ValueError,
            "pair_budget=128",
        ):
            validate_training_run(run)

        run = training()
        run["training_recipe"]["sampler_budget"][
            "geo_budget"
        ] = 64
        with self.assertRaisesRegex(
            ValueError,
            "400->128",
        ):
            validate_training_run(run)

    def test_released_reference_requires_text_dim_512(self):
        run = training()
        run["training_recipe"]["config"][
            "text_dim"
        ] = 32
        with self.assertRaisesRegex(
            ValueError,
            "text_dim=512",
        ):
            validate_training_run(run)

        run = training()
        run["predicate_embedding_shape"] = [3, 32]
        with self.assertRaisesRegex(
            ValueError,
            "text_dim=512",
        ):
            validate_training_run(run)

    def test_released_reference_requires_photometric_augment(self):
        run = training()
        run["training_recipe"]["config"][
            "augment"
        ] = 0.0
        with self.assertRaisesRegex(
            ValueError,
            "augment=0.3",
        ):
            validate_training_run(run)

        run = training()
        run["training_recipe"][
            "augmentation"
        ]["strength"] = 0.2
        with self.assertRaisesRegex(
            ValueError,
            "augment=0.3",
        ):
            validate_training_run(run)

        run = training()
        run["training_recipe"][
            "augmentation"
        ]["horizontal_flip"] = True
        with self.assertRaisesRegex(
            ValueError,
            "preserve geometry",
        ):
            validate_training_run(run)

        run = training()
        run["training_recipe"][
            "augmentation"
        ]["rng_source"] = "custom"
        with self.assertRaisesRegex(
            ValueError,
            "RNG source",
        ):
            validate_training_run(run)

    def test_released_reference_requires_ema_weights(self):
        run = training()
        run["training_recipe"][
            "weight_source"
        ] = "raw"
        with self.assertRaisesRegex(
            ValueError,
            "EMA checkpoint weights",
        ):
            validate_training_run(run)

        run = training()
        run["training_recipe"]["ema"][
            "decay"
        ] = 0.99
        with self.assertRaisesRegex(
            ValueError,
            "ema_decay=0.9998",
        ):
            validate_training_run(run)

        run = training()
        run["training_recipe"]["ema"][
            "state_sha256"
        ] = run["training_recipe"]["ema"][
            "raw_state_sha256"
        ]
        with self.assertRaisesRegex(
            ValueError,
            "must differ",
        ):
            validate_training_run(run)

    def test_sampling_stream_must_be_qualified(self):
        run = training()
        run["train_mixture"][
            "matches_released_sampling_stream"
        ] = False
        with self.assertRaisesRegex(
            ValueError,
            "sampling stream",
        ):
            validate_training_run(run)

    def test_history_must_contain_exact_12_epochs(self):
        run = training()
        run["history"] = run["history"][:-1]
        with self.assertRaisesRegex(
            ValueError,
            "all 12",
        ):
            validate_training_run(run)

        run = training()
        run["history"][5]["epoch"] = 99
        with self.assertRaisesRegex(
            ValueError,
            "epoch sequence",
        ):
            validate_training_run(run)

    def test_source_column_derivation_is_bound(self):
        value = corpus()
        value["source_column_derivation"][
            "restricted_sources"
        ] = ["vg_raw"]
        with self.assertRaisesRegex(
            ValueError,
            "restrict hicodet only",
        ):
            validate_corpus(value)

        value = corpus()
        value["source_column_derivation"][
            "sources"
        ][2]["allowed_predicates"] = [
            "below",
        ]
        with self.assertRaisesRegex(
            ValueError,
            "allowed predicates",
        ):
            validate_corpus(value)

        value = corpus()
        value["source_column_derivation"][
            "sources"
        ][2]["meta_sha256"] = h("1")
        with self.assertRaisesRegex(
            ValueError,
            "same pack meta",
        ):
            validate_corpus(value)

        value = corpus()
        value["source_column_derivation"][
            "sidecar_sha256"
        ] = h("1")
        with self.assertRaisesRegex(
            ValueError,
            "sidecar hash",
        ):
            validate_corpus(value)

        value = corpus()
        value["source_column_derivation"][
            "union_predicate_order"
        ] = [
            "over",
            "above",
            "below",
        ]
        value["source_column_derivation"][
            "union_predicate_order_sha256"
        ] = names_h(
            (
                "over",
                "above",
                "below",
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "named predicate bank",
        ):
            validate_corpus(value)

    def test_text_bank_derivation_is_bound(self):
        value = corpus()
        value["text_bank_derivation"][
            "student_checkpoint_sha256"
        ] = h("1")
        with self.assertRaisesRegex(
            ValueError,
            "released text-student",
        ):
            validate_corpus(value)

        value = corpus()
        value["text_bank_derivation"][
            "predicate_bank"
        ]["templates"] = [
            "{p}",
        ]
        with self.assertRaisesRegex(
            ValueError,
            "predicate text-bank templates",
        ):
            validate_corpus(value)

        value = corpus()
        value["text_bank_derivation"][
            "tokenizer"
        ]["bundle_sha256"] = h("1")
        with self.assertRaisesRegex(
            ValueError,
            "tokenizer bundle hash",
        ):
            validate_corpus(value)

        run = training()
        run["predicate_bank"][
            "predicate_order"
        ] = [
            "over",
            "above",
            "below",
        ]
        run["predicate_bank"][
            "predicate_order_sha256"
        ] = names_h(
            (
                "over",
                "above",
                "below",
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "predicate-bank order",
        ):
            qualify_training_run(
                corpus(),
                run,
            )

        value = corpus()
        value["text_bank_derivation"][
            "object_bank"
        ]["label_order_sha256"] = names_h(
            (
                "horse",
                "person",
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "object-bank order",
        ):
            qualify_training_run(
                value,
                training(),
            )

    def test_indoorvg_exclusion_derivation_is_required_and_bound(self):
        value = corpus()
        del value[
            "exclude_ids_derivation"
        ]
        with self.assertRaisesRegex(
            ValueError,
            "IndoorVG exclusion derivation",
        ):
            validate_corpus(value)

        value = corpus()
        value[
            "exclude_ids_derivation"
        ]["splits"] = [
            "test",
            "val",
        ]
        with self.assertRaisesRegex(
            ValueError,
            r"splits=\['val','test'\]",
        ):
            validate_corpus(value)

        value = corpus()
        value[
            "exclude_ids_derivation"
        ]["mapped_pairs"][0][
            "coco_stem"
        ] = "1"
        with self.assertRaisesRegex(
            ValueError,
            "exactly 12 digits",
        ):
            validate_corpus(value)

        value = corpus()
        split = value[
            "exclude_ids_derivation"
        ]["split_inputs"][0]
        split["vg_ids"] = [
            "100",
            "999",
        ]
        split["vg_id_count"] = 2
        split["vg_ids_sha256"] = names_h(
            (
                "100",
                "999",
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "val/test union",
        ):
            validate_corpus(value)

        value = corpus()
        value[
            "exclude_ids_derivation"
        ]["stems"] = (
            value[
                "exclude_ids_derivation"
            ]["stems"]
            + ["extra"]
        )
        with self.assertRaisesRegex(
            ValueError,
            "VG/COCO union",
        ):
            validate_corpus(value)

        value = corpus()
        value[
            "exclude_ids_derivation"
        ]["output_sha256"] = h("1")
        with self.assertRaisesRegex(
            ValueError,
            "output hash",
        ):
            validate_corpus(value)

        run = training()
        run["train_mixture"][
            "exclude_ids_sha256"
        ] = h("1")
        with self.assertRaisesRegex(
            ValueError,
            "exclusion artifact",
        ):
            qualify_training_run(
                corpus(),
                run,
            )

    def test_pair_opportunity_rebuild_is_bound(self):
        value = corpus()
        del value["pair_opportunity_rebuild"]
        with self.assertRaisesRegex(
            ValueError,
            "pair-opportunity rebuild",
        ):
            validate_corpus(value)

        value = corpus()
        value[
            "pair_opportunity_rebuild"
        ]["output_sha256"] = h("1")
        with self.assertRaisesRegex(
            ValueError,
            "output hash",
        ):
            validate_corpus(value)

        value = corpus()
        value[
            "pair_opportunity_rebuild"
        ]["min_support"] = 49
        with self.assertRaisesRegex(
            ValueError,
            "min_support=50",
        ):
            validate_corpus(value)

        value = corpus()
        value[
            "pair_opportunity_rebuild"
        ]["component_sha256"][
            "rels.npy"
        ] = h("1")
        with self.assertRaisesRegex(
            ValueError,
            "same MegaSG pack",
        ):
            validate_corpus(value)

        value = corpus()
        value[
            "pair_opportunity_rebuild"
        ]["full_scan"] = False
        with self.assertRaisesRegex(
            ValueError,
            "full scan",
        ):
            validate_corpus(value)

        value = corpus()
        value[
            "pair_opportunity_rebuild"
        ]["object_label_order"] = [
            "horse",
            "person",
        ]
        value[
            "pair_opportunity_rebuild"
        ]["object_label_order_sha256"] = names_h(
            (
                "horse",
                "person",
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "category order",
        ):
            qualify_training_run(
                value,
                training(),
            )

        value = corpus()
        value[
            "pair_opportunity_rebuild"
        ]["relation_sum"] = 49
        with self.assertRaisesRegex(
            ValueError,
            "relation summary",
        ):
            validate_corpus(value)

    def test_spatial_flag_derivation_and_routing_are_bound(self):
        value = corpus()
        value[
            "predicate_spatial_flags_derivation"
        ]["majority_comparator"] = ">"
        with self.assertRaisesRegex(
            ValueError,
            "comparator",
        ):
            validate_corpus(value)

        value = corpus()
        value[
            "predicate_spatial_flags_derivation"
        ]["sources"][0]["source_name"] = "vg_raw"
        with self.assertRaisesRegex(
            ValueError,
            "source order",
        ):
            validate_corpus(value)

        value = corpus()
        value[
            "predicate_spatial_flags_derivation"
        ]["sidecar_sha256"] = h("1")
        with self.assertRaisesRegex(
            ValueError,
            "sidecar hash",
        ):
            validate_corpus(value)

        run = training()
        run["routing_warm_start"][
            "spatial_flags_sha256"
        ] = h("1")
        with self.assertRaisesRegex(
            ValueError,
            "spatial-flags hash",
        ):
            qualify_training_run(
                corpus(),
                run,
            )

        run = training()
        run["routing_warm_start"][
            "mse_after"
        ] = 0.25
        with self.assertRaisesRegex(
            ValueError,
            "improve MSE",
        ):
            validate_training_run(run)

        run = training()
        run["routing_warm_start"][
            "semantic_count"
        ] = 1
        with self.assertRaisesRegex(
            ValueError,
            "do not match predicate vocabulary",
        ):
            validate_training_run(run)

        value = corpus()
        value[
            "predicate_spatial_flags_derivation"
        ]["spatial_union_predicate_count"] = 2
        with self.assertRaisesRegex(
            ValueError,
            "spatial derivation count",
        ):
            qualify_training_run(
                value,
                training(),
            )

    def test_object_bank_provenance_is_required_and_bound(self):
        value = corpus()
        del value["object_embeddings_sha256"]
        with self.assertRaisesRegex(
            ValueError,
            "object_embeddings_sha256",
        ):
            validate_corpus(value)

        run = training()
        run["apache_reference_objective"][
            "assets"
        ]["object_embeddings_sha256"] = h("d")
        run["apache_reference_objective"][
            "assets"
        ]["object_bank"]["artifact_sha256"] = h("d")
        with self.assertRaisesRegex(
            ValueError,
            "object_embeddings_sha256",
        ):
            qualify_training_run(
                corpus(),
                run,
            )

        run = training()
        run["apache_reference_objective"][
            "assets"
        ]["object_embeddings_shape"] = [2, 256]
        with self.assertRaisesRegex(
            ValueError,
            r"\[O,512\]",
        ):
            validate_training_run(run)

        run = training()
        run["apache_reference_objective"][
            "assets"
        ]["object_label_order"] = [
            "horse",
            "person",
        ]
        with self.assertRaisesRegex(
            ValueError,
            "named object-bank provenance",
        ):
            validate_training_run(run)

    def test_objective_asset_drift_is_rejected(self):
        run = training()
        run["apache_reference_objective"][
            "assets"
        ]["neg_rate_table_sha256"] = h("d")
        with self.assertRaisesRegex(
            ValueError,
            "neg_rate_table_sha256",
        ):
            qualify_training_run(
                corpus(),
                run,
            )

    def test_wrong_reference_source_commit_is_rejected(self):
        value = corpus()
        value["reference_source_commit"] = "0" * 40
        with self.assertRaisesRegex(
            ValueError,
            "wrong Apache source reference",
        ):
            validate_corpus(value)


if __name__ == "__main__":
    unittest.main()
