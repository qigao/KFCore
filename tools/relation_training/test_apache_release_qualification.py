from __future__ import annotations

import copy
import unittest

from apache_mixture import (
    RELEASED_MIX_FRACTIONS,
    RELEASED_SAMPLES_PER_EPOCH,
    RELEASED_SEED,
    RELEASED_SOURCE_NAMES,
)
from apache_release_scalars import (
    MODEL_SCALARS,
    OBJECTIVE_SCALARS,
    RECIPE_SCALARS,
    validate_released_scalar_contract,
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

    return {
        "schema": CORPUS_SCHEMA,
        "reference_source_commit": (
            REFERENCE_SOURCE_COMMIT
        ),
        "sources": sources,
        "exclude_ids_sha256": h("4"),
        "source_column_allow_sha256": h("5"),
        "ontology_meta_sha256": h("6"),
        "ontology_npz_sha256": h("7"),
        "neg_rate_table_sha256": h("8"),
        "predicate_embeddings_sha256": h("9"),
        "vocabulary_sha256": h("a"),
    }


def training() -> dict:
    c = corpus()
    recipe_config = {
        "epochs": 12,
        "micro_batch_size": 32,
        "grad_accum": 4,
        "ema_decay": 0.9998,
        "augment": 0.3,
        "text_dim": 512,
        "image_size": 448,
        "geo_budget": 400,
        "final_budget": 128,
        **RECIPE_SCALARS,
    }
    model_config = {
        "image_size": 448,
        "max_boxes": 40,
        "pair_budget": 128,
        "hidden_dim": 512,
        "pair_evidence_contract": "apache",
        "pair_sampler_contract": "apache",
        "relation_context_contract": "apache",
        "predicate_head_contract": "apache",
        **MODEL_SCALARS,
    }
    objective_config = dict(
        OBJECTIVE_SCALARS
    )
    scalar_contract = (
        validate_released_scalar_contract(
            recipe_config,
            model_config,
            objective_config,
        )
    )
    return {
        "schema": "kfcore.relation-training-run/1",
        "backbone": "hf_hub:timm/vit_small_patch16_dinov3.lvd1689m",
        "frozen_backbone": False,
        "training_recipe": {
            "name": "apache-reference",
            "released_epoch_target": 12,
            "matches_released_epoch_count": True,
            "config": recipe_config,
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
            "sampler_budget": {
                "geo_budget": 400,
                "final_budget": 128,
                "matches_released": True,
            },
            "scalar_contract": scalar_contract,
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
        "model_config": model_config,
        "checkpoint_sha256": h("b"),
        "predicate_embedding_shape": [3, 512],
        "predicate_embeddings_sha256": c[
            "predicate_embeddings_sha256"
        ],
        "vocabulary_sha256": c[
            "vocabulary_sha256"
        ],
        "apache_reference_objective": {
            "config": objective_config,
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
            },
        },
    }


class ApacheReleaseQualificationTest(
    unittest.TestCase
):
    def test_raw_scalar_drift_and_report_tampering_fail_closed(self):
        run = training()
        run["training_recipe"]["config"][
            "head_lr"
        ] = 1.0e-3
        with self.assertRaisesRegex(
            ValueError,
            "head_lr",
        ):
            validate_training_run(run)

        run = training()
        run["training_recipe"][
            "scalar_contract"
        ] = {
            **run["training_recipe"][
                "scalar_contract"
            ],
            "matches_released": False,
        }
        with self.assertRaisesRegex(
            ValueError,
            "reported released scalar contract",
        ):
            validate_training_run(run)

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
