from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from apache_vg_raw import (
    REBUILD_SCHEMA,
    REFERENCE_SOURCE_COMMIT,
    RELEASED_MAX_WORDS,
    coco_bytes,
    convert_vg_raw,
    rebuild_vg_raw_source,
)
from benchmark import (
    DatasetManifest,
    RelationVocabulary,
    VOCAB_SCHEMA,
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def make_inputs(root: Path) -> dict[str, Path]:
    image_root = root / "VG150_coco_format" / "train"
    image_root.mkdir(parents=True)
    for name in (
        "1.jpg",
        "2.jpg",
        "3.JPG",
        "4.jpg",
        "5.jpg",
        "6.jpg",
        "7.jpg",
    ):
        (image_root / name).write_text(
            "identity-only",
            encoding="utf-8",
        )

    image_data = root / "image_data.json"
    write_json(
        image_data,
        [
            {"image_id": 1, "width": 100, "height": 80},
            {"image_id": 2, "width": 100, "height": 80},
            {"image_id": 3, "width": 100, "height": 80},
            {"image_id": 4, "width": 100, "height": 80},
            # image 5 intentionally absent -> no_dims.
            {"image_id": 6, "width": 100, "height": 80},
            {"image_id": 7, "width": 100, "height": 80},
        ],
    )

    long_predicate = "x" * 61
    relationships = root / "relationships.json"
    write_json(
        relationships,
        [
            {
                "image_id": 1,
                "relationships": [
                    {
                        "predicate": "  Riding   On  ",
                        "subject": {
                            "object_id": 10,
                            "name": "  Person ",
                            "x": 10,
                            "y": 10,
                            "w": 20,
                            "h": 20,
                        },
                        "object": {
                            "object_id": 20,
                            "names": [" Horse "],
                            "x": 50,
                            "y": 30,
                            "w": 20,
                            "h": 20,
                        },
                    },
                    # Same normalized triplet: writer-level dedup.
                    {
                        "predicate": "riding on",
                        "subject": {
                            "object_id": 10,
                            "name": "person",
                            "x": 10,
                            "y": 10,
                            "w": 20,
                            "h": 20,
                        },
                        "object": {
                            "object_id": 20,
                            "name": "horse",
                            "x": 50,
                            "y": 30,
                            "w": 20,
                            "h": 20,
                        },
                    },
                    # <=5 words but >60 chars: converter keeps, writer drops.
                    {
                        "predicate": long_predicate,
                        "subject": {
                            "object_id": 10,
                            "name": "person",
                            "x": 10,
                            "y": 10,
                            "w": 20,
                            "h": 20,
                        },
                        "object": {
                            "object_id": 30,
                            "name": "saddle",
                            "x": 45,
                            "y": 25,
                            "w": 10,
                            "h": 10,
                        },
                    },
                    {
                        "predicate": "one two three four five six",
                        "subject": {
                            "object_id": 10,
                            "name": "person",
                            "x": 10,
                            "y": 10,
                            "w": 20,
                            "h": 20,
                        },
                        "object": {
                            "object_id": 20,
                            "name": "horse",
                            "x": 50,
                            "y": 30,
                            "w": 20,
                            "h": 20,
                        },
                    },
                ],
            },
            # Protected by VG id.
            {
                "image_id": 2,
                "relationships": [],
            },
            # .JPG is not in the case-sensitive allowed stem set.
            {
                "image_id": 3,
                "relationships": [],
            },
            # Protected through VG->COCO->banned COCO.
            {
                "image_id": 4,
                "relationships": [],
            },
            # Allowed file but missing image_data dimensions.
            {
                "image_id": 5,
                "relationships": [],
            },
            # Object name exceeds five words -> no usable relation.
            {
                "image_id": 6,
                "relationships": [
                    {
                        "predicate": "beside",
                        "subject": {
                            "object_id": 60,
                            "name": "one two three four five six",
                            "x": 1,
                            "y": 1,
                            "w": 5,
                            "h": 5,
                        },
                        "object": {
                            "object_id": 61,
                            "name": "chair",
                            "x": 10,
                            "y": 10,
                            "w": 5,
                            "h": 5,
                        },
                    }
                ],
            },
            # Writer allocates categories then rolls annotations back.
            {
                "image_id": 7,
                "relationships": [
                    {
                        "predicate": long_predicate,
                        "subject": {
                            "object_id": 70,
                            "name": "lamp",
                            "x": 1,
                            "y": 1,
                            "w": 5,
                            "h": 5,
                        },
                        "object": {
                            "object_id": 71,
                            "name": "table",
                            "x": 10,
                            "y": 10,
                            "w": 5,
                            "h": 5,
                        },
                    }
                ],
            },
        ],
    )

    registry = root / "registry.json"
    write_json(
        registry,
        {
            "banned_coco_ids": [42],
            "protected_coco_ids": [999],
            "protected_vg_ids": ["2"],
        },
    )
    vg2coco = root / "vg2coco.json"
    write_json(
        vg2coco,
        {
            "4": 42,
            "1": 1001,
        },
    )
    psg2coco = root / "psg2coco.json"
    write_json(
        psg2coco,
        {
            "psg-1": 10,
        },
    )
    return {
        "image_root": image_root,
        "image_data": image_data,
        "relationships": relationships,
        "registry": registry,
        "vg2coco": vg2coco,
        "psg2coco": psg2coco,
    }


class ApacheVGRawTest(unittest.TestCase):
    def test_converter_matches_filtering_writer_and_vocab_order(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = make_inputs(Path(directory))
            payload, evidence = convert_vg_raw(
                relationships_path=paths["relationships"],
                image_data_path=paths["image_data"],
                image_root=paths["image_root"],
                registry_path=paths["registry"],
                vg2coco_path=paths["vg2coco"],
                psg2coco_path=paths["psg2coco"],
            )

        self.assertEqual(
            evidence["schema"],
            REBUILD_SCHEMA,
        )
        self.assertEqual(
            evidence["reference_source_commit"],
            REFERENCE_SOURCE_COMMIT,
        )
        self.assertEqual(
            evidence["max_words"],
            RELEASED_MAX_WORDS,
        )
        self.assertEqual(
            evidence["allowed_image_stems"],
            ["1", "2", "4", "5", "6", "7"],
        )
        self.assertEqual(
            evidence["protected_vg_ids"],
            ["2"],
        )
        self.assertEqual(
            evidence["protected_coco_ids"],
            [42],
        )
        self.assertEqual(
            evidence["drops"],
            {
                "pred_too_long": 1,
                "eval_protected": 2,
                "image_not_on_disk": 1,
                "no_dims": 1,
                "obj_name_too_long": 1,
                "image_no_usable_rels": 1,
            },
        )
        self.assertEqual(
            evidence["writer_drops"],
            {
                "predicate": 2,
                "self_loop": 0,
                "duplicate_triplet": 1,
                "rolled_back_images": 1,
            },
        )
        self.assertEqual(
            evidence["leaked_image_count"],
            0,
        )

        self.assertEqual(
            payload["images"],
            [
                {
                    "id": 1,
                    "file_name": "1.jpg",
                    "width": 100,
                    "height": 80,
                }
            ],
        )
        self.assertEqual(
            payload["categories"],
            [
                {"id": 0, "name": "person"},
                {"id": 1, "name": "horse"},
                {"id": 2, "name": "saddle"},
                {"id": 3, "name": "lamp"},
                {"id": 4, "name": "table"},
            ],
        )
        self.assertEqual(
            payload["rel_categories"],
            [
                {"id": 0, "name": "riding on"},
            ],
        )
        self.assertEqual(
            len(payload["annotations"]),
            3,
        )
        self.assertEqual(
            payload["annotations"][0]["bbox"],
            [10.0, 10.0, 20.0, 20.0],
        )
        self.assertEqual(
            payload["annotations"][1]["bbox"],
            [50.0, 30.0, 20.0, 20.0],
        )
        self.assertEqual(
            payload["annotations"][2]["bbox"],
            [45.0, 25.0, 10.0, 10.0],
        )
        self.assertEqual(
            payload["rel_annotations"],
            [
                {
                    "image_id": 1,
                    "subject_id": 1,
                    "object_id": 2,
                    "predicate_id": 0,
                }
            ],
        )

    def test_registry_banned_falls_back_to_protected_when_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = make_inputs(root)
            write_json(
                paths["registry"],
                {
                    "banned_coco_ids": [],
                    "protected_coco_ids": [1001],
                    "protected_vg_ids": [],
                },
            )
            payload, evidence = convert_vg_raw(
                relationships_path=paths["relationships"],
                image_data_path=paths["image_data"],
                image_root=paths["image_root"],
                registry_path=paths["registry"],
                vg2coco_path=paths["vg2coco"],
                psg2coco_path=paths["psg2coco"],
            )

        self.assertEqual(
            evidence["protected_coco_ids"],
            [1001],
        )
        self.assertNotIn(
            1,
            [image["id"] for image in payload["images"]],
        )

    def test_max_words_is_released_constant(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = make_inputs(Path(directory))
            with self.assertRaisesRegex(
                ValueError,
                "max_words=5",
            ):
                convert_vg_raw(
                    relationships_path=paths["relationships"],
                    image_data_path=paths["image_data"],
                    image_root=paths["image_root"],
                    registry_path=paths["registry"],
                    vg2coco_path=paths["vg2coco"],
                    psg2coco_path=paths["psg2coco"],
                    max_words=4,
                )

    def test_conversion_bytes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = make_inputs(Path(directory))
            one, report_one = convert_vg_raw(
                relationships_path=paths["relationships"],
                image_data_path=paths["image_data"],
                image_root=paths["image_root"],
                registry_path=paths["registry"],
                vg2coco_path=paths["vg2coco"],
                psg2coco_path=paths["psg2coco"],
            )
            two, report_two = convert_vg_raw(
                relationships_path=paths["relationships"],
                image_data_path=paths["image_data"],
                image_root=paths["image_root"],
                registry_path=paths["registry"],
                vg2coco_path=paths["vg2coco"],
                psg2coco_path=paths["psg2coco"],
            )

        self.assertEqual(
            report_one,
            report_two,
        )
        self.assertEqual(
            coco_bytes(one),
            coco_bytes(two),
        )

    def test_end_to_end_rebuild_packs_and_materializes_canonical_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = make_inputs(root)
            vocabulary = root / "vocab.json"
            write_json(
                vocabulary,
                {
                    "schema": VOCAB_SCHEMA,
                    "predicates": [
                        "riding on",
                    ],
                    "objects": [
                        "person",
                        "horse",
                        "saddle",
                        "lamp",
                        "table",
                    ],
                },
            )
            evidence = rebuild_vg_raw_source(
                relationships_path=paths["relationships"],
                image_data_path=paths["image_data"],
                image_root=paths["image_root"],
                registry_path=paths["registry"],
                vg2coco_path=paths["vg2coco"],
                psg2coco_path=paths["psg2coco"],
                vocabulary_path=vocabulary,
                coco_output=root / "vg_raw.json",
                pack_output=root / "pack",
                canonical_output=root / "canonical.jsonl",
            )
            manifest = DatasetManifest.load(
                root / "canonical.jsonl",
                RelationVocabulary.load(vocabulary),
            )

        self.assertEqual(
            evidence["schema"],
            "kfcore.apache-vg-raw-source-rebuild/1",
        )
        self.assertEqual(
            evidence["pack"]["dataset"],
            "vg_raw",
        )
        self.assertEqual(
            evidence["pack"]["split"],
            "train",
        )
        self.assertEqual(
            evidence["pack"]["max_objects"],
            40,
        )
        self.assertEqual(
            evidence["pack"]["min_rels"],
            1,
        )
        self.assertEqual(
            evidence["canonical"]["images"],
            1,
        )
        self.assertEqual(
            evidence["canonical"]["relations"],
            1,
        )
        self.assertEqual(
            evidence["canonical"]["canonical_annotations_sha256"],
            manifest.annotations_sha256,
        )
        self.assertEqual(
            len(manifest.examples),
            1,
        )
        self.assertEqual(
            manifest.examples[0].relations,
            ((0, 0, 1),),
        )

    def test_malformed_registry_or_maps_fail_closed(self):
        cases = (
            ("registry", {"protected_vg_ids": [], "protected_coco_ids": ["42"]}),
            ("vg2coco", {"4": "4.2"}),
            ("psg2coco", {"x": -1}),
        )
        for target, bad_payload in cases:
            with self.subTest(target=target):
                with tempfile.TemporaryDirectory() as directory:
                    paths = make_inputs(Path(directory))
                    write_json(paths[target], bad_payload)
                    with self.assertRaises(ValueError):
                        convert_vg_raw(
                            relationships_path=paths["relationships"],
                            image_data_path=paths["image_data"],
                            image_root=paths["image_root"],
                            registry_path=paths["registry"],
                            vg2coco_path=paths["vg2coco"],
                            psg2coco_path=paths["psg2coco"],
                        )


if __name__ == "__main__":
    unittest.main()
