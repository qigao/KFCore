from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from apache_vg_raw_rebuild import (
    EVIDENCE_SCHEMA,
    RELEASED_ANN_SOURCE_LABEL,
    RELEASED_IMG_DIR_LABEL,
    convert_vg_raw,
    rebuild_vg_raw,
    sha256_file,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class ApacheVGRawRebuildTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, Path]:
        image_root = root / "VG150_coco_format" / "train"
        image_root.mkdir(parents=True)
        for name in ("1.jpg", "2.jpg", "3.jpg", "4.jpg", "6.jpg"):
            (image_root / name).write_bytes(b"x")
        (image_root / "7.JPG").write_bytes(b"x")

        relationships = [
            {
                "image_id": 1,
                "relationships": [
                    {
                        "predicate": "  On   Top Of ",
                        "subject": {
                            "object_id": 10,
                            "name": "  Person ",
                            "x": 1,
                            "y": 2,
                            "w": 0,
                            "h": -3,
                        },
                        "object": {
                            "object_id": 20,
                            "names": [" Red   Ball "],
                            "x": 10,
                            "y": 11,
                            "w": 5,
                            "h": 6,
                        },
                    },
                    {
                        "predicate": "on top of",
                        "subject": {
                            "object_id": 10,
                            "name": "person",
                            "x": 100,
                            "y": 100,
                            "w": 100,
                            "h": 100,
                        },
                        "object": {
                            "object_id": 20,
                            "name": "red ball",
                            "x": 100,
                            "y": 100,
                            "w": 100,
                            "h": 100,
                        },
                    },
                    {
                        "predicate": "touching",
                        "subject": {
                            "object_id": 10,
                            "name": "person",
                            "x": 1,
                            "y": 2,
                            "w": 1,
                            "h": 1,
                        },
                        "object": {
                            "object_id": 30,
                            "names": ["one two three four five six"],
                            "x": 0,
                            "y": 0,
                            "w": 1,
                            "h": 1,
                        },
                    },
                    {
                        "predicate": "one two three four five six",
                        "subject": {
                            "object_id": 40,
                            "name": "never-added",
                            "x": 0,
                            "y": 0,
                            "w": 1,
                            "h": 1,
                        },
                        "object": {
                            "object_id": 50,
                            "name": "never-added-2",
                            "x": 0,
                            "y": 0,
                            "w": 1,
                            "h": 1,
                        },
                    },
                    {
                        "predicate": "x" * 61,
                        "subject": {
                            "object_id": 10,
                            "name": "person",
                            "x": 1,
                            "y": 2,
                            "w": 1,
                            "h": 1,
                        },
                        "object": {
                            "object_id": 60,
                            "name": "pole",
                            "x": 30,
                            "y": 30,
                            "w": 2,
                            "h": 3,
                        },
                    },
                ],
            },
            {
                "image_id": 2,
                "relationships": [],
            },
            {
                "image_id": 3,
                "relationships": [],
            },
            {
                "image_id": 4,
                "relationships": [],
            },
            {
                "image_id": 5,
                "relationships": [],
            },
            {
                "image_id": 6,
                "relationships": [
                    {
                        "predicate": "",
                        "subject": {
                            "object_id": 1,
                            "name": "a",
                            "x": 0,
                            "y": 0,
                            "w": 2,
                            "h": 2,
                        },
                        "object": {
                            "object_id": 2,
                            "name": "b",
                            "x": 2,
                            "y": 2,
                            "w": 2,
                            "h": 2,
                        },
                    }
                ],
            },
            {
                "image_id": 7,
                "relationships": [],
            },
        ]
        image_data = [
            {"image_id": 1, "width": 100, "height": 80},
            {"image_id": 2, "width": 100, "height": 80},
            {"image_id": 3, "width": 100, "height": 80},
            {"image_id": 6, "width": 100, "height": 80},
            {"image_id": 7, "width": 100, "height": 80},
        ]
        registry = {
            "protected_vg_ids": ["2"],
            "protected_coco_ids": [123],
            "banned_coco_ids": [999],
        }
        vg2coco = {"3": 999}
        psg2coco = {"42": 123}

        paths = {
            "relationships": root / "relationships.json",
            "image_data": root / "image_data.json",
            "registry": root / "registry.json",
            "vg2coco": root / "vg2coco.json",
            "psg2coco": root / "psg2coco.json",
            "image_root": image_root,
        }
        write_json(paths["relationships"], relationships)
        write_json(paths["image_data"], image_data)
        write_json(paths["registry"], registry)
        write_json(paths["vg2coco"], vg2coco)
        write_json(paths["psg2coco"], psg2coco)
        return paths

    def test_converter_matches_released_vg_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            paths = self.make_fixture(Path(td))
            payload, evidence = convert_vg_raw(
                relationships_path=paths["relationships"],
                image_data_path=paths["image_data"],
                image_root=paths["image_root"],
                registry_path=paths["registry"],
                vg2coco_path=paths["vg2coco"],
                psg2coco_path=paths["psg2coco"],
            )

            self.assertEqual(evidence["schema"], EVIDENCE_SCHEMA)
            self.assertFalse(evidence["input_hashes_pinned"])
            self.assertEqual(payload["images"], [
                {"id": 1, "file_name": "1.jpg", "width": 100, "height": 80}
            ])
            self.assertEqual(
                [row["name"] for row in payload["categories"]],
                ["person", "red ball", "pole"],
            )
            self.assertEqual(
                [row["name"] for row in payload["rel_categories"]],
                ["on top of"],
            )
            self.assertEqual(
                [row["bbox"] for row in payload["annotations"]],
                [
                    [1.0, 2.0, 1.0, 1.0],
                    [10.0, 11.0, 5.0, 6.0],
                    [30.0, 30.0, 2.0, 3.0],
                ],
            )
            self.assertEqual(len(payload["rel_annotations"]), 1)
            relation = payload["rel_annotations"][0]
            self.assertEqual(
                (relation["subject_id"], relation["object_id"], relation["predicate_id"]),
                (1, 2, 0),
            )

            self.assertEqual(evidence["drops"]["eval_protected"], 2)
            self.assertEqual(evidence["drops"]["no_dims"], 1)
            self.assertEqual(evidence["drops"]["image_not_on_disk"], 2)
            self.assertEqual(evidence["drops"]["obj_name_too_long"], 1)
            self.assertEqual(evidence["drops"]["pred_too_long"], 1)
            self.assertEqual(evidence["drops"]["pred_empty"], 1)
            self.assertEqual(evidence["drops"]["image_no_usable_rels"], 1)
            self.assertEqual(evidence["output"]["leakage_count"], 0)
            self.assertEqual(evidence["output"]["image_count"], 1)
            self.assertEqual(evidence["output"]["annotation_count"], 3)
            self.assertEqual(evidence["output"]["relation_count"], 1)
            self.assertEqual(evidence["image_stem_allow_set"]["count"], 5)

    def test_banned_coco_supersedes_protected_coco(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.make_fixture(root)
            # Image 3 maps to 999. It must be protected by banned_coco_ids,
            # not by the fallback protected_coco_ids=[123].
            _, evidence = convert_vg_raw(
                relationships_path=paths["relationships"],
                image_data_path=paths["image_data"],
                image_root=paths["image_root"],
                registry_path=paths["registry"],
                vg2coco_path=paths["vg2coco"],
                psg2coco_path=paths["psg2coco"],
            )
            self.assertEqual(
                evidence["registry_policy"]["protected_coco_key"],
                "banned_coco_ids",
            )

    def test_pinned_input_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            paths = self.make_fixture(Path(td))
            expected = {
                "relationships_json": sha256_file(paths["relationships"]),
                "image_data_json": sha256_file(paths["image_data"]),
                "registry_json": sha256_file(paths["registry"]),
                "vg2coco_json": sha256_file(paths["vg2coco"]),
                "psg2coco_json": sha256_file(paths["psg2coco"]),
            }
            _, evidence = convert_vg_raw(
                relationships_path=paths["relationships"],
                image_data_path=paths["image_data"],
                image_root=paths["image_root"],
                registry_path=paths["registry"],
                vg2coco_path=paths["vg2coco"],
                psg2coco_path=paths["psg2coco"],
                expected_input_sha256=expected,
            )
            self.assertTrue(evidence["input_hashes_pinned"])

            write_json(paths["vg2coco"], {"3": 998})
            with self.assertRaisesRegex(ValueError, "vg2coco_json SHA-256 drift"):
                convert_vg_raw(
                    relationships_path=paths["relationships"],
                    image_data_path=paths["image_data"],
                    image_root=paths["image_root"],
                    registry_path=paths["registry"],
                    vg2coco_path=paths["vg2coco"],
                    psg2coco_path=paths["psg2coco"],
                    expected_input_sha256=expected,
                )

    def test_rebuild_pack_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.make_fixture(root)
            vocabulary = root / "vocabulary.json"
            write_json(
                vocabulary,
                {
                    "schema": "kfcore.relation-vocab/1",
                    "predicates": ["on top of"],
                    "objects": ["person", "red ball", "pole"],
                },
            )

            reports = []
            for suffix in ("a", "b"):
                report = rebuild_vg_raw(
                    relationships_path=paths["relationships"],
                    image_data_path=paths["image_data"],
                    image_root=paths["image_root"],
                    registry_path=paths["registry"],
                    vg2coco_path=paths["vg2coco"],
                    psg2coco_path=paths["psg2coco"],
                    coco_output=root / suffix / "vg_raw_train_coco.json",
                    pack_output=root / suffix / "vg_raw" / "train",
                    evidence_output=root / suffix / "evidence.json",
                    vocabulary_path=vocabulary,
                    canonical_output=root / suffix / "canonical.jsonl",
                )
                reports.append(report)

            self.assertEqual(
                reports[0]["output"]["coco_sgg_sha256"],
                reports[1]["output"]["coco_sgg_sha256"],
            )
            self.assertEqual(
                reports[0]["pack"]["component_sha256"],
                reports[1]["pack"]["component_sha256"],
            )
            self.assertEqual(
                reports[0]["pack"]["ann_source_label"],
                RELEASED_ANN_SOURCE_LABEL,
            )
            self.assertEqual(
                reports[0]["pack"]["img_dir_label"],
                RELEASED_IMG_DIR_LABEL,
            )
            self.assertIsNotNone(reports[0]["canonical_materialization"])
            self.assertEqual(
                reports[0]["canonical_materialization"]["canonical_annotations_sha256"],
                reports[1]["canonical_materialization"]["canonical_annotations_sha256"],
            )
            self.assertEqual(reports[0]["canonical_materialization"]["images"], 1)
            self.assertEqual(reports[0]["canonical_materialization"]["relations"], 1)

    def test_released_max_words_is_not_a_runtime_knob(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            paths = self.make_fixture(Path(td))
            with self.assertRaisesRegex(ValueError, "max_words=5"):
                convert_vg_raw(
                    relationships_path=paths["relationships"],
                    image_data_path=paths["image_data"],
                    image_root=paths["image_root"],
                    registry_path=paths["registry"],
                    vg2coco_path=paths["vg2coco"],
                    psg2coco_path=paths["psg2coco"],
                    max_words=4,
                )


if __name__ == "__main__":
    unittest.main()
