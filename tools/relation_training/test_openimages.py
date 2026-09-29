from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from benchmark import DatasetManifest, RelationVocabulary
from openimages import (
    build_vocabulary,
    convert_relationship_file,
    image_id_payload,
    load_class_descriptions,
    scan_relationship_files,
    stable_manifest_json,
    vocabulary_payload,
)


HEADER = [
    "ImageID",
    "LabelName1",
    "LabelName2",
    "XMin1",
    "XMax1",
    "YMin1",
    "YMax1",
    "XMin2",
    "XMax2",
    "YMin2",
    "YMax2",
    "RelationLabel",
]


def relationship(
    image_id: str,
    subject_mid: str,
    object_mid: str,
    predicate: str,
    subject_box=(0.0, 0.4, 0.0, 1.0),
    object_box=(0.5, 1.0, 0.0, 1.0),
):
    return [
        image_id,
        subject_mid,
        object_mid,
        str(subject_box[0]),
        str(subject_box[1]),
        str(subject_box[2]),
        str(subject_box[3]),
        str(object_box[0]),
        str(object_box[1]),
        str(object_box[2]),
        str(object_box[3]),
        predicate,
    ]


def write_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)


class OpenImagesConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.classes = self.root / "classes.csv"
        self.classes.write_text(
            "/m/person,Person\n"
            "/m/bike,Bicycle\n"
            "/m/red,Red\n",
            encoding="utf-8",
        )
        Image.new("RGB", (100, 50), (10, 20, 30)).save(
            self.root / "abc123.jpg"
        )
        Image.new("RGB", (80, 40), (30, 20, 10)).save(
            self.root / "def456.jpg"
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def rows(self):
        return [
            relationship(
                "abc123", "/m/person", "/m/bike", "on"
            ),
            relationship(
                "abc123", "/m/person", "/m/bike", "near"
            ),
            relationship(
                "abc123",
                "/m/person",
                "/m/red",
                "is",
                subject_box=(0.0, 0.4, 0.0, 1.0),
                object_box=(0.0, 0.4, 0.0, 1.0),
            ),
            relationship(
                "def456",
                "/m/bike",
                "/m/person",
                "behind",
                subject_box=(0.1, 0.5, 0.2, 0.8),
                object_box=(0.55, 0.95, 0.1, 0.9),
            ),
        ]

    def shared_vocabulary(self, csv_path: Path):
        classes = load_class_descriptions(self.classes)
        summary = scan_relationship_files([csv_path])
        return build_vocabulary([summary], classes), summary

    def test_scan_builds_downloader_ids_and_skips_attributes(self):
        csv_path = self.root / "train.csv"
        write_csv(csv_path, self.rows())
        vocabulary, summary = self.shared_vocabulary(csv_path)

        self.assertEqual(summary.relationships, 4)
        self.assertEqual(summary.object_relationships, 3)
        self.assertEqual(summary.skipped_attributes, 1)
        self.assertEqual(summary.images, ("abc123", "def456"))
        self.assertEqual(
            summary.predicates, ("behind", "near", "on")
        )
        self.assertEqual(
            vocabulary.predicates, ("behind", "near", "on")
        )
        self.assertEqual(
            vocabulary.object_labels, ("Bicycle", "Person")
        )
        self.assertEqual(
            image_id_payload("train", summary.images),
            "train/abc123\ntrain/def456\n",
        )
        payload = json.loads(vocabulary_payload(vocabulary))
        self.assertEqual(payload["schema"], "kfcore.relation-vocab/1")

    def test_conversion_preserves_multi_label_pair_and_validates_manifest(self):
        csv_path = self.root / "train.csv"
        write_csv(csv_path, self.rows())
        classes = load_class_descriptions(self.classes)
        vocabulary, _ = self.shared_vocabulary(csv_path)

        output, manifest = convert_relationship_file(
            csv_path,
            split="train",
            image_root=self.root,
            class_descriptions=classes,
            vocabulary=vocabulary,
        )
        records = [
            json.loads(line)
            for line in output.splitlines()
        ]
        self.assertEqual(len(records), 2)
        first = records[0]
        self.assertEqual(first["image"], "abc123.jpg")
        self.assertEqual(first["width"], 100)
        self.assertEqual(first["height"], 50)
        self.assertEqual(first["object_labels"], ["Bicycle", "Person"])
        self.assertEqual(
            first["boxes_xyxy"],
            [
                [50.0, 0.0, 100.0, 50.0],
                [0.0, 0.0, 40.0, 50.0],
            ],
        )

        # The same ordered object pair carries both near and on.
        pair = (1, 0)
        predicates = {
            relation[1]
            for relation in first["relations"]
            if (relation[0], relation[2]) == pair
        }
        self.assertEqual(
            predicates,
            {
                vocabulary.predicates.index("near"),
                vocabulary.predicates.index("on"),
            },
        )
        self.assertEqual(manifest["source_rows"], 4)
        self.assertEqual(manifest["skipped_attribute_rows"], 1)
        self.assertEqual(manifest["relations"], 3)

        validation_path = self.root / "converted.jsonl"
        validation_path.write_text(output, encoding="utf-8")
        loaded = DatasetManifest.load(
            validation_path, vocabulary
        )
        self.assertEqual(len(loaded.examples), 2)

    def test_row_order_permutation_is_byte_identical(self):
        first_path = self.root / "first.csv"
        second_path = self.root / "second.csv"
        rows = self.rows()
        write_csv(first_path, rows)
        write_csv(second_path, list(reversed(rows)))
        classes = load_class_descriptions(self.classes)

        first_summary = scan_relationship_files([first_path])
        second_summary = scan_relationship_files([second_path])
        vocabulary = build_vocabulary(
            [first_summary, second_summary], classes
        )

        first, first_manifest = convert_relationship_file(
            first_path,
            split="train",
            image_root=self.root,
            class_descriptions=classes,
            vocabulary=vocabulary,
        )
        second, second_manifest = convert_relationship_file(
            second_path,
            split="train",
            image_root=self.root,
            class_descriptions=classes,
            vocabulary=vocabulary,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first_manifest["output_sha256"],
            second_manifest["output_sha256"],
        )

        # Source hashes differ, while stable manifest serialization itself is valid.
        self.assertNotEqual(
            first_manifest["source_sha256"],
            second_manifest["source_sha256"],
        )
        self.assertTrue(
            stable_manifest_json(first_manifest).endswith("\n")
        )

    def test_missing_image_invalid_box_and_self_relation_fail(self):
        classes = load_class_descriptions(self.classes)
        vocabulary = RelationVocabulary(
            predicates=("on",),
            object_labels=("Bicycle", "Person"),
        )

        missing = self.root / "missing.csv"
        write_csv(
            missing,
            [relationship("not_here", "/m/person", "/m/bike", "on")],
        )
        with self.assertRaises(FileNotFoundError):
            convert_relationship_file(
                missing,
                split="train",
                image_root=self.root,
                class_descriptions=classes,
                vocabulary=vocabulary,
            )

        invalid = self.root / "invalid.csv"
        write_csv(
            invalid,
            [
                relationship(
                    "abc123",
                    "/m/person",
                    "/m/bike",
                    "on",
                    object_box=(0.5, 1.2, 0.0, 1.0),
                )
            ],
        )
        with self.assertRaises(ValueError):
            scan_relationship_files([invalid])

        self_relation = self.root / "self.csv"
        write_csv(
            self_relation,
            [
                relationship(
                    "abc123",
                    "/m/person",
                    "/m/person",
                    "on",
                    subject_box=(0.0, 0.4, 0.0, 1.0),
                    object_box=(0.0, 0.4, 0.0, 1.0),
                )
            ],
        )
        with self.assertRaises(ValueError):
            scan_relationship_files([self_relation])


if __name__ == "__main__":
    unittest.main()
