from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from apache_pack_builder import (
    EVIDENCE_SCHEMA,
    FLAG_GEOMETRIC,
    FLAG_SPATIAL,
    PACK_COMPONENTS,
    RELEASED_MAX_OBJECTS,
    RELEASED_MIN_RELS,
    ROUND_SHIFT,
    pack_coco_sgg_train,
)
from apache_pack_materializer import (
    materialize_pack,
)
from benchmark import (
    DatasetManifest,
    RelationVocabulary,
)


def write_coco(
    path: Path,
    payload: dict,
) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def base_payload() -> dict:
    return {
        "images": [
            {
                "id": 100,
                "file_name": "keep.jpg",
                "width": 100,
                "height": 100,
            },
            {
                "id": 200,
                "file_name": "norel.jpg",
                "width": 80,
                "height": 60,
            },
            {
                "id": 300,
                "file_name": "onebox.jpg",
                "width": 80,
                "height": 60,
            },
        ],
        "annotations": [
            {
                "id": 11,
                "image_id": 100,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
            },
            {
                "id": 12,
                "image_id": 100,
                "category_id": 10,
                "bbox": [50, 50, -10, 5],
            },
            {
                "id": 13,
                "image_id": 100,
                "category_id": 1,
                "bbox": [-20, 5, 0.5, 200],
            },
            {
                "id": 21,
                "image_id": 200,
                "category_id": 1,
                "bbox": [1, 1, 10, 10],
            },
            {
                "id": 22,
                "image_id": 200,
                "category_id": 10,
                "bbox": [20, 20, 10, 10],
            },
            {
                "id": 31,
                "image_id": 300,
                "category_id": 1,
                "bbox": [1, 1, 10, 10],
            },
        ],
        "categories": [
            {
                "id": 10,
                "name": "horse",
            },
            {
                "id": 1,
                "name": "person",
            },
        ],
        "rel_categories": [
            {
                "id": 7,
                "name": "holding",
            },
            {
                "id": 2,
                "name": "above",
            },
        ],
        "rel_annotations": [
            {
                "image_id": 100,
                "subject_id": 11,
                "object_id": 12,
                "predicate_id": 7,
                "spatial": True,
                "source": "geometric",
                "round": 2,
                "predicate_raw": "holds onto",
            },
            {
                "image_id": 100,
                "subject_id": 12,
                "object_id": 11,
                "predicate_id": 2,
            },
            {
                "image_id": 100,
                "subject_id": 11,
                "object_id": 11,
                "predicate_id": 7,
            },
        ],
    }


class ApachePackBuilderTest(
    unittest.TestCase
):
    def test_exact_pack_arrays_flags_vocab_and_drops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            write_coco(
                source,
                base_payload(),
            )
            pack = root / "pack"
            evidence = pack_coco_sgg_train(
                source,
                pack,
                dataset_name="fixture",
                ann_source_label="rebuild/input.json",
                img_dir_label="datasets/fixture/train",
            )

            img_meta = np.load(
                pack / "img_meta.npy",
                allow_pickle=False,
            )
            boxes = np.load(
                pack / "boxes.npy",
                allow_pickle=False,
            )
            cats = np.load(
                pack / "box_cats.npy",
                allow_pickle=False,
            )
            rels = np.load(
                pack / "rels.npy",
                allow_pickle=False,
            )
            meta = json.loads(
                (
                    pack / "meta.json"
                ).read_text(
                    encoding="utf-8"
                )
            )
            file_names = json.loads(
                (
                    pack
                    / "file_names.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            evidence["schema"],
            EVIDENCE_SCHEMA,
        )
        self.assertEqual(
            evidence["max_objects"],
            RELEASED_MAX_OBJECTS,
        )
        self.assertEqual(
            evidence["min_rels"],
            RELEASED_MIN_RELS,
        )
        self.assertEqual(
            img_meta.tolist(),
            [[100, 100, 100, 0, 3, 0, 2]],
        )
        self.assertEqual(
            file_names,
            ["keep.jpg"],
        )
        self.assertEqual(
            cats.tolist(),
            [1, 0, 1],
        )
        self.assertTrue(
            np.array_equal(
                boxes,
                np.asarray(
                    [
                        [0.20, 0.20, 0.20, 0.20],
                        [0.55, 0.525, 0.10, 0.05],
                        [0.0, 1.0, 0.005, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )
        expected_flags = (
            FLAG_SPATIAL
            | FLAG_GEOMETRIC
            | (2 << ROUND_SHIFT)
        )
        self.assertEqual(
            rels.tolist(),
            [
                [0, 1, 0, expected_flags, 0],
                [1, 0, 1, 0, -1],
            ],
        )

        self.assertEqual(
            meta["predicates"],
            ["holding", "above"],
        )
        self.assertEqual(
            meta["categories"],
            ["horse", "person"],
        )
        self.assertEqual(
            meta["category_ids"],
            [10, 1],
        )
        self.assertEqual(
            meta["raw_predicates"],
            ["holds onto"],
        )
        self.assertEqual(
            meta["raw_links"],
            [
                {
                    "raw": "holds onto",
                    "predicate": "holding",
                    "count": 1,
                }
            ],
        )
        self.assertEqual(
            meta["predicate_counts"],
            {
                "holding": 1,
                "above": 1,
            },
        )
        self.assertEqual(
            meta["drops"],
            {
                "images_fewer_than_2_boxes": 1,
                "images_below_min_rels": 1,
                "rels_beyond_max_objects": 0,
                "rels_self_loop": 1,
                "degenerate_boxes_kept": 1,
            },
        )
        self.assertEqual(
            meta["ann_source"],
            "rebuild/input.json",
        )
        self.assertEqual(
            meta["img_dir"],
            "datasets/fixture/train",
        )

    def test_max_objects_40_window_drops_outside_relation(self):
        payload = {
            "images": [
                {
                    "id": 1,
                    "file_name": "x.jpg",
                    "width": 100,
                    "height": 100,
                }
            ],
            "annotations": [
                {
                    "id": index + 1,
                    "image_id": 1,
                    "category_id": (
                        10
                        if index % 2
                        else 1
                    ),
                    "bbox": [
                        1 + index * 0.1,
                        2,
                        2,
                        2,
                    ],
                }
                for index in range(41)
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 10, "name": "horse"},
            ],
            "rel_categories": [
                {"id": 3, "name": "near"},
            ],
            "rel_annotations": [
                {
                    "image_id": 1,
                    "subject_id": 1,
                    "object_id": 2,
                    "predicate_id": 3,
                },
                {
                    "image_id": 1,
                    "subject_id": 1,
                    "object_id": 41,
                    "predicate_id": 3,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            write_coco(source, payload)
            pack = root / "pack"
            pack_coco_sgg_train(
                source,
                pack,
                dataset_name="window",
                ann_source_label="window.json",
                img_dir_label="datasets/window/train",
            )
            meta = json.loads(
                (pack / "meta.json").read_text(encoding="utf-8")
            )
            img_meta = np.load(pack / "img_meta.npy")
            rels = np.load(pack / "rels.npy")

        self.assertEqual(
            int(img_meta[0, 4]),
            40,
        )
        self.assertEqual(
            rels.shape,
            (1, 5),
        )
        self.assertEqual(
            meta["drops"]["rels_beyond_max_objects"],
            1,
        )

    def test_exclusion_happens_before_duplicate_image_id_check(self):
        payload = base_payload()
        payload["images"] = [
            {
                "id": 100,
                "file_name": "excluded.jpg",
                "width": 100,
                "height": 100,
            },
            payload["images"][0],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            write_coco(source, payload)
            pack = root / "pack"
            evidence = pack_coco_sgg_train(
                source,
                pack,
                dataset_name="exclude",
                ann_source_label="exclude.json",
                img_dir_label="datasets/exclude/train",
                exclude_file_names=["excluded.jpg"],
            )

        self.assertEqual(
            evidence["excluded_image_count"],
            1,
        )
        self.assertEqual(
            evidence["packed_image_count"],
            1,
        )

    def test_duplicate_kept_image_id_fails_closed(self):
        payload = base_payload()
        payload["images"] = [
            payload["images"][0],
            {
                **payload["images"][0],
                "file_name": "other.jpg",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            write_coco(source, payload)
            with self.assertRaisesRegex(
                ValueError,
                "duplicate image id",
            ):
                pack_coco_sgg_train(
                    source,
                    root / "pack",
                    dataset_name="bad",
                    ann_source_label="bad.json",
                    img_dir_label="datasets/bad/train",
                )

    def test_unknown_category_and_predicate_ids_fail_closed(self):
        for kind in ("category", "predicate"):
            with self.subTest(kind=kind):
                payload = base_payload()
                if kind == "category":
                    payload["annotations"][0]["category_id"] = 999
                    message = "absent from categories"
                else:
                    payload["rel_annotations"][0]["predicate_id"] = 999
                    message = "absent from rel_categories"
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = root / "input.json"
                    write_coco(source, payload)
                    with self.assertRaisesRegex(ValueError, message):
                        pack_coco_sgg_train(
                            source,
                            root / "pack",
                            dataset_name="bad",
                            ann_source_label="bad.json",
                            img_dir_label="datasets/bad/train",
                        )

    def test_identical_logical_inputs_produce_byte_identical_components(self):
        payload = base_payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            write_coco(source, payload)
            first = root / "first"
            second = root / "second"
            report_one = pack_coco_sgg_train(
                source,
                first,
                dataset_name="deterministic",
                ann_source_label="logical/input.json",
                img_dir_label="datasets/deterministic/train",
            )
            report_two = pack_coco_sgg_train(
                source,
                second,
                dataset_name="deterministic",
                ann_source_label="logical/input.json",
                img_dir_label="datasets/deterministic/train",
            )
            first_bytes = {
                name: (first / name).read_bytes()
                for name in PACK_COMPONENTS
            }
            second_bytes = {
                name: (second / name).read_bytes()
                for name in PACK_COMPONENTS
            }

        self.assertEqual(
            report_one,
            report_two,
        )
        self.assertEqual(
            first_bytes,
            second_bytes,
        )

    def test_pack_round_trips_through_existing_materializer(self):
        payload = {
            "images": [
                {
                    "id": 9,
                    "file_name": "round.jpg",
                    "width": 100,
                    "height": 80,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 9,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                },
                {
                    "id": 2,
                    "image_id": 9,
                    "category_id": 2,
                    "bbox": [50, 30, 20, 20],
                },
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "horse"},
            ],
            "rel_categories": [
                {"id": 4, "name": "riding"},
            ],
            "rel_annotations": [
                {
                    "image_id": 9,
                    "subject_id": 1,
                    "object_id": 2,
                    "predicate_id": 4,
                }
            ],
        }
        vocabulary = RelationVocabulary(
            ("riding",),
            ("person", "horse"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            write_coco(source, payload)
            pack = root / "pack"
            pack_coco_sgg_train(
                source,
                pack,
                dataset_name="round",
                ann_source_label="round.json",
                img_dir_label="datasets/round/train",
            )
            canonical = root / "canonical.jsonl"
            evidence = materialize_pack(
                pack,
                vocabulary,
                canonical,
            )
            manifest = DatasetManifest.load(
                canonical,
                vocabulary,
            )

        self.assertEqual(
            evidence["images"],
            1,
        )
        self.assertEqual(
            evidence["relations"],
            1,
        )
        self.assertEqual(
            len(manifest.examples),
            1,
        )
        self.assertEqual(
            manifest.examples[0].relations,
            ((0, 0, 1),),
        )
        self.assertEqual(
            manifest.examples[0].object_labels,
            ("person", "horse"),
        )


if __name__ == "__main__":
    unittest.main()
