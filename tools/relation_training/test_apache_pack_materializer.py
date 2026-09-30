from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from apache_pack_materializer import (
    EVIDENCE_SCHEMA,
    REQUIRED_PACK_FILES,
    materialize_pack,
)
from benchmark import (
    DatasetManifest,
    RelationVocabulary,
)


def write_pack(
    root: Path,
    *,
    predicates: list[str] | None = None,
    categories: list[str] | None = None,
    boxes: np.ndarray | None = None,
    img_meta: np.ndarray | None = None,
    rels: np.ndarray | None = None,
) -> Path:
    pack = root / "pack"
    pack.mkdir()

    predicates = (
        predicates
        if predicates is not None
        else ["under", "riding"]
    )
    categories = (
        categories
        if categories is not None
        else ["horse", "person"]
    )
    boxes = (
        boxes
        if boxes is not None
        else np.asarray(
            [
                [0.25, 0.25, 0.50, 0.50],
                [0.75, 0.75, 0.50, 0.50],
            ],
            dtype=np.float32,
        )
    )
    rels = (
        rels
        if rels is not None
        else np.asarray(
            [
                [0, 1, 1, 0, -1],
                [1, 0, 0, 0, -1],
            ],
            dtype=np.int32,
        )
    )
    img_meta = (
        img_meta
        if img_meta is not None
        else np.asarray(
            [
                [
                    11,
                    8,
                    8,
                    0,
                    len(boxes),
                    0,
                    len(rels),
                ]
            ],
            dtype=np.int64,
        )
    )
    box_cats = np.asarray(
        [1, 0],
        dtype=np.int32,
    )
    if len(boxes) != len(box_cats):
        box_cats = np.zeros(
            len(boxes),
            dtype=np.int32,
        )

    meta = {
        "dataset": "fixture",
        "split": "train",
        "num_images": int(
            img_meta.shape[0]
        ),
        "num_boxes": int(
            boxes.shape[0]
        ),
        "num_rels": int(
            rels.shape[0]
        ),
        "predicates": predicates,
        "categories": categories,
    }
    (pack / "meta.json").write_text(
        json.dumps(meta),
        encoding="utf-8",
    )
    (pack / "file_names.json").write_text(
        json.dumps(["x.jpg"]),
        encoding="utf-8",
    )
    np.save(
        pack / "img_meta.npy",
        img_meta,
    )
    np.save(
        pack / "boxes.npy",
        boxes,
    )
    np.save(
        pack / "box_cats.npy",
        box_cats,
    )
    np.save(
        pack / "rels.npy",
        rels,
    )
    return pack


def vocab() -> RelationVocabulary:
    return RelationVocabulary(
        ("riding", "under"),
        ("person", "horse"),
    )


class ApachePackMaterializerTest(
    unittest.TestCase
):
    def test_pack_materializes_by_name_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(root)
            out = root / "train.jsonl"
            evidence = materialize_pack(
                pack,
                vocab(),
                out,
            )
            manifest = DatasetManifest.load(
                out,
                vocab(),
            )

        self.assertEqual(
            evidence["schema"],
            EVIDENCE_SCHEMA,
        )
        self.assertEqual(
            evidence["images"],
            1,
        )
        self.assertEqual(
            evidence["relations"],
            2,
        )
        self.assertEqual(
            evidence[
                "canonical_annotations_sha256"
            ],
            manifest.annotations_sha256,
        )
        self.assertEqual(
            set(
                evidence[
                    "component_sha256"
                ]
            ),
            set(REQUIRED_PACK_FILES),
        )
        self.assertTrue(
            all(
                len(value) == 64
                for value in evidence[
                    "component_sha256"
                ].values()
            )
        )

        example = manifest.examples[0]
        self.assertEqual(
            example.boxes_xyxy,
            (
                (0.0, 0.0, 4.0, 4.0),
                (4.0, 4.0, 8.0, 8.0),
            ),
        )
        self.assertEqual(
            example.object_labels,
            ("person", "horse"),
        )
        self.assertEqual(
            example.relations,
            (
                (0, 0, 1),
                (1, 1, 0),
            ),
        )

    def test_output_bytes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(root)
            one = root / "one.jsonl"
            two = root / "two.jsonl"
            first = materialize_pack(
                pack,
                vocab(),
                one,
            )
            second = materialize_pack(
                pack,
                vocab(),
                two,
            )
            first_bytes = one.read_bytes()
            second_bytes = two.read_bytes()

        self.assertEqual(
            first_bytes,
            second_bytes,
        )
        self.assertEqual(
            first[
                "canonical_annotations_sha256"
            ],
            second[
                "canonical_annotations_sha256"
            ],
        )

    def test_unknown_local_predicate_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                predicates=[
                    "unknown-relation",
                    "riding",
                ],
            )
            with self.assertRaisesRegex(
                ValueError,
                "predicate is absent",
            ):
                materialize_pack(
                    pack,
                    vocab(),
                    root / "out.jsonl",
                )

    def test_unknown_local_category_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                categories=[
                    "horse",
                    "unknown-object",
                ],
            )
            with self.assertRaisesRegex(
                ValueError,
                "category is absent",
            ):
                materialize_pack(
                    pack,
                    vocab(),
                    root / "out.jsonl",
                )

    def test_nonrepresentable_box_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                boxes=np.asarray(
                    [
                        [0.0, 0.25, 0.50, 0.50],
                        [0.75, 0.75, 0.50, 0.50],
                    ],
                    dtype=np.float32,
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "cannot be represented",
            ):
                materialize_pack(
                    pack,
                    vocab(),
                    root / "out.jsonl",
                )

    def test_malformed_offsets_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                img_meta=np.asarray(
                    [
                        [
                            11,
                            8,
                            8,
                            99,
                            2,
                            0,
                            2,
                        ]
                    ],
                    dtype=np.int64,
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "invalid box range",
            ):
                materialize_pack(
                    pack,
                    vocab(),
                    root / "out.jsonl",
                )

    def test_duplicate_canonical_relation_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                rels=np.asarray(
                    [
                        [0, 1, 1, 0, -1],
                        [0, 1, 1, 0, -1],
                    ],
                    dtype=np.int32,
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "duplicate canonical relation",
            ):
                materialize_pack(
                    pack,
                    vocab(),
                    root / "out.jsonl",
                )

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(root)
            out = root / "out.jsonl"
            out.write_text(
                "do not overwrite",
                encoding="utf-8",
            )
            with self.assertRaises(
                FileExistsError
            ):
                materialize_pack(
                    pack,
                    vocab(),
                    out,
                )


if __name__ == "__main__":
    unittest.main()
