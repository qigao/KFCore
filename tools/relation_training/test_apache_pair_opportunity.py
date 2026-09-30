from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from apache_pair_opportunity import (
    NPZ_FORMAT,
    REBUILD_SCHEMA,
    RELEASED_MIN_SUPPORT,
    RELEASED_SCAN_BOX_CAP,
    rebuild_pair_opportunity,
    sha256_file,
    write_pair_opportunity,
)
from apache_pair_sampler import (
    PairOpportunityTable,
)


def write_pack(
    root: Path,
    *,
    categories: list[str],
    image_box_categories: list[
        list[int]
    ],
    image_relations: list[
        list[tuple[int, int]]
    ],
) -> Path:
    if len(image_box_categories) != len(
        image_relations
    ):
        raise ValueError("fixture length mismatch")

    pack = root / "megasg_clean" / "train"
    pack.mkdir(
        parents=True,
        exist_ok=True,
    )
    img_meta: list[
        tuple[
            int,
            int,
            int,
            int,
            int,
            int,
            int,
        ]
    ] = []
    box_rows: list[int] = []
    rel_rows: list[
        tuple[int, int, int, int, int]
    ] = []

    for image_index, (
        cats,
        relations,
    ) in enumerate(
        zip(
            image_box_categories,
            image_relations,
        )
    ):
        box_start = len(box_rows)
        rel_start = len(rel_rows)
        box_rows.extend(cats)
        rel_rows.extend(
            (
                subject,
                object_,
                0,
                0,
                -1,
            )
            for subject, object_ in relations
        )
        img_meta.append(
            (
                image_index + 1,
                640,
                480,
                box_start,
                len(cats),
                rel_start,
                len(relations),
            )
        )

    np.save(
        pack / "img_meta.npy",
        np.asarray(
            img_meta,
            dtype=np.int64,
        ),
    )
    np.save(
        pack / "box_cats.npy",
        np.asarray(
            box_rows,
            dtype=np.int32,
        ),
    )
    np.save(
        pack / "rels.npy",
        np.asarray(
            rel_rows,
            dtype=np.int32,
        ).reshape(-1, 5),
    )
    (pack / "meta.json").write_text(
        json.dumps(
            {
                "dataset": "megasg",
                "split": "train",
                "num_images": len(
                    img_meta
                ),
                "num_boxes": len(
                    box_rows
                ),
                "num_rels": len(
                    rel_rows
                ),
                "predicates": [
                    "related_to",
                ],
                "categories": (
                    categories
                ),
            }
        ),
        encoding="utf-8",
    )
    return pack


class ApachePairOpportunityRebuildTest(
    unittest.TestCase
):
    def test_exact_ordered_instance_pair_golden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                categories=[
                    "person",
                    "shirt",
                    "horse",
                ],
                image_box_categories=[
                    [0, 0, 1],
                    [0, 0],
                ],
                image_relations=[
                    [
                        (0, 2),
                        (1, 2),
                        (2, 0),
                    ],
                    [
                        (0, 1),
                    ],
                ],
            )
            arrays, evidence = (
                rebuild_pair_opportunity(
                    pack,
                    (
                        "person",
                        "shirt",
                        "horse",
                    ),
                )
            )

        C = 3
        opp = arrays[
            "opportunities"
        ].reshape(C, C)
        rel = arrays[
            "relations"
        ].reshape(C, C)
        rate = arrays[
            "rate"
        ].reshape(C, C)

        # image 1: person count=2, shirt count=1
        # image 2: person count=2
        self.assertEqual(
            int(opp[0, 0]),
            4,
        )
        self.assertEqual(
            int(opp[0, 1]),
            2,
        )
        self.assertEqual(
            int(opp[1, 0]),
            2,
        )
        self.assertEqual(
            int(opp[1, 1]),
            0,
        )

        self.assertEqual(
            int(rel[0, 0]),
            1,
        )
        self.assertEqual(
            int(rel[0, 1]),
            2,
        )
        self.assertEqual(
            int(rel[1, 0]),
            1,
        )
        self.assertEqual(
            float(rate[0, 0]),
            0.25,
        )
        self.assertEqual(
            float(rate[0, 1]),
            1.0,
        )
        self.assertEqual(
            float(rate[1, 0]),
            0.5,
        )
        self.assertNotEqual(
            float(rate[0, 1]),
            float(rate[1, 0]),
        )

        self.assertEqual(
            evidence["schema"],
            REBUILD_SCHEMA,
        )
        self.assertEqual(
            evidence["scan_box_cap"],
            RELEASED_SCAN_BOX_CAP,
        )
        self.assertEqual(
            evidence["min_support"],
            RELEASED_MIN_SUPPORT,
        )
        self.assertTrue(
            evidence["full_scan"]
        )
        self.assertFalse(
            evidence["extrapolated"]
        )
        self.assertEqual(
            evidence[
                "opportunity_semantics"
            ],
            "ordered-instance-pairs-minus-self-on-diagonal",
        )

    def test_relations_beyond_400_box_cap_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cats = [0] * 400 + [1]
            pack = write_pack(
                root,
                categories=[
                    "person",
                    "shirt",
                ],
                image_box_categories=[
                    cats
                ],
                image_relations=[
                    [
                        (0, 1),
                        (0, 400),
                    ]
                ],
            )
            arrays, evidence = (
                rebuild_pair_opportunity(
                    pack,
                    (
                        "person",
                        "shirt",
                    ),
                )
            )

        opp = arrays[
            "opportunities"
        ].reshape(2, 2)
        rel = arrays[
            "relations"
        ].reshape(2, 2)

        self.assertEqual(
            int(opp[0, 0]),
            400 * 399,
        )
        self.assertEqual(
            int(opp[0, 1]),
            0,
        )
        self.assertEqual(
            int(rel[0, 0]),
            1,
        )
        self.assertEqual(
            int(rel[0, 1]),
            0,
        )
        self.assertEqual(
            evidence[
                "relations_dropped_by_box_cap"
            ],
            1,
        )
        self.assertEqual(
            evidence[
                "relations_scanned"
            ],
            1,
        )

    def test_category_order_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                categories=[
                    "horse",
                    "person",
                ],
                image_box_categories=[
                    [0, 1]
                ],
                image_relations=[
                    []
                ],
            )
            with self.assertRaisesRegex(
                ValueError,
                "category order",
            ):
                rebuild_pair_opportunity(
                    pack,
                    (
                        "person",
                        "horse",
                    ),
                )

    def test_released_algorithm_constants_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                categories=[
                    "person",
                    "horse",
                ],
                image_box_categories=[
                    [0, 1]
                ],
                image_relations=[
                    []
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "source_name=megasg_clean",
            ):
                rebuild_pair_opportunity(
                    pack,
                    (
                        "person",
                        "horse",
                    ),
                    source_name="vg_raw",
                )
            with self.assertRaisesRegex(
                ValueError,
                "scan_box_cap=400",
            ):
                rebuild_pair_opportunity(
                    pack,
                    (
                        "person",
                        "horse",
                    ),
                    scan_box_cap=399,
                )
            with self.assertRaisesRegex(
                ValueError,
                "min_support=50",
            ):
                rebuild_pair_opportunity(
                    pack,
                    (
                        "person",
                        "horse",
                    ),
                    min_support=49,
                )

    def test_written_npz_is_byte_deterministic_and_runtime_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                categories=[
                    "person",
                    "horse",
                ],
                image_box_categories=[
                    [0, 1],
                    [0, 1],
                ],
                image_relations=[
                    [(0, 1)],
                    [],
                ],
            )
            one = root / "one.npz"
            two = root / "two.npz"
            one_evidence = (
                root / "one.json"
            )
            two_evidence = (
                root / "two.json"
            )

            report_one = (
                write_pair_opportunity(
                    pack_split=pack,
                    object_labels=(
                        "person",
                        "horse",
                    ),
                    output=one,
                    evidence_output=(
                        one_evidence
                    ),
                )
            )
            report_two = (
                write_pair_opportunity(
                    pack_split=pack,
                    object_labels=(
                        "person",
                        "horse",
                    ),
                    output=two,
                    evidence_output=(
                        two_evidence
                    ),
                )
            )
            table = (
                PairOpportunityTable.load(
                    one
                )
            )
            one_bytes = (
                one.read_bytes()
            )
            two_bytes = (
                two.read_bytes()
            )
            box_cats_sha = sha256_file(
                pack
                / "box_cats.npy"
            )

        self.assertEqual(
            one_bytes,
            two_bytes,
        )
        self.assertEqual(
            report_one[
                "output_sha256"
            ],
            report_two[
                "output_sha256"
            ],
        )
        self.assertEqual(
            report_one[
                "output_sha256"
            ],
            hashlib_sha256(
                one_bytes
            ),
        )
        self.assertEqual(
            report_one["npz_format"],
            NPZ_FORMAT,
        )
        self.assertEqual(
            table.num_cats,
            2,
        )
        self.assertEqual(
            table.min_support,
            50,
        )
        self.assertEqual(
            table.opportunities.tolist(),
            [0, 2, 2, 0],
        )
        self.assertEqual(
            table.relations.tolist(),
            [0, 1, 0, 0],
        )
        self.assertEqual(
            box_cats_sha,
            report_one[
                "component_sha256"
            ]["box_cats.npy"],
        )

    def test_invalid_relation_endpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                categories=[
                    "person",
                    "horse",
                ],
                image_box_categories=[
                    [0, 1]
                ],
                image_relations=[
                    [(0, 2)]
                ],
            )
            with self.assertRaisesRegex(
                ValueError,
                "invalid relation endpoints",
            ):
                rebuild_pair_opportunity(
                    pack,
                    (
                        "person",
                        "horse",
                    ),
                )


def hashlib_sha256(
    payload: bytes,
) -> str:
    import hashlib

    return hashlib.sha256(
        payload
    ).hexdigest()


if __name__ == "__main__":
    unittest.main()
