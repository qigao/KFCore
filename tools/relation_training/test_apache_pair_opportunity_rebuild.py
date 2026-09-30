from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from apache_pair_opportunity_rebuild import (
    DERIVATION_SCHEMA,
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
    image_categories: list[
        list[int]
    ],
    image_relations: list[
        list[tuple[int, int]]
    ],
) -> Path:
    split = (
        root
        / "megasg_clean"
        / "train"
    )
    split.mkdir(
        parents=True
    )
    (split / "meta.json").write_text(
        json.dumps(
            {
                "categories": (
                    categories
                ),
            }
        ),
        encoding="utf-8",
    )

    img_meta: list[
        tuple[int, int, int, int, int, int, int]
    ] = []
    cats: list[int] = []
    rels: list[
        tuple[int, int, int, int, int]
    ] = []
    for index, (
        local_categories,
        local_relations,
    ) in enumerate(
        zip(
            image_categories,
            image_relations,
        )
    ):
        b0 = len(cats)
        r0 = len(rels)
        cats.extend(
            local_categories
        )
        rels.extend(
            (
                subject,
                object_,
                0,
                0,
                -1,
            )
            for subject, object_
            in local_relations
        )
        img_meta.append(
            (
                index,
                10,
                10,
                b0,
                len(
                    local_categories
                ),
                r0,
                len(
                    local_relations
                ),
            )
        )

    np.save(
        split / "img_meta.npy",
        np.asarray(
            img_meta,
            dtype=np.int64,
        ),
    )
    np.save(
        split / "box_cats.npy",
        np.asarray(
            cats,
            dtype=np.int32,
        ),
    )
    np.save(
        split / "rels.npy",
        np.asarray(
            rels,
            dtype=np.int32,
        ).reshape(-1, 5),
    )
    return split


class ApachePairOpportunityRebuildTest(
    unittest.TestCase
):
    def test_ordered_instance_pair_math_and_diagonal_self_subtraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                categories=[
                    "person",
                    "shirt",
                    "horse",
                ],
                image_categories=[
                    [0, 0, 1],
                    [1, 2],
                ],
                image_relations=[
                    [
                        (0, 2),
                        # Only one B -> A edge:
                        (2, 0),
                    ],
                    [
                        # shirt -> horse only.
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

        opportunities = arrays[
            "opportunities"
        ].reshape(3, 3)
        relations = arrays[
            "relations"
        ].reshape(3, 3)
        rate = arrays[
            "rate"
        ].reshape(3, 3)

        # Two person instances produce 2 ordered non-self person->person pairs.
        self.assertEqual(
            int(
                opportunities[
                    0,
                    0,
                ]
            ),
            2,
        )
        # person->shirt and shirt->person each have 2 instance opportunities.
        self.assertEqual(
            int(
                opportunities[
                    0,
                    1,
                ]
            ),
            2,
        )
        self.assertEqual(
            int(
                opportunities[
                    1,
                    0,
                ]
            ),
            2,
        )
        self.assertEqual(
            int(
                relations[
                    0,
                    1,
                ]
            ),
            1,
        )
        self.assertEqual(
            int(
                relations[
                    1,
                    0,
                ]
            ),
            1,
        )
        self.assertAlmostEqual(
            float(
                rate[
                    0,
                    1,
                ]
            ),
            0.5,
            places=7,
        )
        self.assertAlmostEqual(
            float(
                rate[
                    1,
                    0,
                ]
            ),
            0.5,
            places=7,
        )
        # Second image has exactly one shirt->horse opportunity and relation.
        self.assertEqual(
            int(
                opportunities[
                    1,
                    2,
                ]
            ),
            1,
        )
        self.assertEqual(
            float(
                rate[
                    1,
                    2,
                ]
            ),
            1.0,
        )
        # No reverse relation: ordered directions are independent.
        self.assertEqual(
            int(
                relations[
                    2,
                    1,
                ]
            ),
            0,
        )
        self.assertEqual(
            float(
                rate[
                    2,
                    1,
                ]
            ),
            0.0,
        )
        self.assertEqual(
            evidence[
                "scan_box_cap"
            ],
            RELEASED_SCAN_BOX_CAP,
        )
        self.assertEqual(
            evidence[
                "min_support"
            ],
            RELEASED_MIN_SUPPORT,
        )

    def test_relations_beyond_400_box_cap_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            categories = [
                0
                for _ in range(
                    401
                )
            ]
            categories[399] = 1
            categories[400] = 1
            pack = write_pack(
                root,
                categories=[
                    "person",
                    "horse",
                ],
                image_categories=[
                    categories
                ],
                image_relations=[
                    [
                        # Kept: both endpoints < 400.
                        (0, 399),
                        # Dropped by released scan cap.
                        (0, 400),
                    ]
                ],
            )
            arrays, _ = (
                rebuild_pair_opportunity(
                    pack,
                    (
                        "person",
                        "horse",
                    ),
                )
            )

        relations = arrays[
            "relations"
        ].reshape(2, 2)
        self.assertEqual(
            int(
                relations[
                    0,
                    1,
                ]
            ),
            1,
        )

    def test_category_order_must_match_object_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                categories=[
                    "horse",
                    "person",
                ],
                image_categories=[
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

    def test_released_constants_cannot_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                categories=[
                    "person",
                    "horse",
                ],
                image_categories=[
                    [0, 1]
                ],
                image_relations=[
                    []
                ],
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

    def test_written_npz_is_runtime_valid_and_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                categories=[
                    "person",
                    "horse",
                ],
                image_categories=[
                    [0, 1]
                ],
                image_relations=[
                    [(0, 1)]
                ],
            )
            one = root / "one.npz"
            one_evidence = (
                root / "one.json"
            )
            two = root / "two.npz"
            two_evidence = (
                root / "two.json"
            )
            first = write_pair_opportunity(
                pack_split=pack,
                object_labels=(
                    "person",
                    "horse",
                ),
                output=one,
                evidence_output=one_evidence,
            )
            second = write_pair_opportunity(
                pack_split=pack,
                object_labels=(
                    "person",
                    "horse",
                ),
                output=two,
                evidence_output=two_evidence,
            )
            loaded = (
                PairOpportunityTable.load(
                    one
                )
            )

            self.assertEqual(
                first["schema"],
                DERIVATION_SCHEMA,
            )
            self.assertEqual(
                first[
                    "output_sha256"
                ],
                sha256_file(one),
            )
            self.assertEqual(
                second[
                    "output_sha256"
                ],
                sha256_file(two),
            )
            self.assertEqual(
                one.read_bytes(),
                two.read_bytes(),
            )
            self.assertEqual(
                loaded.num_cats,
                2,
            )
            self.assertEqual(
                loaded.min_support,
                50,
            )
            self.assertEqual(
                first["pack"][
                    "object_label_order"
                ],
                [
                    "person",
                    "horse",
                ],
            )

    def test_invalid_pack_ranges_and_categories_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                categories=[
                    "person",
                    "horse",
                ],
                image_categories=[
                    [0, 2]
                ],
                image_relations=[
                    []
                ],
            )
            with self.assertRaisesRegex(
                ValueError,
                "category id",
            ):
                rebuild_pair_opportunity(
                    pack,
                    (
                        "person",
                        "horse",
                    ),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                categories=[
                    "person",
                    "horse",
                ],
                image_categories=[
                    [0, 1]
                ],
                image_relations=[
                    [(0, 1)]
                ],
            )
            image_meta = np.load(
                pack / "img_meta.npy"
            )
            image_meta[
                0,
                3,
            ] = 999
            np.save(
                pack / "img_meta.npy",
                image_meta,
            )
            with self.assertRaisesRegex(
                ValueError,
                "invalid box range",
            ):
                rebuild_pair_opportunity(
                    pack,
                    (
                        "person",
                        "horse",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
