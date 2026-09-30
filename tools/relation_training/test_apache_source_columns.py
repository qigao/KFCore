from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from apache_objective import (
    load_source_column_allow,
)
from apache_source_columns import (
    DERIVATION_SCHEMA,
    RELEASED_RESTRICTED_SOURCES,
    derive_source_column_allow,
    sha256_file,
    write_source_column_allow,
)


def write_pack(
    root: Path,
    source: str,
    predicates: list[str],
) -> Path:
    split = root / source / "train"
    split.mkdir(
        parents=True,
        exist_ok=True,
    )
    (split / "meta.json").write_text(
        json.dumps(
            {
                "predicates": predicates,
            }
        ),
        encoding="utf-8",
    )
    return split


class ApacheSourceColumnDerivationTest(
    unittest.TestCase
):
    def make_packs(
        self,
        root: Path,
        *,
        hico_predicates: list[str] | None = None,
    ) -> list[Path]:
        return [
            write_pack(
                root,
                "megasg_clean",
                [
                    "holding",
                    "on",
                ],
            ),
            write_pack(
                root,
                "vg_raw",
                [
                    "left",
                    "on",
                ],
            ),
            write_pack(
                root,
                "hicodet",
                (
                    hico_predicates
                    if hico_predicates is not None
                    else [
                        "riding",
                        "unknown-local",
                        "holding",
                    ]
                ),
            ),
        ]

    def test_released_rows_match_upstream_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packs = self.make_packs(
                root
            )
            union = (
                "on",
                "holding",
                "left",
                "riding",
            )
            payload, evidence = (
                derive_source_column_allow(
                    packs,
                    union,
                )
            )

        self.assertEqual(
            [
                source["name"]
                for source in payload[
                    "sources"
                ]
            ],
            [
                "megasg_clean",
                "vg_raw",
                "hicodet",
            ],
        )
        self.assertEqual(
            payload["sources"][0][
                "predicates"
            ],
            list(union),
        )
        self.assertEqual(
            payload["sources"][1][
                "predicates"
            ],
            list(union),
        )
        self.assertEqual(
            payload["sources"][2][
                "predicates"
            ],
            [
                "holding",
                "riding",
            ],
        )
        self.assertEqual(
            evidence["schema"],
            DERIVATION_SCHEMA,
        )
        self.assertEqual(
            evidence[
                "restricted_sources"
            ],
            list(
                RELEASED_RESTRICTED_SOURCES
            ),
        )
        self.assertEqual(
            evidence["sources"][2][
                "unknown_local_predicates"
            ],
            ["unknown-local"],
        )
        self.assertTrue(
            evidence["sources"][2][
                "restricted"
            ]
        )
        self.assertFalse(
            evidence["sources"][0][
                "restricted"
            ]
        )

    def test_hico_local_order_does_not_change_union_columns(self):
        with tempfile.TemporaryDirectory() as one_dir, tempfile.TemporaryDirectory() as two_dir:
            union = (
                "on",
                "holding",
                "left",
                "riding",
            )
            first, _ = (
                derive_source_column_allow(
                    self.make_packs(
                        Path(one_dir),
                        hico_predicates=[
                            "riding",
                            "holding",
                        ],
                    ),
                    union,
                )
            )
            second, _ = (
                derive_source_column_allow(
                    self.make_packs(
                        Path(two_dir),
                        hico_predicates=[
                            "holding",
                            "riding",
                        ],
                    ),
                    union,
                )
            )

        self.assertEqual(
            first,
            second,
        )
        self.assertEqual(
            first["sources"][2][
                "predicates"
            ],
            [
                "holding",
                "riding",
            ],
        )

    def test_written_sidecar_is_deterministic_and_runtime_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packs = self.make_packs(
                root
            )
            union = (
                "on",
                "holding",
                "left",
                "riding",
            )
            one = root / "one.json"
            two = root / "two.json"
            one_evidence = (
                root / "one.evidence.json"
            )
            two_evidence = (
                root / "two.evidence.json"
            )
            report_one = (
                write_source_column_allow(
                    pack_splits=packs,
                    predicates=union,
                    output=one,
                    evidence_output=(
                        one_evidence
                    ),
                )
            )
            report_two = (
                write_source_column_allow(
                    pack_splits=packs,
                    predicates=union,
                    output=two,
                    evidence_output=(
                        two_evidence
                    ),
                )
            )
            names, allow = (
                load_source_column_allow(
                    one,
                    union,
                )
            )
            one_bytes = (
                one.read_bytes()
            )
            two_bytes = (
                two.read_bytes()
            )

        self.assertEqual(
            one_bytes,
            two_bytes,
        )
        self.assertEqual(
            report_one[
                "sidecar_sha256"
            ],
            report_two[
                "sidecar_sha256"
            ],
        )
        self.assertEqual(
            names,
            (
                "megasg_clean",
                "vg_raw",
                "hicodet",
            ),
        )
        self.assertEqual(
            allow.tolist(),
            [
                [
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    False,
                    True,
                    False,
                    True,
                ],
            ],
        )

    def test_meta_hashes_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packs = self.make_packs(
                root
            )
            _, evidence = (
                derive_source_column_allow(
                    packs,
                    (
                        "on",
                        "holding",
                        "left",
                        "riding",
                    ),
                )
            )
            expected = [
                sha256_file(
                    split
                    / "meta.json"
                )
                for split in packs
            ]

        self.assertEqual(
            [
                source[
                    "meta_sha256"
                ]
                for source in evidence[
                    "sources"
                ]
            ],
            expected,
        )

    def test_source_order_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packs = self.make_packs(
                root
            )
            with self.assertRaisesRegex(
                ValueError,
                "source order",
            ):
                derive_source_column_allow(
                    [
                        packs[1],
                        packs[0],
                        packs[2],
                    ],
                    (
                        "on",
                        "holding",
                        "left",
                        "riding",
                    ),
                )

    def test_restricted_source_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packs = self.make_packs(
                root
            )
            with self.assertRaisesRegex(
                ValueError,
                "restricted_sources",
            ):
                derive_source_column_allow(
                    packs,
                    (
                        "on",
                        "holding",
                        "left",
                        "riding",
                    ),
                    restricted_sources=(
                        "vg_raw",
                    ),
                )

    def test_duplicate_local_predicate_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packs = self.make_packs(
                root,
                hico_predicates=[
                    "holding",
                    "holding",
                ],
            )
            with self.assertRaisesRegex(
                ValueError,
                "unique",
            ):
                derive_source_column_allow(
                    packs,
                    (
                        "on",
                        "holding",
                    ),
                )

    def test_restricted_source_must_intersect_union(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packs = self.make_packs(
                root,
                hico_predicates=[
                    "unknown-only",
                ],
            )
            with self.assertRaisesRegex(
                ValueError,
                "no predicates in the union",
            ):
                derive_source_column_allow(
                    packs,
                    (
                        "on",
                        "holding",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
