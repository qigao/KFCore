from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from apache_objective import (
    load_source_column_allow,
)
from apache_source_allow import (
    DERIVATION_SCHEMA,
    RELEASED_RESTRICTED_SOURCES,
    derive_source_allow,
    sha256_file,
    write_source_allow,
)


def write_pack(
    root: Path,
    name: str,
    predicates: list[str],
) -> Path:
    split = root / name / "train"
    split.mkdir(
        parents=True
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


class ApacheSourceAllowTest(
    unittest.TestCase
):
    def test_released_hico_row_is_local_union_intersection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packs = [
                write_pack(
                    root,
                    "megasg_clean",
                    ["holding", "on"],
                ),
                write_pack(
                    root,
                    "vg_raw",
                    ["left", "on"],
                ),
                write_pack(
                    root,
                    "hicodet",
                    [
                        "riding",
                        "holding",
                        "local-only",
                    ],
                ),
            ]
            payload, evidence = (
                derive_source_allow(
                    packs,
                    (
                        "on",
                        "holding",
                        "riding",
                        "left",
                    ),
                )
            )

        self.assertEqual(
            RELEASED_RESTRICTED_SOURCES,
            ("hicodet",),
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
            [
                "on",
                "holding",
                "riding",
                "left",
            ],
        )
        self.assertEqual(
            payload["sources"][1][
                "predicates"
            ],
            [
                "on",
                "holding",
                "riding",
                "left",
            ],
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
            ["hicodet"],
        )
        self.assertEqual(
            evidence["sources"][2][
                "ignored_predicates"
            ],
            ["local-only"],
        )
        self.assertTrue(
            evidence["sources"][2][
                "restricted"
            ]
        )

    def test_local_predicate_order_does_not_change_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = [
                write_pack(
                    root / "one",
                    "megasg_clean",
                    ["on"],
                ),
                write_pack(
                    root / "one",
                    "vg_raw",
                    ["left"],
                ),
                write_pack(
                    root / "one",
                    "hicodet",
                    [
                        "riding",
                        "holding",
                    ],
                ),
            ]
            second = [
                write_pack(
                    root / "two",
                    "megasg_clean",
                    ["on"],
                ),
                write_pack(
                    root / "two",
                    "vg_raw",
                    ["left"],
                ),
                write_pack(
                    root / "two",
                    "hicodet",
                    [
                        "holding",
                        "riding",
                    ],
                ),
            ]
            payload_one, _ = derive_source_allow(
                first,
                (
                    "holding",
                    "riding",
                    "on",
                ),
            )
            payload_two, _ = derive_source_allow(
                second,
                (
                    "holding",
                    "riding",
                    "on",
                ),
            )

        self.assertEqual(
            payload_one,
            payload_two,
        )

    def test_written_sidecar_loads_with_existing_runtime_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packs = [
                write_pack(
                    root,
                    "megasg_clean",
                    ["on"],
                ),
                write_pack(
                    root,
                    "vg_raw",
                    ["left"],
                ),
                write_pack(
                    root,
                    "hicodet",
                    ["holding"],
                ),
            ]
            out = (
                root
                / "source_allow.json"
            )
            evidence_path = (
                root
                / "source_allow.evidence.json"
            )
            evidence = write_source_allow(
                pack_splits=packs,
                predicates=(
                    "on",
                    "holding",
                    "left",
                ),
                output=out,
                evidence_output=evidence_path,
            )
            names, allow = (
                load_source_column_allow(
                    out,
                    (
                        "on",
                        "holding",
                        "left",
                    ),
                )
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
                    [True, True, True],
                    [True, True, True],
                    [False, True, False],
                ],
            )
            self.assertEqual(
                evidence[
                    "sidecar_sha256"
                ],
                sha256_file(out),
            )
            for pack, report in zip(
                packs,
                evidence["sources"],
            ):
                self.assertEqual(
                    report[
                        "meta_sha256"
                    ],
                    sha256_file(
                        pack
                        / "meta.json"
                    ),
                )

    def test_missing_restricted_hico_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packs = [
                write_pack(
                    root,
                    "megasg_clean",
                    ["on"],
                ),
                write_pack(
                    root,
                    "vg_raw",
                    ["left"],
                ),
            ]
            with self.assertRaisesRegex(
                ValueError,
                "missing restricted source",
            ):
                derive_source_allow(
                    packs,
                    ("on", "left"),
                )

    def test_restricted_source_with_no_union_support_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packs = [
                write_pack(
                    root,
                    "megasg_clean",
                    ["on"],
                ),
                write_pack(
                    root,
                    "vg_raw",
                    ["left"],
                ),
                write_pack(
                    root,
                    "hicodet",
                    ["local-only"],
                ),
            ]
            with self.assertRaisesRegex(
                ValueError,
                "no predicates in union",
            ):
                derive_source_allow(
                    packs,
                    ("on", "left"),
                )

    def test_duplicate_source_pack_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                "hicodet",
                ["holding"],
            )
            with self.assertRaisesRegex(
                ValueError,
                "duplicate source pack",
            ):
                derive_source_allow(
                    [pack, pack],
                    ("holding",),
                )

    def test_malformed_meta_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "hicodet" / "train"
            pack.mkdir(
                parents=True
            )
            (pack / "meta.json").write_text(
                json.dumps(
                    {
                        "predicates": [
                            "holding",
                            "holding",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "unique",
            ):
                derive_source_allow(
                    [pack],
                    ("holding",),
                )


if __name__ == "__main__":
    unittest.main()
