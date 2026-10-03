from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from apache_indoorvg_holdout import (
    DERIVATION_SCHEMA,
    RELEASED_NOTE,
    RELEASED_SOURCE,
    RELEASED_SPLITS,
    derive_indoorvg_holdout,
    sha256_file,
    write_indoorvg_holdout,
)
from apache_mixture import (
    _load_excluded_stems,
)


def write_fixture(
    root: Path,
    *,
    val_names: list[str],
    test_names: list[str],
    mapping: dict[str, object],
) -> tuple[Path, Path]:
    indoor = (
        root
        / "IndoorVG_coco_format"
    )
    for split, names in (
        ("val", val_names),
        ("test", test_names),
    ):
        directory = (
            indoor / split
        )
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        for name in names:
            (
                directory / name
            ).write_text(
                "identity-only",
                encoding="utf-8",
            )
    mapping_path = (
        root / "vg2coco.json"
    )
    mapping_path.write_text(
        json.dumps(mapping),
        encoding="utf-8",
    )
    return indoor, mapping_path


class ApacheIndoorVGHoldoutTest(
    unittest.TestCase
):
    def test_released_union_mapping_and_suffix_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            indoor, mapping = (
                write_fixture(
                    root,
                    val_names=[
                        "100.jpg",
                        "200.jpg",
                        "ignore.txt",
                        "UPPER.JPG",
                    ],
                    test_names=[
                        "200.jpg",
                        "300.jpg",
                        "400.jpg",
                    ],
                    mapping={
                        "100": "7",
                        "200": 42,
                        "300": "000005",
                    },
                )
            )
            payload, evidence = (
                derive_indoorvg_holdout(
                    indoor,
                    mapping,
                )
            )

        self.assertEqual(
            payload["source"],
            RELEASED_SOURCE,
        )
        self.assertEqual(
            payload["splits"],
            list(RELEASED_SPLITS),
        )
        self.assertEqual(
            payload["note"],
            RELEASED_NOTE,
        )
        self.assertEqual(
            payload["vg_ids"],
            [
                "100",
                "200",
                "300",
                "400",
            ],
        )
        self.assertEqual(
            payload["coco_stems"],
            [
                "000000000005",
                "000000000007",
                "000000000042",
            ],
        )
        self.assertEqual(
            payload["stems"],
            sorted(
                {
                    "100",
                    "200",
                    "300",
                    "400",
                    "000000000005",
                    "000000000007",
                    "000000000042",
                }
            ),
        )

        self.assertEqual(
            evidence["schema"],
            DERIVATION_SCHEMA,
        )
        self.assertEqual(
            evidence["mapped_vg_count"],
            3,
        )
        self.assertEqual(
            evidence["unmapped_vg_ids"],
            ["400"],
        )
        self.assertEqual(
            evidence["unmapped_vg_count"],
            1,
        )
        self.assertEqual(
            evidence["split_inputs"][0][
                "vg_ids"
            ],
            ["100", "200"],
        )
        self.assertEqual(
            evidence["split_inputs"][1][
                "vg_ids"
            ],
            [
                "200",
                "300",
                "400",
            ],
        )

    def test_duplicate_coco_aliases_collapse_but_mapped_vg_count_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            indoor, mapping = (
                write_fixture(
                    root,
                    val_names=[
                        "1.jpg",
                    ],
                    test_names=[
                        "2.jpg",
                    ],
                    mapping={
                        "1": 77,
                        "2": "77",
                    },
                )
            )
            payload, evidence = (
                derive_indoorvg_holdout(
                    indoor,
                    mapping,
                )
            )

        self.assertEqual(
            evidence["mapped_vg_count"],
            2,
        )
        self.assertEqual(
            payload["coco_stems"],
            ["000000000077"],
        )
        self.assertEqual(
            evidence["coco_stem_count"],
            1,
        )
        self.assertEqual(
            payload["stems"],
            [
                "000000000077",
                "1",
                "2",
            ],
        )

    def test_output_and_evidence_are_deterministic_and_loader_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            indoor, mapping = (
                write_fixture(
                    root,
                    val_names=[
                        "9.jpg",
                        "3.jpg",
                    ],
                    test_names=[
                        "4.jpg",
                    ],
                    mapping={
                        "3": 1,
                        "4": 2,
                        "9": 3,
                    },
                )
            )
            one = (
                root / "one.json"
            )
            two = (
                root / "two.json"
            )
            one_evidence = (
                root / "one.evidence.json"
            )
            two_evidence = (
                root / "two.evidence.json"
            )
            first = (
                write_indoorvg_holdout(
                    indoorvg_root=indoor,
                    vg2coco_path=mapping,
                    output=one,
                    evidence_output=(
                        one_evidence
                    ),
                )
            )
            second = (
                write_indoorvg_holdout(
                    indoorvg_root=indoor,
                    vg2coco_path=mapping,
                    output=two,
                    evidence_output=(
                        two_evidence
                    ),
                )
            )
            one_bytes = (
                one.read_bytes()
            )
            two_bytes = (
                two.read_bytes()
            )
            one_evidence_bytes = (
                one_evidence.read_bytes()
            )
            two_evidence_bytes = (
                two_evidence.read_bytes()
            )
            loaded = (
                _load_excluded_stems(
                    one
                )
            )
            output_sha = (
                sha256_file(one)
            )

        self.assertEqual(
            one_bytes,
            two_bytes,
        )
        self.assertEqual(
            one_evidence_bytes,
            two_evidence_bytes,
        )
        self.assertEqual(
            first,
            second,
        )
        self.assertEqual(
            first["output_sha256"],
            output_sha,
        )
        self.assertEqual(
            loaded,
            {
                "3",
                "4",
                "9",
                "000000000001",
                "000000000002",
                "000000000003",
            },
        )

    def test_split_order_or_membership_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            indoor, mapping = (
                write_fixture(
                    root,
                    val_names=["1.jpg"],
                    test_names=["2.jpg"],
                    mapping={},
                )
            )
            with self.assertRaisesRegex(
                ValueError,
                r"splits=\['val','test'\]",
            ):
                derive_indoorvg_holdout(
                    indoor,
                    mapping,
                    splits=(
                        "test",
                        "val",
                    ),
                )
            with self.assertRaisesRegex(
                ValueError,
                r"splits=\['val','test'\]",
            ):
                derive_indoorvg_holdout(
                    indoor,
                    mapping,
                    splits=(
                        "val",
                    ),
                )
            with self.assertRaisesRegex(
                ValueError,
                r"splits=\['val','test'\]",
            ):
                derive_indoorvg_holdout(
                    indoor,
                    mapping,
                    splits=(
                        "train",
                        "val",
                        "test",
                    ),
                )

    def test_malformed_mapping_values_fail_closed(self):
        bad_values = [
            True,
            1.5,
            "12.5",
            "abc",
            "",
            -1,
            "-2",
        ]
        for value in bad_values:
            with self.subTest(
                value=value
            ):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(
                        directory
                    )
                    indoor, mapping = (
                        write_fixture(
                            root,
                            val_names=[
                                "1.jpg",
                            ],
                            test_names=[],
                            mapping={
                                "1": value
                            },
                        )
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "must be",
                    ):
                        derive_indoorvg_holdout(
                            indoor,
                            mapping,
                        )

    def test_empty_jpg_stem_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            indoor, mapping = (
                write_fixture(
                    root,
                    val_names=[
                        ".jpg",
                    ],
                    test_names=[],
                    mapping={},
                )
            )
            with self.assertRaisesRegex(
                ValueError,
                "empty .jpg stem",
            ):
                derive_indoorvg_holdout(
                    indoor,
                    mapping,
                )

    def test_mapping_keys_must_be_strings(self):
        # JSON object keys are always strings, so exercise the public
        # derivation's object-shape gate instead.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            indoor, mapping = (
                write_fixture(
                    root,
                    val_names=[
                        "1.jpg",
                    ],
                    test_names=[],
                    mapping={},
                )
            )
            mapping.write_text(
                json.dumps(
                    ["not", "an", "object"]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "must be an object",
            ):
                derive_indoorvg_holdout(
                    indoor,
                    mapping,
                )


if __name__ == "__main__":
    unittest.main()
