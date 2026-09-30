from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from apache_spatial_flags import (
    DERIVATION_SCHEMA,
    derive_spatial_flags,
    sha256_file,
    write_spatial_flags,
)
from apache_vocab_head import (
    load_predicate_spatial_flags,
)


def write_pack(
    root: Path,
    name: str,
    *,
    predicates: list[str],
    rows: list[tuple[int, int]],
) -> Path:
    split = root / name
    split.mkdir()
    (split / "meta.json").write_text(
        json.dumps(
            {
                "predicates": predicates,
            }
        ),
        encoding="utf-8",
    )
    rels = np.asarray(
        [
            [0, 1, predicate_id, flags, -1]
            for predicate_id, flags in rows
        ],
        dtype=np.int32,
    )
    if not rows:
        rels = np.empty(
            (0, 5),
            dtype=np.int32,
        )
    np.save(
        split / "rels.npy",
        rels,
    )
    return split


class ApacheSpatialFlagsTest(
    unittest.TestCase
):
    def test_multi_pack_majority_union_matches_apache_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_pack(
                root,
                "first",
                predicates=[
                    "holding",
                    "on",
                ],
                rows=[
                    # holding: exactly 1/2 spatial => spatial.
                    (0, 1),
                    (0, 0),
                    # on: 0/1 spatial => semantic here.
                    (1, 0),
                ],
            )
            second = write_pack(
                root,
                "second",
                predicates=[
                    "left",
                    "on",
                    "extra-local",
                ],
                rows=[
                    # Local order differs from the union vocabulary.
                    (0, 1),
                    # on becomes spatial in this source, so union is spatial.
                    (1, 1),
                    (1, 0),
                    # Predicate absent from union is ignored, as upstream.
                    (2, 1),
                ],
            )
            payload, evidence = derive_spatial_flags(
                [first, second],
                (
                    "on",
                    "holding",
                    "left",
                    "near",
                ),
            )

        self.assertEqual(
            payload["predicates"],
            [
                "on",
                "holding",
                "left",
                "near",
            ],
        )
        self.assertEqual(
            payload["is_spatial"],
            [
                True,
                True,
                True,
                False,
            ],
        )
        self.assertEqual(
            evidence["schema"],
            DERIVATION_SCHEMA,
        )
        self.assertEqual(
            evidence["majority_threshold"],
            0.5,
        )
        self.assertEqual(
            evidence["majority_comparator"],
            ">=",
        )
        self.assertEqual(
            evidence["unsupported_predicates"],
            ["near"],
        )
        self.assertEqual(
            evidence["sources"][1][
                "ignored_predicates"
            ],
            ["extra-local"],
        )

    def test_written_sidecar_loads_with_runtime_contract_and_hashes_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                "pack",
                predicates=[
                    "holding",
                    "beside",
                ],
                rows=[
                    (0, 1),
                    (1, 0),
                ],
            )
            out = root / "flags.json"
            evidence_path = (
                root / "flags.evidence.json"
            )
            evidence = write_spatial_flags(
                pack_splits=[pack],
                predicates=(
                    "holding",
                    "beside",
                ),
                output=out,
                evidence_output=evidence_path,
            )
            loaded = load_predicate_spatial_flags(
                out,
                (
                    "holding",
                    "beside",
                ),
            )

            self.assertEqual(
                loaded.tolist(),
                [True, False],
            )
            self.assertEqual(
                evidence[
                    "sidecar_sha256"
                ],
                sha256_file(out),
            )
            self.assertEqual(
                evidence["sources"][0][
                    "meta_sha256"
                ],
                sha256_file(
                    pack / "meta.json"
                ),
            )
            self.assertEqual(
                evidence["sources"][0][
                    "rels_sha256"
                ],
                sha256_file(
                    pack / "rels.npy"
                ),
            )

    def test_invalid_predicate_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                "bad",
                predicates=["holding"],
                rows=[(1, 1)],
            )
            with self.assertRaisesRegex(
                ValueError,
                "predicate id is out of range",
            ):
                derive_spatial_flags(
                    [pack],
                    ("holding", "beside"),
                )

    def test_negative_relation_flags_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                "bad-flags",
                predicates=["holding"],
                rows=[(0, -1)],
            )
            with self.assertRaisesRegex(
                ValueError,
                "non-negative bitfields",
            ):
                derive_spatial_flags(
                    [pack],
                    ("holding", "beside"),
                )

    def test_malformed_relation_table_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "bad"
            pack.mkdir()
            (pack / "meta.json").write_text(
                json.dumps(
                    {
                        "predicates": [
                            "holding",
                            "beside",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            np.save(
                pack / "rels.npy",
                np.zeros(
                    (3, 4),
                    dtype=np.int32,
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                r"\[R,5\]",
            ):
                derive_spatial_flags(
                    [pack],
                    ("holding", "beside"),
                )

    def test_empty_support_remains_semantic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(
                root,
                "empty",
                predicates=[
                    "holding",
                    "beside",
                ],
                rows=[],
            )
            payload, evidence = (
                derive_spatial_flags(
                    [pack],
                    (
                        "holding",
                        "beside",
                    ),
                )
            )

        self.assertEqual(
            payload["is_spatial"],
            [False, False],
        )
        self.assertEqual(
            evidence[
                "supported_union_predicate_count"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
