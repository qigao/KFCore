from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from apache_object_bank import (
    OBJECT_BANK_SCHEMA,
    load_object_text_bank,
    sha256_file,
)


class ApacheObjectBankTest(unittest.TestCase):
    def write_bank(
        self,
        root: Path,
        *,
        names: list[str] | None = None,
        shape: tuple[int, int] = (2, 512),
        finite: bool = True,
    ) -> Path:
        path = root / "obj_embeds.npz"
        names = (
            names
            if names is not None
            else ["person", "horse"]
        )
        values = np.arange(
            shape[0] * shape[1],
            dtype=np.float32,
        ).reshape(shape)
        values = values / max(
            float(values.max()),
            1.0,
        )
        if not finite:
            values[0, 0] = np.nan
        np.savez_compressed(
            path,
            names=np.asarray(
                names,
                dtype=np.str_,
            ),
            embeddings=values.astype(
                np.float16
            ),
        )
        return path

    def test_exact_named_bank_loads_and_reports_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_bank(root)
            tensor, report = load_object_text_bank(
                path,
                ("person", "horse"),
            )

        self.assertEqual(
            tuple(tensor.shape),
            (2, 512),
        )
        self.assertEqual(
            report["schema"],
            OBJECT_BANK_SCHEMA,
        )
        self.assertEqual(
            report["artifact_sha256"],
            sha256_file(path),
        )
        self.assertEqual(
            report["shape"],
            [2, 512],
        )
        self.assertEqual(
            report["object_label_order"],
            ["person", "horse"],
        )
        self.assertEqual(
            report["text_dim"],
            512,
        )
        self.assertEqual(
            report["source_dtype"],
            "float16",
        )

    def test_name_order_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_bank(
                root,
                names=["horse", "person"],
            )
            with self.assertRaisesRegex(
                ValueError,
                "names/order",
            ):
                load_object_text_bank(
                    path,
                    ("person", "horse"),
                )

    def test_non_512_width_fails_reference_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_bank(
                root,
                shape=(2, 256),
            )
            with self.assertRaisesRegex(
                ValueError,
                r"\[2,512\]",
            ):
                load_object_text_bank(
                    path,
                    ("person", "horse"),
                )

    def test_row_count_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_bank(
                root,
                names=[
                    "person",
                    "horse",
                    "dog",
                ],
                shape=(3, 512),
            )
            with self.assertRaisesRegex(
                ValueError,
                "names/order",
            ):
                load_object_text_bank(
                    path,
                    ("person", "horse"),
                )

    def test_non_finite_embeddings_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_bank(
                root,
                finite=False,
            )
            with self.assertRaisesRegex(
                ValueError,
                "finite",
            ):
                load_object_text_bank(
                    path,
                    ("person", "horse"),
                )

    def test_missing_named_arrays_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "obj_embeds.npz"
            np.savez_compressed(
                path,
                wrong=np.zeros(
                    (2, 512),
                    dtype=np.float16,
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "names and embeddings",
            ):
                load_object_text_bank(
                    path,
                    ("person", "horse"),
                )


if __name__ == "__main__":
    unittest.main()
