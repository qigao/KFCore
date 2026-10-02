from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from prepare_hico_test_qualification import (
    DATASET_REVISION,
    _decode_row,
    materialize,
)


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), color=(128, 128, 128)).save(
        buffer,
        format="JPEG",
        quality=95,
    )
    return buffer.getvalue()


def actions() -> list[dict[str, str]]:
    rows = [
        {
            "nname": "bicycle",
            "vname": "ride",
            "vname_ing": "riding",
        },
        {
            "nname": "bicycle",
            "vname": "no_interaction",
            "vname_ing": "no_interaction",
        },
    ]
    rows.extend(
        {
            "nname": f"object-{index}",
            "vname": "no_interaction",
            "vname_ing": "no_interaction",
        }
        for index in range(2, 600)
    )
    return rows


def row() -> dict[str, object]:
    return {
        "image": {
            "bytes": jpeg_bytes(),
            "path": "HICO_test2015_00000001.jpg",
        },
        "objects": [
            {
                "id": 1,
                "bbox_human": [10, 40, 10, 70],
                "bbox_object": [45, 90, 30, 75],
                "invis": 0,
            },
            {
                "id": 1,
                "bbox_human": [10, 40, 10, 70],
                "bbox_object": [45, 90, 30, 75],
                "invis": 0,
            },
        ],
    }


class HicoTestQualificationTest(unittest.TestCase):
    def test_decode_row_keeps_released_predicate_and_merges_duplicates(self):
        decoded = _decode_row(
            row(),
            actions(),
            {"riding"},
        )
        self.assertIsNotNone(decoded)
        assert decoded is not None
        width, height, _, boxes, labels, relations = decoded
        self.assertEqual((width, height), (100, 80))
        self.assertEqual(len(boxes), 2)
        self.assertEqual(labels, ["person", "bicycle"])
        self.assertEqual(relations, [(0, "riding", 1)])

    def test_decode_row_rejects_predicates_outside_bank(self):
        self.assertIsNone(
            _decode_row(
                row(),
                actions(),
                {"holding"},
            )
        )

    def test_materialize_is_deterministic_and_records_source_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            action_path = root / "list_action.csv"
            action_path.write_text(
                "nname,vname,vname_ing\n"
                + "\n".join(
                    f"{value['nname']},{value['vname']},{value['vname_ing']}"
                    for value in actions()
                )
                + "\n",
                encoding="utf-8",
            )
            bank_path = root / "bank.npz"
            np.savez(
                bank_path,
                names=np.asarray(["riding"], dtype=object),
            )
            digest = hashlib.sha256(bank_path.read_bytes()).hexdigest()

            with patch(
                "prepare_hico_test_qualification.PREDICATE_BANK_SHA256",
                digest,
            ):
                report = materialize(
                    actions_path=action_path,
                    predicate_bank_path=bank_path,
                    rows=[row()],
                    output_dir=root / "out",
                    max_images=1,
                    dataset_revision=DATASET_REVISION,
                )

            self.assertEqual(report["selection"]["images"], 1)
            self.assertEqual(report["selection"]["relations"], 1)
            self.assertEqual(report["selection"]["predicates"], ["riding"])
            self.assertEqual(
                report["selection"]["source_rows"][0]["stream_row_index"],
                0,
            )
            self.assertTrue((root / "out" / "test.jsonl").is_file())
            self.assertTrue((root / "out" / "vocabulary.json").is_file())
            self.assertTrue((root / "out" / "images" / "hico_test_000000.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
