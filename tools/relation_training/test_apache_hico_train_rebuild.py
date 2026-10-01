from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from apache_hico_train_rebuild import (
    RELEASED_IOU_MERGE,
    _expected_input_contract,
    convert_hico_rows,
    merge_boxes,
)


def make_actions() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.append({"nname": "bike", "vname": "ride", "vname_ing": "riding"})
    rows.append({"nname": "bike", "vname": "hold", "vname_ing": "holding"})
    rows.append({
        "nname": "bike",
        "vname": "no_interaction",
        "vname_ing": "no_interaction",
    })
    rows.append({"nname": "ball", "vname": "throw", "vname_ing": "throwing"})
    rows.append({"nname": "person", "vname": "greet", "vname_ing": "greeting"})
    for index in range(5, 600):
        rows.append(
            {
                "nname": f"object_{index}",
                "vname": f"verb_{index}",
                "vname_ing": f"verb_{index}_ing",
            }
        )
    return rows


def image_payload(width: int, height: int, path: str) -> dict[str, object]:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(
        buffer,
        format="PNG",
    )
    return {"bytes": buffer.getvalue(), "path": path}


def hico_object(
    category_id: int,
    human_box: list[float],
    object_box: list[float],
    *,
    invis: bool = False,
) -> dict[str, object]:
    return {
        "id": category_id,
        "bbox_human": human_box,
        "bbox_object": object_box,
        "invis": invis,
    }


class ApacheHicoTrainRebuildTest(unittest.TestCase):
    def test_same_category_iou_merge_uses_mean_and_remaps(self) -> None:
        boxes = [
            ([0.0, 0.0, 5.0, 5.0], "bike"),
            ([1.0, 0.0, 5.0, 5.0], "bike"),
            ([20.0, 0.0, 5.0, 5.0], "bike"),
        ]
        merged, remap = merge_boxes(boxes, RELEASED_IOU_MERGE)
        self.assertEqual(remap, [0, 0, 1])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0][0], [0.5, 0.0, 5.0, 5.0])
        self.assertEqual(merged[0][1], "bike")

    def test_converter_pins_merge_dedup_selfloop_and_negative_image(self) -> None:
        actions = make_actions()
        rows = [
            {
                "image": image_payload(40, 30, "hico_train_000001.jpg"),
                "objects": [
                    hico_object(1, [0, 5, 0, 10], [10, 15, 0, 5]),
                    hico_object(1, [0, 5, 0, 10], [11, 16, 0, 5]),
                    hico_object(2, [0, 5, 0, 10], [10, 15, 0, 5]),
                    # Synthetic person object sharing the exact human box:
                    # add_box reuses the coordinate key and the positive is a self-loop.
                    hico_object(5, [0, 5, 0, 10], [0, 5, 0, 10]),
                    hico_object(4, [0, 5, 0, 10], [25, 29, 10, 14], invis=True),
                ],
                "positive_captions": [],
                "negative_captions": [],
                "ambiguous_captions": [],
            },
            {
                "image": image_payload(40, 30, "hico_train_000002.jpg"),
                "objects": [
                    hico_object(3, [0, 5, 0, 10], [10, 15, 0, 5]),
                ],
                # Parse string columns exactly like the parquet converter.
                "positive_captions": "[]",
                "negative_captions": "[]",
                "ambiguous_captions": "[]",
            },
        ]

        coco, negatives, evidence = convert_hico_rows(
            actions=actions,
            rows=rows,
        )

        self.assertEqual(len(coco["images"]), 2)
        self.assertEqual(coco["images"][0]["file_name"], "hico_train_000001.jpg")
        self.assertEqual(coco["images"][1]["file_name"], "hico_train_000002.jpg")

        # Two near-identical bike boxes merge; duplicate riding positives collapse.
        self.assertEqual(evidence["counts"]["positive_duplicates_dropped"], 1)
        self.assertEqual(evidence["counts"]["self_loops_dropped"], 1)
        self.assertEqual(evidence["counts"]["invalid_or_invisible_objects"], 1)

        predicates = [row["name"] for row in coco["rel_categories"]]
        self.assertEqual(predicates[:2], ["riding", "holding"])
        self.assertEqual(len(coco["rel_annotations"]), 2)

        # The no_interaction-only second image remains in COCO because it carries
        # explicit negative cells, but contributes no positive relation rows.
        self.assertIn("2", negatives["by_image_id"])
        self.assertGreater(len(negatives["by_image_id"]["2"]), 0)

    def test_first_40_box_window_drops_positive_outside_window(self) -> None:
        actions = make_actions()
        objects = []
        # One shared human plus forty non-overlapping bike instances => 41 boxes.
        for index in range(40):
            objects.append(
                hico_object(
                    1,
                    [0, 5, 0, 10],
                    [10 + index * 3, 12 + index * 3, 15, 17],
                )
            )
        row = {
            "image": image_payload(200, 40, "hico_train_000003.jpg"),
            "objects": objects,
            "positive_captions": [],
            "negative_captions": [],
            "ambiguous_captions": [],
        }
        coco, _, evidence = convert_hico_rows(actions=actions, rows=[row])

        self.assertEqual(len(coco["annotations"]), 40)
        self.assertEqual(evidence["counts"]["relations_outside_40_boxes"], 1)
        self.assertEqual(len(coco["rel_annotations"]), 39)

    def test_negative_only_predicate_fails_deterministic_rebuild(self) -> None:
        actions = make_actions()
        row = {
            "image": image_payload(40, 30, "hico_train_000004.jpg"),
            "objects": [
                hico_object(3, [0, 5, 0, 10], [10, 15, 0, 5]),
            ],
            "positive_captions": [],
            # Unknown action remains a surface predicate and would be inserted
            # by upstream through a Python set after the scan.
            "negative_captions": [["bike", "never_seen_action"]],
            "ambiguous_captions": [],
        }
        with self.assertRaisesRegex(ValueError, "negative-only predicates"):
            convert_hico_rows(actions=actions, rows=[row])

    def test_input_contract_preserves_sorted_shard_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            action_csv = Path(td) / "list_action.csv"
            action_csv.write_text("x\n", encoding="utf-8")
            reports = [
                {"name": "train-00000.parquet", "sha256": "a" * 64},
                {"name": "train-00001.parquet", "sha256": "b" * 64},
            ]
            contract = _expected_input_contract(
                action_csv=action_csv,
                shard_reports=reports,
            )
            self.assertEqual(
                [row["name"] for row in contract["parquets"]],
                ["train-00000.parquet", "train-00001.parquet"],
            )
            reversed_contract = _expected_input_contract(
                action_csv=action_csv,
                shard_reports=list(reversed(reports)),
            )
            self.assertNotEqual(contract, reversed_contract)


if __name__ == "__main__":
    unittest.main()
