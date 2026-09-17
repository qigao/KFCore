from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from prepare_dataset import (
    DEFAULT_DATASET_ROOT,
    GESTURE_LABEL_CONTRACT,
    MANIFEST_SCHEMA,
    main,
    select_jester_samples,
    write_manifest,
)


class JesterManifestTests(unittest.TestCase):
    @staticmethod
    def _write_frame(frames_root: Path, sample_id: str) -> None:
        sample = frames_root / sample_id
        sample.mkdir(parents=True, exist_ok=True)
        (sample / "00001.jpg").write_bytes(b"frame")

    def test_selects_only_compatible_jester_labels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frames_root = root / "frames"
            labels_csv = root / "train.csv"
            labels_csv.write_text(
                "40;Doing other things\n"
                "10;Swiping Left\n"
                "30;Thumb Up\n"
                "20;No gesture\n"
                "50;Swiping Right\n",
                encoding="utf-8",
            )
            for sample_id in ("10", "20", "40", "50"):
                self._write_frame(frames_root, sample_id)

            result = select_jester_samples(labels_csv, frames_root, "train")

            self.assertEqual(
                [sample.source_sample_id for sample in result.samples],
                ["10", "20", "40", "50"],
            )
            self.assertEqual(
                [sample.target_gesture_label for sample in result.samples],
                [1, 0, 0, 2],
            )
            self.assertEqual(
                [sample.target_gesture_name for sample in result.samples],
                ["swipe_left", "none", "none", "swipe_right"],
            )
            self.assertEqual(
                [sample.role for sample in result.samples],
                ["positive", "hard_negative", "hard_negative", "positive"],
            )
            self.assertTrue(all(sample.subject_id is None for sample in result.samples))
            self.assertTrue(
                all(sample.usage == "pretrain_only" for sample in result.samples)
            )
            self.assertTrue(
                all(sample.source_path.is_absolute() for sample in result.samples)
            )
            self.assertEqual([sample.frame_count for sample in result.samples], [1] * 4)
            self.assertEqual(result.skipped_by_label, {"Thumb Up": 1})

    def test_rejects_duplicate_source_sample_id(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frames_root = root / "frames"
            labels_csv = root / "train.csv"
            labels_csv.write_text(
                "10;Swiping Left\n10;Swiping Right\n", encoding="utf-8"
            )
            self._write_frame(frames_root, "10")

            with self.assertRaisesRegex(ValueError, "duplicate source sample id '10'"):
                select_jester_samples(labels_csv, frames_root, "train")

    def test_rejects_path_like_source_sample_id(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frames_root = root / "frames"
            frames_root.mkdir()
            labels_csv = root / "train.csv"
            labels_csv.write_text("../escape;Swiping Left\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid source sample id"):
                select_jester_samples(labels_csv, frames_root, "train")

    def test_rejects_selected_sample_without_image_frames(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frames_root = root / "frames"
            (frames_root / "10").mkdir(parents=True)
            labels_csv = root / "train.csv"
            labels_csv.write_text("10;Swiping Left\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "contains no supported image frames"):
                select_jester_samples(labels_csv, frames_root, "train")

    def test_rejects_empty_source_split(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frames_root = root / "frames"
            labels_csv = root / "train.csv"
            labels_csv.write_text("10;Swiping Left\n", encoding="utf-8")
            self._write_frame(frames_root, "10")

            with self.assertRaisesRegex(ValueError, "source_split must not be empty"):
                select_jester_samples(labels_csv, frames_root, "  ")

    def test_write_manifest_preserves_contract_and_refuses_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frames_root = root / "frames"
            labels_csv = root / "train.csv"
            output = root / "manifests" / "jester-train.jsonl"
            labels_csv.write_text("10;Swiping Left\n", encoding="utf-8")
            self._write_frame(frames_root, "10")
            result = select_jester_samples(labels_csv, frames_root, "train")

            write_manifest(result, output)

            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["manifest_schema"], MANIFEST_SCHEMA)
            self.assertEqual(
                records[0]["gesture_label_contract"], GESTURE_LABEL_CONTRACT
            )
            self.assertEqual(records[0]["subject_id"], None)
            self.assertEqual(records[0]["usage"], "pretrain_only")
            self.assertNotIn("landmarks", records[0])
            self.assertNotIn("phase_label", records[0])
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_manifest(result, output)

    def test_cli_uses_dataset_root_manifest_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frames_root = root / "frames"
            labels_csv = root / "train.csv"
            dataset_root = root / "dataset"
            labels_csv.write_text(
                "10;Swiping Left\n20;Thumb Up\n", encoding="utf-8"
            )
            self._write_frame(frames_root, "10")
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--labels",
                        str(labels_csv),
                        "--frames-root",
                        str(frames_root),
                        "--source-split",
                        "validation",
                        "--dataset-root",
                        str(dataset_root),
                    ]
                )

            output = dataset_root / "manifests" / "jester-validation.jsonl"
            summary = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(output.is_file())
            self.assertEqual(summary["selected_samples"], 1)
            self.assertEqual(summary["skipped_by_label"], {"Thumb Up": 1})
            self.assertEqual(Path(summary["output"]), output.resolve())

    def test_default_dataset_root_is_on_requested_f_drive(self) -> None:
        self.assertEqual(
            DEFAULT_DATASET_ROOT,
            Path(r"F:\KFCoreDatasets\temporal_gesture"),
        )


if __name__ == "__main__":
    unittest.main()
