from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from dataset import _label, load_sequences
from evaluate import load_model
from export_onnx import _load_model as load_export_model
from model import (
    FEATURE_COUNT,
    GESTURE_CLASSES,
    GESTURE_COUNT,
    GESTURE_LABEL_CONTRACT,
    HIDDEN_SIZE,
    NUM_LAYERS,
    PHASE_COUNT,
    StreamingExportWrapper,
    TemporalGestureGru,
    checkpoint_config,
)


class TemporalGestureContractTests(unittest.TestCase):
    def test_five_class_label_contract_is_compact(self) -> None:
        self.assertEqual(
            GESTURE_CLASSES,
            ("none", "swipe_left", "swipe_right", "grab", "release"),
        )
        self.assertEqual(GESTURE_COUNT, len(GESTURE_CLASSES))

    def test_sequence_and_streaming_heads_emit_five_gesture_logits(self) -> None:
        model = TemporalGestureGru().eval()
        sequence_features = torch.zeros(1, 3, FEATURE_COUNT, dtype=torch.float32)
        hidden = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE, dtype=torch.float32)

        with torch.no_grad():
            gesture, phase, hidden_out = model.forward_sequence(
                sequence_features, hidden
            )
            stream_gesture, stream_phase, stream_hidden = StreamingExportWrapper(model)(
                sequence_features[:, 0, :], hidden
            )

        self.assertEqual(tuple(gesture.shape), (1, 3, GESTURE_COUNT))
        self.assertEqual(tuple(phase.shape), (1, 3, PHASE_COUNT))
        self.assertEqual(tuple(hidden_out.shape), (NUM_LAYERS, 1, HIDDEN_SIZE))
        self.assertEqual(tuple(stream_gesture.shape), (1, GESTURE_COUNT))
        self.assertEqual(tuple(stream_phase.shape), (1, PHASE_COUNT))
        self.assertEqual(tuple(stream_hidden.shape), (NUM_LAYERS, 1, HIDDEN_SIZE))

    def test_removed_label_ids_are_rejected(self) -> None:
        self.assertEqual(
            _label({"gesture_label": 4}, "gesture_label", GESTURE_COUNT), 4
        )
        with self.assertRaisesRegex(
            ValueError, r"gesture_label must be within \[0,4\]"
        ):
            _label({"gesture_label": 5}, "gesture_label", GESTURE_COUNT)

    def test_checkpoint_with_reordered_labels_is_rejected(self) -> None:
        model = TemporalGestureGru()
        payload = {
            "model_state": model.state_dict(),
            "config": checkpoint_config(),
            "gesture_classes": [
                "none",
                "swipe_right",
                "swipe_left",
                "grab",
                "release",
            ],
            "phase_classes": ["idle", "start", "active", "end"],
        }
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "reordered.pt"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "label contract"):
                load_model(checkpoint)

    def test_legacy_eight_class_checkpoint_is_rejected_by_consumers(self) -> None:
        config = checkpoint_config()
        config["gesture_count"] = 8
        payload = {
            "model_state": TemporalGestureGru().state_dict(),
            "config": config,
            "gesture_classes": [
                "none",
                "wave",
                "swipe_left",
                "swipe_right",
                "grab",
                "release",
                "point",
                "click",
            ],
            "phase_classes": ["idle", "start", "active", "end"],
        }
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy-eight-class.pt"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                load_model(checkpoint)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                load_export_model(checkpoint, contract_smoke=False)

    def test_legacy_dataset_without_label_contract_is_rejected(self) -> None:
        legacy_record = {
            "sequence_id": "legacy-eight-class-subset",
            "subject_id": "subject-legacy",
            "track_id": 1,
            "timestamp_ns": 1_000_000_000,
            "image_width": 640,
            "image_height": 480,
            "landmarks": [
                [100.0 + index, 200.0 - index, float(index) * 0.1]
                for index in range(21)
            ],
            "palm_confidence": 0.95,
            "landmark_confidence": 0.95,
            "handedness": "right",
            "static_pose": "open",
            "gesture_label": 1,
            "phase_label": 2,
        }
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "legacy.jsonl"
            dataset.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gesture_label_contract"):
                load_sequences(dataset)

            legacy_record["gesture_label_contract"] = (
                "kfcore-temporal-gesture-classes/0"
            )
            dataset.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gesture_label_contract"):
                load_sequences(dataset)

            legacy_record["gesture_label_contract"] = GESTURE_LABEL_CONTRACT
            dataset.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")
            examples = load_sequences(dataset)
            self.assertEqual(examples[0].gesture_labels, [1])


if __name__ == "__main__":
    unittest.main()
