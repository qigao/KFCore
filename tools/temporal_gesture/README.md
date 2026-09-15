# Temporal Gesture GRU tools

This directory contains the reference training/export path for `gesture.temporal-gru`.
It is deliberately separate from the production C++ runtime.

## Python environment

Use Python 3.10+ with PyTorch and ONNX installed. Record the exact package versions
used for a production training run together with the dataset/model artifact.

## Dataset JSONL contract

Each line is one tracked-hand frame:

```json
{
  "sequence_id": "subject-001-wave-03",
  "subject_id": "subject-001",
  "track_id": 7,
  "timestamp_ns": 1000000000,
  "image_width": 640,
  "image_height": 480,
  "landmarks": [[100.0,220.0,0.0], "... 20 more xyz triplets ..."],
  "palm_confidence": 0.97,
  "landmark_confidence": 0.95,
  "handedness": "right",
  "static_pose": "open",
  "gesture_label": 1,
  "phase_label": 2
}
```

Gesture labels are fixed: `0 none, 1 wave, 2 swipe_left, 3 swipe_right,
4 grab, 5 release, 6 point, 7 click`.
Phase labels are fixed: `0 idle, 1 start, 2 active, 3 end`.

`features.py` is the Python reference implementation of the same frozen 78-value
encoder used by C++. Training sequences are grouped by `(sequence_id, track_id)` and
sorted by timestamp. Train/validation subject IDs must not overlap when `subject_id`
is present.

## Train

```text
python tools/temporal_gesture/train.py \
  --train data/gesture_train.jsonl \
  --validation data/gesture_validation.jsonl \
  --output build/gesture/temporal_gesture.pt
```

The trainer uses the same two-layer GRU (`input=78`, `hidden=64`) as the exported
streaming model. Training uses full sequences; export wraps the same weights as one
causal frame step with explicit hidden state.

## Evaluate event quality

```text
python tools/temporal_gesture/evaluate.py \
  --checkpoint build/gesture/temporal_gesture.pt \
  --dataset data/gesture_test.jsonl \
  --output build/gesture/test_metrics.json
```

Evaluation reports event precision/recall/F1, false activations per minute,
completion latency and an 8x8 frame confusion matrix. Use a subject-separated test
set; do not tune runtime thresholds on the test set.

## Export a trained Model Package

```text
python tools/temporal_gesture/export_onnx.py \
  --checkpoint build/gesture/temporal_gesture.pt \
  --package-dir build/gesture/package
```

Export verifies the exact fixed ONNX contract and writes `model.json` with the ONNX
SHA-256 digest.

## Contract-only smoke package

Before real training data exists, generate deterministic untrained weights only to
exercise the runtime path:

```text
python tools/temporal_gesture/export_onnx.py \
  --contract-smoke \
  --package-id temporal-gesture-contract-smoke \
  --version 0.0.0 \
  --package-dir build/gesture/contract-smoke
```

This package is **not a gesture model**. It must not be used for quality claims.

Validate and run it with an ONNX Runtime backend plugin:

```text
build/bin/kfmodel validate build/gesture/contract-smoke
build/bin/kfgesture-smoke \
  build/gesture/contract-smoke \
  build/bin/plugins/kfcore_backend_onnxruntime.dll
```

A successful smoke proves Model Package loading, ORT plugin execution, fixed tensor
binding, and recurrent hidden-state progression. It does not prove wave/swipe/etc.
accuracy.
