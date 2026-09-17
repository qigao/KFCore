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
  "sequence_id": "subject-001-swipe-left-03",
  "gesture_label_contract": "kfcore-temporal-gesture-classes/1",
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

`gesture_label_contract` is required on every record and must exactly equal the
value shown above. Gesture labels are fixed: `0 none, 1 swipe_left, 2 swipe_right, 3 grab,
4 release`.
Phase labels are fixed: `0 idle, 1 start, 2 active, 3 end`.

Old datasets without this contract marker and labels `5..7` from the former
eight-class experiment are invalid. Old datasets, checkpoints, and model packages
must be relabeled/retrained; no compatibility map is applied.

`features.py` is the Python reference implementation of the same frozen 78-value
encoder used by C++. Training sequences are grouped by `(sequence_id, track_id)` and
sorted by timestamp. Train/validation subject IDs must not overlap when `subject_id`
is present.

Capture data through the production `HandDetector -> HandTracker` path so the records
match the deployed landmark, handedness, confidence, static-pose, timestamp, and track
semantics. A generic RGB gesture dataset is useful only if it can be converted through
that same path and relabeled with the class/phase contract above; it is not a drop-in
replacement for these JSONL sequences.

Record all four dynamic classes on left and right hands, with swipes in both
directions and at different speeds, amplitudes, distances, backgrounds, and frame
rates. Record more `none` time than
gesture time, including ordinary reaching, hand entry/exit, incomplete swipes,
open/close fidgeting, tracker reacquisition, and object manipulation as hard negatives.
Split train/validation/test by subject before extracting clips; never split adjacent
frames from one recording across sets.

## Prepare a public-dataset source manifest

The [Qualcomm Jester dataset](https://www.qualcomm.com/developer/software/jester-dataset)
can contribute `swipe_left`, `swipe_right`, and hard-negative `none` clips.
It does not contain equivalent `grab`/`release` labels, and its public split metadata
does not expose subject identities. Treat its output as `pretrain_only`: it cannot be
used for leakage-safe final validation or test results.

After accepting the terms on the
[official download page](https://www.qualcomm.com/developer/software/jester-dataset/downloads)
yourself and placing the decoded frames on `F:`, run:

```text
python tools/temporal_gesture/prepare_dataset.py --labels F:\KFCoreDatasets\temporal_gesture\raw\jester\jester-v1-train.csv --frames-root F:\KFCoreDatasets\temporal_gesture\raw\jester\20bn-jester-v1 --source-split train
```

The command writes
`F:\KFCoreDatasets\temporal_gesture\manifests\jester-train.jsonl` by default. It
selects only the exact Jester labels `Swiping Left`, `Swiping Right`, `No gesture`,
and `Doing other things`, records all skipped-label counts, validates each selected
frame directory, and refuses to overwrite an existing manifest.

This source manifest is intentionally not accepted by `train.py`. It contains no
landmarks or phase labels. Run the selected RGB clips through the production
`HandDetector -> HandTracker` path, preserve timestamps and tracking semantics, then
annotate phases consistently to produce the frame JSONL contract above. Collect
subject-identified KFCore recordings for `grab`, `release`, fine-tuning, validation,
and final testing.

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
completion latency and a 5x5 frame confusion matrix. Use a subject-separated test
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
binding, and recurrent hidden-state progression. It does not prove recognition
accuracy.
