# Temporal hand gesture recognition

`KFCore::hand_gesture` is the learned causal temporal-recognition layer for tracked hands.
It is backend-neutral and uses the existing KFCore Runtime / Model Package path; backend
plugins remain tensor executors only.

## Data flow

```text
HandDetector
  -> HandTracker
  -> GestureFeatureEncoder
  -> TemporalGestureRecognizer
  -> GestureEvent
  -> HandInteractionPipeline
```

`GestureFeatureEncoder` consumes one tracked `HandResult` and emits exactly 78 FP32 values.
It does not consume THIG observations or hand-authored gesture primitives.

The fixed V1 feature layout is:

- 63 local landmark values: 21 x `(x,y,z)`, wrist-centered and palm-scale normalized;
- normalized global wrist `(x,y)`;
- normalized global wrist velocity `(vx,vy)` in units/second;
- palm scale normalized by image diagonal;
- palm orientation as `sin(theta), cos(theta)`;
- handedness scalar `-1/0/+1`;
- landmark and Palm confidence;
- `dt` in seconds;
- four-value static-pose one-hot: Unknown/Open/Closed/Pointer.

Left hands mirror only local x. Global wrist position/velocity keep physical image direction,
so `SwipeLeft` and `SwipeRight` semantics are not destroyed by canonicalization.

## Model Package V1

Canonical model type:

```text
gesture.temporal-gru
```

Required V1 flavor:

```text
causal-gru-v1
```

V1 requires one explicit execution route:

```text
backend = onnxruntime
device  = cpu
```

There is no implicit fallback.

Fixed tensor contract:

```text
inputs
  features          FP32 [1,78]
  hidden_in         FP32 [2,1,64]

outputs
  gesture_logits    FP32 [1,8]
  phase_logits      FP32 [1,4]
  hidden_out        FP32 [2,1,64]
```

The model therefore uses the existing fixed Host-tensor ABI. No dynamic-output capability or
plugin ABI change is required.

## Streaming state

`TemporalGestureRecognizer` owns bounded state per non-negative `track_id`:

```text
previous feature-encoder state
128 FP32 GRU hidden values
structural event-decoder state
```

The first accepted observation uses `dt=0` and zero velocity. A non-monotonic timestamp or a
gap above `maximum_observation_gap_ns` resets that track before the current observation is
processed as a new first observation.

A complete frame is transactional: candidate states for all hands are committed only after all
model steps succeed. A failed track is reset and no partially computed hidden state is committed.

## Learned outputs

Gesture classes are frozen as:

```text
0 None
1 Wave
2 SwipeLeft
3 SwipeRight
4 Grab
5 Release
6 Point
7 Click
```

Phases are:

```text
0 Idle
1 Start
2 Active
3 End
```

The runtime decoder enforces only structural phase continuity and the global
`minimum_confidence` floor. It contains no gesture-specific dwell, duration, reversal, repeat, or
speed threshold.

`HandInteractionPipeline` decides application semantics such as whether a completed swipe while
grabbing means `Drag Left`; it does not re-evaluate whether the physical swipe or wave occurred.

## Training boundary

Training/export is intentionally outside the C++ runtime. A training pipeline must reproduce the
same 78-value canonicalization and export the fixed tensor contract above. Dataset/evaluation
requirements are frozen in:

`docs/superpowers/specs/2026-09-15-temporal-gesture-gru-v1-design.md`

Model quality is evaluated as gesture events, not only per-frame accuracy: event precision/recall,
false activations per minute, completion latency, incomplete-gesture rejection, and subject-separated
validation/test results.
