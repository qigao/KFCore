# Temporal Gesture GRU V1 Design

## Status

Experimental architecture retained for future training and evaluation. It is not
the current production `hand_interaction` architecture; the production restoration
is specified in `docs/superpowers/specs/2026-09-15-hand-interaction-thig-restoration.md`.

The implemented runtime/tooling does not change the execution-plugin ABI and does
not add a new backend. Activation requires a trained artifact and a separate public
API decision.

## Goals

1. Recognize left/right swipe and grab/release from tracked hand motion without encoding each gesture as a collection of hand-authored dwell, duration, or action thresholds.
2. Keep recognition causal and streaming: one frame enters, one hidden state is updated, and no future frame is required.
3. Reuse KFCore Model Package V1 and the backend-neutral runtime. ONNX Runtime CPU is the required V1 execution route; TensorRT is an optional derived artifact later.
4. Separate learned gesture recognition from deterministic application state/action mapping.
5. Keep per-hand memory bounded and resettable.
6. Preserve explicit failure, capacity, and confidence boundaries. There is no implicit fallback.

## Non-goals

- Video/RGB sequence modeling in V1.
- Transformers, TCNs, bidirectional recurrent models, or future-frame lookahead.
- Multi-person or multi-hand joint recurrent inference in V1.
- Learned hand identity/re-identification.
- Moving model-specific behavior into backend plugins.
- TensorRT engine generation inside KFCore runtime.
- Encoding application navigation/business rules in the GRU.
- Keeping compatibility aliases for the old THIG hand-gesture contract.

## Experimental architecture

Hand gesture recognition becomes:

```text
HandDetector
    |
HandTracker
    |
per-track HandFrame
    |
GestureFeatureEncoder
    |
TemporalGestureRecognizer (causal GRU)
    |
GestureEvent
    |
InteractionStateMachine
    |
ActionEvent
```

If activated in a future migration, the GRU would replace the hand-specific
temporal-pattern recognition currently expressed through THIG windows, dwell times,
sequence/repeat nodes, gesture cooldowns, and related thresholds.

That migration is deferred. Production `hand_interaction` currently depends on THIG
for swipe/grab/release recognition and does not consume `GestureEvent`.

The deterministic layer after recognition is intentionally thin. It may enforce legal state transitions and application context, but it must not reconstruct gesture recognition through new duration/dwell/reversal thresholds.

## Why GRU

Dynamic hand gestures are trajectory patterns rather than isolated categorical states. A causal GRU can learn:

- directional trajectories such as left and right swipes;
- amplitude and velocity jointly;
- variable execution speed;
- incomplete versus completed gestures;
- pose transitions such as open -> closed -> open;
- tolerance to frame-rate variation, mild jitter, dropped observations, and natural user variation.

These are poorly represented by an expanding set of independently tuned millisecond and distance thresholds.

## Module boundaries

### `vision/core/hand_gesture`

New static KFCore module. It owns model semantics and the typed C++ API.

Primary types:

```cpp
namespace kfcore::hand_gesture {

enum class GestureClass : std::uint8_t {
    None = 0,
    SwipeLeft,
    SwipeRight,
    Grab,
    Release,
};

enum class GesturePhase : std::uint8_t {
    Idle = 0,
    Start,
    Active,
    End,
};

struct GestureFrameMetadata {
    std::uint64_t timestamp_ns = 0;
    int image_width = 0;
    int image_height = 0;
};

struct GestureEvent {
    int track_id = -1;
    GestureClass gesture = GestureClass::None;
    GesturePhase phase = GesturePhase::Idle;
    float confidence = 0.0F;
    std::uint64_t timestamp_ns = 0;
};

class GestureFeatureEncoder final;
class TemporalGestureRecognizer final;

} // namespace kfcore::hand_gesture
```

`GestureFeatureEncoder` contains deterministic canonicalization only. It does not classify temporal gestures.

`TemporalGestureRecognizer` owns one recurrent state per live hand track and executes one model step per observation.

### Future `hand_interaction` integration

An activated migration would make `hand_interaction` consume `GestureEvent` plus
application/external context. The current production interface does not.

It may retain deterministic state such as "menu open", "drag active", or "selected object", but not gesture-specific dwell/reversal windows.

### `thig`

THIG remains the production owner of selected hand-gesture timing. Existing generic
THIG APIs remain unchanged by the experimental GRU implementation.

## Per-frame feature contract

V1 uses one fixed FP32 feature vector per tracked hand.

### Canonicalization

For every valid hand observation:

1. Use the wrist landmark as the local origin.
2. Normalize local landmark coordinates by palm scale.
3. Mirror local x coordinates for left hands so local hand-shape geometry shares one canonical orientation.
4. Preserve global motion direction separately; left/right swipe semantics must not be destroyed by local mirroring.
5. Reject non-finite or invalid geometry rather than silently substituting zeros.

Palm scale is derived from stable palm landmarks and must be strictly positive.

The first accepted observation after track creation or reset is encoded with `dt=0` and global wrist velocity `(0,0)`. For subsequent observations the timestamp must strictly increase. If the gap exceeds `maximum_observation_gap_ns`, the track state is reset and the current sample is encoded as a new first observation. Otherwise `dt` is the exact positive elapsed time in seconds and global wrist velocity is computed from the previous accepted global wrist position. V1 does not clamp valid `dt` to an arbitrary gesture-specific interval.

### V1 feature vector: 78 FP32 values

| Range | Count | Meaning |
| --- | ---: | --- |
| local landmarks | 63 | 21 x `(x,y,z)`, wrist-centered and palm-scale normalized |
| global wrist position | 2 | image-normalized `(x,y)` |
| global wrist velocity | 2 | image-normalized units per second |
| palm scale | 1 | palm scale normalized by image diagonal |
| palm orientation | 2 | `sin(theta), cos(theta)` |
| handedness | 1 | `-1=left, 0=unknown, +1=right` |
| landmark confidence | 1 | `[0,1]` |
| palm confidence | 1 | `[0,1]` |
| delta time | 1 | exact seconds since previous accepted observation, or zero after reset |
| static pose one-hot | 4 | `Unknown, Open, Closed, Pointer` from `HandResult::gesture` |
| **total** | **78** | |

V1 intentionally does not feed THIG `Observation` values into the GRU. Otherwise the old threshold system would remain an upstream hidden dependency.

## Track and state semantics

Recognition state is keyed by `HandResult::track_id` in V1.

Each active track owns:

```text
previous accepted timestamp
previous global wrist position
GRU hidden state [2,1,64]
active gesture decoder state
```

State is reset when:

- the caller explicitly calls `reset()` or `reset_track()`;
- a track disappears longer than the configured maximum observation gap;
- the same track id receives a non-monotonic timestamp;
- model execution or feature validation fails for that track.

When a reset is caused by an excessive gap, the current valid sample may immediately seed a fresh state using the first-observation encoding. A non-monotonic timestamp is an invalid call and does not seed a new state.

V1 does not carry GRU state across a ByteTrack identity change. Canonical long-gap hand re-identification can be designed separately.

## Model contract

Canonical model type:

```text
gesture.temporal-gru
```

Typed API:

```text
kfcore::hand_gesture::TemporalGestureRecognizer
```

V1 model architecture used for training/reference export:

```text
input size: 78
GRU layers: 2
hidden size: 64
unidirectional: true
gesture head: 64 -> 5
phase head: 64 -> 4
```

The runtime contract is fixed even if training implementation details change.

### Tensor names and shapes

Inputs:

```text
features   FP32 [1,78]
hidden_in  FP32 [2,1,64]
```

Outputs:

```text
gesture_logits FP32 [1,5]
phase_logits   FP32 [1,4]
hidden_out     FP32 [2,1,64]
```

All dimensions are fixed in V1. Dynamic recurrent shapes are deliberately excluded.

The ONNX export may internally reshape/unsqueeze for the standard ONNX GRU operator, but the public model tensor contract above remains fixed.

## Streaming execution

For each accepted hand observation:

```text
HandResult + GestureFrameMetadata
    -> encode 78 features
    -> execute model(features, hidden_in)
    -> validate finite logits and hidden_out
    -> softmax heads
    -> decode class/phase
    -> commit hidden state
    -> emit zero or one GestureEvent for that track
```

The recognizer is synchronous and causal. Hidden state and previous-frame metadata are committed only after a successful complete model step.

A failed model step must not partially advance recurrent state.

## Event decoding

The model predicts both gesture class and gesture phase every frame.

V1 gesture classes are exactly:

```text
0 none
1 swipe_left
2 swipe_right
3 grab
4 release
```

V1 phases are exactly:

```text
0 idle
1 start
2 active
3 end
```

The runtime decoder may enforce only structural validity:

- `None` always produces no public event and closes no gesture by itself;
- a non-`None` class with `Idle` produces no public event;
- `Start` opens or replaces the active event for that track;
- `Active` continues only a matching active class;
- `End` closes only a matching active class;
- an explicit minimum confidence floor may suppress low-confidence output.

Mismatched `Active`/`End` predictions are treated as no public event, not as a reason to invent timing heuristics. They do not roll back a successfully computed hidden state.

The decoder must not use gesture-specific dwell times, reversal windows, repeat counters, or duration thresholds.

`confidence` is the selected gesture probability after softmax. Phase probability remains an internal diagnostic in V1 unless later promoted to the public event type.

## Model Package integration

`gesture.temporal-gru` is added to Model Package V1 canonical model types.

Required V1 artifact:

```json
{
  "id": "onnx-cpu",
  "format": "onnx",
  "path": "temporal_gesture.onnx",
  "flavor": "causal-gru-v1",
  "sha256": "<sha256>",
  "backend": "onnxruntime",
  "device": "cpu"
}
```

The package may later contain TensorRT derived artifacts under the existing exact-runtime rules. TensorRT support is not required for Temporal Gesture V1 acceptance.

No runtime ABI extension is required: the model uses ordinary fixed-shape Host tensors and explicit recurrent hidden-state tensors.

## Runtime policy

V1 requires an explicit ONNX Runtime CPU execution policy for the temporal model. Example:

```cpp
const auto gesture_policy = runtime::ExecutionPolicy::exact(
    runtime::ExecutionPreference{"onnxruntime", "cpu"});
```

The hand detector may independently execute on TensorRT CUDA, ORT CUDA, or ORT CPU. Gesture recognition is a separate logical model and therefore may choose its own backend/device.

There is no implicit fallback from a requested gesture execution route.

## Public API sketch

```cpp
namespace kfcore::hand_gesture {

struct TemporalGestureOptions {
    float minimum_confidence = 0.70F;
    std::uint64_t maximum_observation_gap_ns = 350'000'000ULL;
    std::size_t maximum_tracks = 8U;
};

class TemporalGestureRecognizer final {
public:
    static std::unique_ptr<TemporalGestureRecognizer> load(
        runtime::Runtime& runtime,
        const runtime::ModelPackage& package,
        const runtime::ExecutionPolicy& policy,
        const TemporalGestureOptions& options = {});

    std::vector<GestureEvent> update(
        const hand_models::HandFrame& frame,
        const GestureFrameMetadata& metadata);

    void reset();
    void reset_track(int track_id);

    const runtime::ExecutionRoute& execution_route() const noexcept;
};

} // namespace kfcore::hand_gesture
```

V1 remains C++17-compatible.

## Capacity and failure rules

- `maximum_tracks` is a hard bound.
- No unbounded per-track history is stored. Only the GRU hidden state, previous-frame feature metadata, and active event state are retained.
- Unknown or negative `track_id` values are not accepted for recurrent state.
- `image_width` and `image_height` must be positive.
- Timestamps must be monotonic per track.
- Feature values, logits, probabilities, and hidden state must be finite.
- A model tensor contract mismatch fails at load time.
- A runtime failure fails the affected call; hidden state is not committed.
- Capacity exhaustion is an explicit error, not eviction by undocumented policy.

## Training data contract

Training is outside the C++ runtime, but KFCore defines the semantic dataset contract so exported models are reproducible.

Each frame record contains at least:

```text
sequence_id
gesture_label_contract = kfcore-temporal-gesture-classes/1
track_id
timestamp_ns
image_width
image_height
21 xyz hand landmarks
palm box / palm confidence
landmark confidence
handedness
static pose class
gesture class label
gesture phase label
```

The training pipeline must call the same canonical feature transformation as the C++ runtime, or use a bit-for-bit compatible reference implementation verified against golden vectors.

### Required augmentation

Training should include:

- variable frame spacing / FPS;
- temporal speed scaling;
- mild landmark noise;
- short dropped-observation gaps;
- left/right hand examples and mirroring;
- different gesture amplitude;
- incomplete gesture prefixes;
- non-gesture motion as hard negatives.

Augmentation must not change the semantic direction label for global left/right swipe motion.

## Labeling semantics

A dynamic gesture is labeled with both class and phase.

Example left swipe:

```text
none/idle
swipe_left/start
swipe_left/active
swipe_left/active
swipe_left/end
none/idle
```

Incomplete swipe attempts should remain `none` or terminate without a valid `end`, depending on the dataset annotation policy. The policy must be consistent across training and evaluation sets.

The train/validation/test split must be subject-separated where user identity data is available. Frame-random splitting is not acceptable because adjacent frames leak nearly identical trajectories.

## Evaluation gates

A model is not accepted only because per-frame accuracy is high.

Required evaluation includes:

- gesture event precision/recall/F1;
- false activations per minute on non-gesture motion;
- event completion latency;
- incomplete-gesture rejection;
- 5x5 confusion matrix, especially left versus right swipe and grab versus release;
- results across multiple frame rates or irregular frame spacing;
- subject-separated validation/test performance.

Threshold selection uses validation data only. Test-set thresholds must not be tuned post hoc.

## Interaction-state layer

The deterministic state layer consumes `GestureEvent`, not raw landmark primitives.

Examples:

```text
swipe_left/end + gallery     -> previous page
grab/start     + object hit  -> begin drag
release/start  + dragging    -> drop
```

This layer may include application legality, exclusivity, target binding, and cooldown required by product semantics. It must not decide whether a physical swipe occurred by re-checking motion duration.

## Deferred THIG migration

The following migration is deferred until a trained model passes its quality gate:

1. Add `vision/core/hand_gesture` and `gesture.temporal-gru`.
2. Feed GRU `GestureEvent` output into `hand_interaction`.
3. Remove hand-specific THIG gesture-pattern construction and the corresponding dwell/reversal/window settings.
4. Keep generic THIG library code untouched unless no remaining consumer exists.
5. Do not provide a legacy adapter that synthesizes THIG `Observation` events from GRU output.

Settings expected to disappear from the hand gesture-recognition path include the hand-authored timing controls for direction dwell/windows, grab transition timing, and similar gesture-specific temporal thresholds.

Basic data-quality bounds such as model confidence, maximum observation gap, maximum tracks, and invalid-tensor rejection remain explicit runtime controls.

## Naming cleanup

The existing Model Package documentation still refers to the merged pre-cleanup `HandBackend` name for the three hand stages. V1 implementation must update that documentation to the current `HandDetector` naming while adding `gesture.temporal-gru`.

## Deployment

V1 deployment is intentionally lightweight:

```text
GPU or CPU:
  HandDetector

CPU:
  temporal_gesture.onnx
  ONNX Runtime backend plugin

stateful C++:
  HandTracker
  TemporalGestureRecognizer recurrent state
  InteractionStateMachine
```

The GRU is expected to be small enough that CPU execution avoids unnecessary GPU synchronization. Performance must still be measured rather than assumed.

## Experimental migration compatibility

No compatibility promise is made for a future activation of this experimental path.

No old `HandInteractionSettings` gesture-timing fields are preserved as aliases when their recognition responsibility moves into the model.

The five-class contract is intentionally incompatible with the former eight-class
experiment. Wave, Point, and Click are removed, labels are compactly renumbered, and
old datasets/checkpoints/packages must be relabeled or retrained rather than mapped at
runtime.

Generic THIG public API compatibility is outside this spec because THIG itself is not being redesigned here.

## Experimental acceptance criteria

Temporal Gesture V1 is complete when:

1. `gesture.temporal-gru` loads through existing Runtime/ModelPackage using explicit ORT CPU policy.
2. C++ feature encoding matches the frozen 78-value contract and a training/reference encoder on golden inputs.
3. A fixed-shape causal GRU step accepts `[1,78] + [2,1,64]` and returns gesture logits, phase logits, and next hidden state.
4. Per-track recurrent state is bounded, resettable, and transactionally updated only after successful inference.
5. A trained candidate recognizes `swipe`, `grab`, and `release` from learned sequence behavior without THIG gesture timing rules.
6. Any future `hand_interaction` integration is an explicit migration rather than the current production behavior.
7. There is no implicit backend/device fallback and no runtime ABI extension.
8. Documentation and Model Package canonical model type tables reflect `HandDetector` and `gesture.temporal-gru`.
9. Manual build remains the project acceptance path; model-quality claims require the evaluation protocol above rather than compile success alone.
