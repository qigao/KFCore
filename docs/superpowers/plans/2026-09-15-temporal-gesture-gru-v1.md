# Temporal Gesture GRU V1 Implementation Plan

> **Status (2026-09-15):** The GRU runtime and tooling remain available as an
> experimental capability, but the production `hand_interaction` migration in
> Tasks 4-5 is superseded by
> `docs/superpowers/specs/2026-09-15-hand-interaction-thig-restoration.md`.
> Production gesture recognition currently uses THIG.

> **Execution note:** This project does not use a RED/GREEN/TDD workflow for this branch. Implement directly, then verify by source-level contract review, focused build configuration checks, and the user's manual Windows build/runtime smoke.

**Goal:** Add a causal GRU-based hand gesture recognizer as an experimental capability while reusing KFCore Model Package V1 and the backend-neutral runtime. The former production replacement goal is cancelled until a trained deployment model exists.

**Architecture:** Add a new static `vision/core/hand_gesture` module with deterministic 78-value feature encoding and a stateful typed runtime model. The model uses fixed Host tensors (`features`, `hidden_in`, `gesture_logits`, `phase_logits`, `hidden_out`) so no plugin ABI extension is needed. Production `hand_interaction` remains independent and consumes primitive observations through THIG; Tasks 4-5 below are retained only as superseded implementation history.

**Tech Stack:** C++17, KFCore runtime/ModelPackage V1, ONNX Runtime backend via existing plugin ABI, static KFCore CMake targets.

**Spec:** `docs/superpowers/specs/2026-09-15-temporal-gesture-gru-v1-design.md`

## Global Constraints

- Canonical model type is exactly `gesture.temporal-gru`.
- V1 feature vector is exactly 78 FP32 values.
- V1 hidden state is exactly FP32 `[2,1,64]`.
- V1 outputs are exactly `gesture_logits [1,5]`, `phase_logits [1,4]`, `hidden_out [2,1,64]`.
- V1 execution route is explicit ONNX Runtime CPU; no implicit fallback.
- No execution-plugin ABI extension.
- No legacy THIG hand-gesture adapter or compatibility aliases.
- No gesture-specific dwell, reversal, repeat, or duration thresholds in the learned-recognition path.
- First observation for a track uses `dt=0` and zero global wrist velocity; a non-monotonic timestamp or a gap above `maximum_observation_gap_ns` resets the track before processing that observation as a first observation.
- Recurrent state commits only after a successful complete model step.
- `maximum_tracks` is a hard capacity bound; no undocumented eviction.
- User manual compile/runtime smoke is the acceptance gate; do not claim compilation success without user evidence.

---

### Task 1: Add deterministic hand gesture feature contract

**Files:**
- Create: `vision/core/hand_gesture/include/kfcore/hand_gesture/types.hpp`
- Create: `vision/core/hand_gesture/include/kfcore/hand_gesture/feature_encoder.hpp`
- Create: `vision/core/hand_gesture/src/feature_encoder.cpp`
- Create: `vision/core/hand_gesture/CMakeLists.txt`
- Create: `vision/core/hand_gesture/tests/CMakeLists.txt`
- Create: `vision/core/hand_gesture/tests/test_feature_encoder.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces `GestureClass`, `GesturePhase`, `GestureEvent`, `GestureFrameMetadata`, and `GestureFeatureEncoder`.
- `GestureFeatureEncoder` accepts one `hand_models::HandResult`, image dimensions, current timestamp, and optional previous per-track encoder state.
- Produces `std::array<float, 78>` plus the next deterministic encoder state.

**Implementation steps:**
- [ ] Define the exact public enums/types from the spec, including explicit numeric class order.
- [ ] Implement wrist-centered, palm-scale normalized local XYZ landmarks.
- [ ] Mirror only local x for left hands; preserve global image-space motion direction.
- [ ] Encode global wrist position, velocity, palm scale, palm orientation, handedness, confidences, `dt`, and static-pose one-hot into fixed slots.
- [ ] Reject invalid image dimensions, non-finite landmarks/confidences, invalid palm scale, and impossible timestamps.
- [ ] Encode first observation as `dt=0`, velocity `(0,0)`.
- [ ] Add focused deterministic tests/golden values for feature count, mirroring, first observation, and frame-rate-independent velocity. Tests are verification artifacts, not a RED/GREEN workflow.
- [ ] Add `KFCore::hand_gesture` static target and root CMake wiring.

**Verification:**
- Inspect generated compile/link closure: only `KFCore::hand_model_core` plus standard library dependencies for the feature-only layer.
- No THIG include or target dependency is allowed in `vision/core/hand_gesture`.

---

### Task 2: Add `TemporalGestureRecognizer` typed runtime

**Files:**
- Create: `vision/core/hand_gesture/include/kfcore/hand_gesture/error.hpp`
- Create: `vision/core/hand_gesture/include/kfcore/hand_gesture/runtime.hpp`
- Create: `vision/core/hand_gesture/src/error.cpp`
- Create: `vision/core/hand_gesture/src/runtime.cpp`
- Create: `vision/core/hand_gesture/tests/test_runtime_contract.cpp`
- Modify: `vision/core/hand_gesture/CMakeLists.txt`

**Interfaces:**
- `TemporalGestureRecognizer::load(runtime, package, policy, options)`.
- `update(const hand_models::HandFrame&, const GestureFrameMetadata&) -> std::vector<GestureEvent>`.
- `reset()` and `reset_track(int)`.
- `execution_route()`.

**Implementation steps:**
- [ ] Validate `package.model_type() == "gesture.temporal-gru"`.
- [ ] Validate exactly two input tensors named `features` and `hidden_in` with FP32 fixed shapes `[1,78]` and `[2,1,64]`.
- [ ] Validate exactly three output tensors named `gesture_logits`, `phase_logits`, and `hidden_out` with FP32 fixed shapes `[1,5]`, `[1,4]`, `[2,1,64]`.
- [ ] Keep one bounded track-state record per non-negative `track_id`: encoder previous state, hidden state, and structural event decoder state.
- [ ] Reset a track on non-monotonic timestamp or observation gap above the configured maximum, then process current observation as first frame.
- [ ] Execute each live track synchronously using existing `ExecutionContext::run()` fixed Host tensors.
- [ ] Validate finite logits/hidden output before committing state.
- [ ] Softmax both heads; expose gesture confidence from the gesture head.
- [ ] Apply only structural phase handling (`Start/Active/End`) plus global `minimum_confidence`; no class-specific time thresholds.
- [ ] Commit hidden and encoder state transactionally after successful inference.
- [ ] Enforce `maximum_tracks` explicitly.
- [ ] Contain `runtime::RuntimeError`, allocation failures, and contract failures in typed `HandGestureError` categories.

**Verification:**
- Source-level tensor descriptor inspection must prove all runtime tensor shapes are fixed and Host-based.
- No `run_dynamic()` and no new backend ABI capability.

---

### Task 3: Integrate canonical Model Package identity and documentation

**Files:**
- Modify: `docs/runtime/model-package-v1.md`
- Modify parser/validation source only if canonical-model validation is enumerated there; otherwise keep parser generic and document the typed-model contract.
- Modify: `vision/core/hand_models/README.md` only where it still describes old `HandBackend` names.
- Create: `vision/core/hand_gesture/README.md`

**Implementation steps:**
- [ ] Add `gesture.temporal-gru -> kfcore::hand_gesture::TemporalGestureRecognizer` to canonical model types.
- [ ] Correct the three existing hand stage descriptions from `HandBackend` to `HandDetector`.
- [ ] Document V1 artifact flavor `causal-gru-v1`, required ORT CPU artifact, and optional future TensorRT derived artifacts under existing exact-runtime rules.
- [ ] Document fixed tensor names/shapes and state reset semantics.
- [ ] Explicitly state no runtime ABI extension and no implicit route fallback.

**Verification:**
- `kfmodel validate` remains generic Model Package validation; typed tensor contract is validated at model load.

---

### Task 4 (superseded): Move `hand_interaction` to learned `GestureEvent`

**Files:**
- Modify: `hand_interaction/include/kfcore/hand_interaction/hand_interaction.hpp`
- Modify: `hand_interaction/src/hand_interaction.cpp`
- Modify or replace: `hand_interaction/src/gesture_graph.cpp`
- Modify: `hand_interaction/CMakeLists.txt`
- Modify relevant `hand_interaction/tests/*`.

**Interfaces:**
- Consume `std::vector<hand_gesture::GestureEvent>` plus application/external context.
- Preserve deterministic application-state semantics where still useful.
- Produce semantic `ActionEvent` (existing public action representation may be retained if it is not THIG-coupled; otherwise introduce a hand-interaction-owned action type).

**Implementation steps:**
- [ ] Add public dependency on `KFCore::hand_gesture`.
- [ ] Remove recognition decisions based on primitive temporal windows/reversals/repeats.
- [ ] Map learned gesture phases to semantic actions/state transitions only.
- [ ] Keep legality, exclusivity, target binding, and product cooldowns only where they are application semantics rather than physical-gesture recognition.
- [ ] Ensure swipe recognition itself is never reconstructed from direction primitive history.
- [ ] Keep `HandPrimitiveExtractor` only for non-learned spatial/context features that still have a consumer; otherwise schedule it for removal in Task 5.

**Verification:**
- `hand_interaction` source must no longer decide `swipe`, `grab`, or `release` from THIG pattern timing.

---

### Task 5 (superseded): Remove hand-specific THIG recognition configuration and dead paths

**Files:**
- Modify: `hand_interaction/include/kfcore/hand_interaction/hand_interaction.hpp`
- Modify: `hand_interaction/include/kfcore/hand_interaction/types.hpp`
- Modify: `hand_interaction/src/primitive_extractor.cpp` as required by remaining non-GRU consumers.
- Modify: `hand_interaction/README.md`
- Modify: `hand_interaction/CMakeLists.txt`
- Remove obsolete `gesture_graph.cpp` or graph-specific tests if no longer used.
- Leave `thig/` generic library unchanged unless the repository has no remaining consumer; deletion of generic THIG is out of V1 scope.

**Implementation steps:**
- [ ] Remove gesture-recognition settings such as direction dwell/window, grab timing, and similar action-pattern thresholds without aliases.
- [ ] Remove THIG observation-window/state-graph capacity settings that existed solely for hand gesture recognition.
- [ ] Remove `KFCore::thig` link dependency from `hand_interaction` if no remaining deterministic state semantics require it.
- [ ] Keep data-quality/runtime bounds (`minimum_confidence`, max observation gap, max tracks) in `hand_gesture`, not `hand_interaction`.
- [ ] Update README architecture and examples to `HandDetector -> HandTracker -> TemporalGestureRecognizer -> HandInteraction`.

**Verification:**
- Search changed hand gesture/interaction code for obsolete identifiers: `wave_reversal_max_ms`, `wave_total_max_ms`, `direction_dwell_ms`, `grab_select_stable_ms`, `click_ready_dwell_ms`, and `HandBackend`.
- Confirm no compatibility alias reintroduces them.

---

### Final manual acceptance

- [ ] User performs fresh configure on `architecture/temporal-gesture-gru-v1`.
- [ ] User builds Windows MSVC configuration.
- [ ] Install/export target closure is checked by `cmake --install` or existing install preset.
- [ ] Once a real `gesture.temporal-gru` ONNX package exists, run `kfmodel validate` and an ORT CPU runtime smoke.
- [ ] Do not claim model-quality acceptance until a separately trained artifact is evaluated on subject-separated event metrics defined by the spec.
