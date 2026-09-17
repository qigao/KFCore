# Hand Interaction THIG Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore production swipe/grab/release recognition to `hand_interaction`'s THIG graph while keeping the untrained GRU subsystem independent and experimental.

**Architecture:** Primitive observations remain the single input to a per-pipeline `TemporalGraphEngine`; the graph owns swipe, grasp/release, and drag state. The facade removes learned `GestureEvent` input and the target drops its `hand_gesture` dependency, while Region observations remain validated output context outside the graph.

**Tech Stack:** C++17, KFCore THIG, TinyTest, CMake presets.

**Spec:** `docs/superpowers/specs/2026-09-15-hand-interaction-thig-restoration.md`

## Global Constraints

- Dynamic production actions are limited to Swipe Left, Swipe Right, Grasp, Release, and the existing left/right drag lifecycle.
- Wave, Point, Click, and Pinch must not be restored.
- The graph version is exactly `kfcore-hand-interaction-v3`.
- `HandInteractionPipeline::process()` consumes `HandFrame`, `GestureFrameContext`, and optional external Region observations only.
- `KFCore::hand_interaction` must not link `KFCore::hand_gesture`.
- Validation and THIG failures must not commit staged primitive state.
- No fallback from THIG to GRU or from GRU to THIG is permitted.

---

### Task 1: Restore selected dynamic THIG actions

**Files:**
- Modify: `hand_interaction/include/kfcore/hand_interaction/hand_interaction.hpp`
- Modify: `hand_interaction/src/gesture_graph.cpp`
- Test: `hand_interaction/tests/test_hand_interaction.cpp`

**Interfaces:**
- Consumes: primitive relations `Direction Left/Right`, `Direction Neutral`, `Shape Open/Fist`, and `Motion Stationary`.
- Produces: THIG actions `Swipe Left/Right`, `Grasp`, `Release`, `Drag Start Left/Right`, `Drag Left/Right`, `Drag End`, and `Drag Cancelled`.

- [x] **Step 1: Add a failing graph test for open-hand swipes**

```cpp
it("recognizes left and right open-hand swipes in THIG")
{
    TemporalGraphEngine left(build_hand_interaction_graph(immediate_settings()));
    check_true(has_action(feed(left, 1, 0,
        {relation("Shape Open", 1, 1), relation("Direction Left", 1, 1)}),
        "Swipe Left"));
}
```

- [x] **Step 2: Build and run the focused test to verify RED**

Run: `cmake --build --preset win-release-user --target test_hand_interaction`

Run: `ctest --preset win-release-user -R test_hand_interaction`

Expected: FAIL because the current graph contains no Swipe action.

- [x] **Step 3: Restore direction settings, windows, swipe actions, and the selected interaction state graph**

```cpp
inline constexpr char kHandInteractionSpecVersion[] = "kfcore-hand-interaction-v3";

struct HandInteractionSettings {
    int direction_dwell_ms = 0;
    int direction_window_ms = 34;
    int grab_select_stable_ms = 167;
    int grab_release_stable_ms = 167;
    int grab_transition_max_ms = 2000;
    int neutral_rearm_ms = 0;
};
```

Add exact-capacity-validated direction observation windows, open-hand `Swipe Left/Right` action patterns, and the old bounded interaction state machine restricted to left/right drag. Do not add Wave or Click relations, patterns, settings, or state graphs.

- [x] **Step 4: Run the focused graph tests to verify GREEN**

Run: `ctest --preset win-release-user -R test_hand_interaction`

Expected: PASS for swipe, grasp/release, drag, static/spatial, reset, and capacity behavior.

### Task 2: Restore the THIG-only facade and dependency boundary

**Files:**
- Modify: `hand_interaction/include/kfcore/hand_interaction/hand_interaction.hpp`
- Modify: `hand_interaction/src/hand_interaction.cpp`
- Modify: `hand_interaction/CMakeLists.txt`
- Test: `hand_interaction/tests/test_hand_interaction.cpp`

**Interfaces:**
- Consumes: `process(const HandFrame&, const GestureFrameContext&, const std::vector<thig::Observation>& = {})`.
- Produces: owned `PrimitiveFrame` plus THIG `ActionEvent` values; `graph_state(const std::string&) const` exposes state for diagnostics.

- [x] **Step 1: Add a failing facade test that omits learned events**

```cpp
const auto result = pipeline.process(frame, frame_context(1, 0));
check_true(result.primitives.hands.size() == 1U);
```

- [x] **Step 2: Build to verify the old two-argument call does not compile**

Run: `cmake --build --preset win-release-user --target test_hand_interaction`

Expected: FAIL because the current public signature requires a `GestureEvent` vector.

- [x] **Step 3: Remove learned-event mapping/state and restore transactional THIG processing**

```cpp
result.primitives = staged_primitives.process(frame, context);
result.actions = impl_->engine.ProcessFrame(
    result.primitives.observations, context.observed_at);
impl_->primitives = std::move(staged_primitives);
```

Validate external Region observations against current canonical IDs, append them to the returned primitive frame only after THIG succeeds, restore `graph_state()`, rename `options.semantic` to `options.temporal`, and remove `KFCore::hand_gesture` from target links.

- [x] **Step 4: Build and run the complete hand interaction test**

Run: `cmake --build --preset win-release-user --target test_hand_interaction`

Run: `ctest --preset win-release-user -R test_hand_interaction`

Expected: PASS.

### Task 3: Align documentation and verify dependency closure

**Files:**
- Modify: `hand_interaction/README.md`
- Modify: `vision/core/hand_gesture/README.md`
- Modify: `docs/superpowers/specs/2026-09-15-temporal-gesture-gru-v1-design.md`
- Modify: `docs/superpowers/plans/2026-09-15-hand-interaction-thig-restoration.md`

**Interfaces:**
- Consumes: restored public API and graph actions from Tasks 1-2.
- Produces: documentation that labels GRU as experimental and THIG as the current production recognizer.

- [x] **Step 1: Update docs without deleting the experimental GRU contract**

```text
HandDetector -> HandTracker -> HandPrimitiveExtractor -> THIG -> ActionEvent
```

Document the four selected dynamic gestures, excluded Wave/Point/Click/Pinch classes, configuration ownership, public API incompatibility, and migration back to THIG.

- [x] **Step 2: Check source and link closure**

Run: `rg.exe -n "GestureEvent|KFCore::hand_gesture|gesture.temporal-gru|Wave|Click" hand_interaction`

Expected: no production-code or target dependency matches; excluded gestures may appear only in explanatory documentation/tests asserting absence.

- [x] **Step 3: Run scoped and adjacent verification**

Run: `ctest --preset win-release-user -R "test_hand_interaction|test_thig_temporal_graph|test_hand_primitive_extractor|test_hand_gesture_runtime_contract"`

Run from `tools/temporal_gesture`: `python -m unittest discover -s . -p "test*.py" -v`

Expected: all scoped tests PASS; GRU tooling remains independently testable.

- [x] **Step 4: Inspect the final diff**

Run: `git diff --check -- hand_interaction vision/core/hand_gesture/README.md docs/superpowers/specs/2026-09-15-temporal-gesture-gru-v1-design.md docs/superpowers/specs/2026-09-15-hand-interaction-thig-restoration.md docs/superpowers/plans/2026-09-15-hand-interaction-thig-restoration.md`

Expected: no whitespace errors.
