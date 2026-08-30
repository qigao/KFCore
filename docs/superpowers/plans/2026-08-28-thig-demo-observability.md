# THIG Demo Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the camera demo distinguish raw model classes from derived hand primitives and keep semantic THIG actions visible long enough to inspect.

**Architecture:** Keep `vision_models::HandResult`, `HandInteractionFrame`, and THIG's one-shot `ActionEvent` contract unchanged. Add presentation-only formatting and a bounded recent-action history to the demo UI, then pass built-in graph states from the demo loop into the overlay.

**Tech Stack:** C++17, OpenCV-lite HighGUI/imgproc, TurboUtils TinyTest, CMake presets.

**Spec:** `docs/superpowers/specs/2026-08-28-thig-hand-interaction-demo-design.md`, refined by the current user request to expose derived gestures and complex-action diagnostics.

## Global Constraints

- Inference continues to consume the original, unmirrored camera frame.
- THIG actions remain one-shot domain events; only their demo presentation is retained for 1500 ms.
- Display history is bounded to four actions and resets with tracker/THIG state.
- No public KFCore library API or model format changes.

---

### Task 1: Per-hand raw and derived labels

**Files:**
- Modify: `hand_interaction/examples/hand_interaction_demo_ui.hpp`
- Modify: `hand_interaction/examples/hand_interaction_demo_ui.cpp`
- Test: `hand_interaction/tests/test_hand_interaction_demo_ui.cpp`

**Interfaces:**
- Consumes: `vision_models::HandResult`, canonical hand ID, and `PrimitiveFrame::observations`.
- Produces: `HandOverlayText format_hand_overlay_text(int, const HandResult&, const PrimitiveFrame&)` with separate identity/raw and derived/motion lines.

- [x] **Step 1: Write the failing test**

Create a real `PrimitiveFrame` containing `Shape V`, `Motion Stationary`, and `Pose OK` observations for hand 1, then require literal output:

```cpp
check_equal(text.identity, std::string("Hand ID:1 | Track ID:3 | Raw:Open"));
check_equal(text.derived, std::string("Derived:V | Motion:Stationary | Pose:OK"));
```

- [x] **Step 2: Run test to verify it fails**

Run the `test_hand_interaction_demo_ui` target and CTest filter with `win-release-user`; expect compilation to fail because `HandOverlayText` and `format_hand_overlay_text` do not exist.

- [x] **Step 3: Write minimal implementation**

Filter observations by `source.kind == "hand"` and canonical ID, strip the `Shape ` and `Motion ` prefixes, append `Pose:OK` when present, and use `Unknown` for absent groups. Replace the existing raw-only hand label with the two returned lines.

- [x] **Step 4: Run test to verify it passes**

Build and run the same CPU target/filter; expect one passing test executable.

### Task 2: Bounded recent-action presentation

**Files:**
- Modify: `hand_interaction/examples/hand_interaction_demo_ui.hpp`
- Modify: `hand_interaction/examples/hand_interaction_demo_ui.cpp`
- Test: `hand_interaction/tests/test_hand_interaction_demo_ui.cpp`

**Interfaces:**
- Consumes: current-frame `std::vector<thig::ActionEvent>` and injected `steady_clock::time_point`.
- Produces: `RecentActionHistory::update(...)`, returning at most four actions received during the last 1500 ms, plus `reset()`.

- [x] **Step 1: Write the failing tests**

Require a `Grasp` event to remain visible at 1499 ms, disappear at 1500 ms, discard the oldest event on the fifth insertion, and become empty after `reset()`.

- [x] **Step 2: Run test to verify it fails**

Build the CPU UI-test target; expect compilation failure because `RecentActionHistory` does not exist.

- [x] **Step 3: Write minimal implementation**

Store action plus expiry in a private vector, erase expired entries before insertion, append new events with the configured expiry, and erase from the front when capacity is exceeded. Reject zero retention or capacity in the constructor.

- [x] **Step 4: Run test to verify it passes**

Build and run the CPU UI CTest filter; expect all UI cases to pass.

### Task 3: Wire persistent actions and graph states into the overlay

**Files:**
- Modify: `hand_interaction/examples/hand_interaction_demo.cpp`
- Modify: `hand_interaction/examples/hand_interaction_demo_ui.hpp`
- Modify: `hand_interaction/examples/hand_interaction_demo_ui.cpp`
- Modify: `hand_interaction/tests/test_hand_interaction_demo_ui.cpp`
- Modify: `hand_interaction/README.md`

**Interfaces:**
- Consumes: `RecentActionHistory`, `HandInteractionPipeline::graph_state`, and the current `HandInteractionFrame`.
- Produces: `DemoThigStatus` and stable overlay lines for `hand_interaction_cycle`, `wave_cycle`, and `screen_click_cycle`.

- [x] **Step 1: Write the failing formatting test**

Require literal state and action labels:

```cpp
check_equal(format_thig_state_line(status),
            std::string("THIG hand=armed | wave=armed | click=ready"));
check_equal(format_action_line(grasp),
            std::string("ACTION: Grasp | hand ID:1"));
```

- [x] **Step 2: Run test to verify it fails**

Build the CPU UI-test target; expect missing `DemoThigStatus` and formatter symbols.

- [x] **Step 3: Write minimal implementation and wiring**

Render the THIG state line below capture metrics, render retained actions below it, query all three graph IDs after each pipeline call, update history with `thig_finished`, and clear history when `R` resets the pipeline.

- [x] **Step 4: Verify CPU and TensorRT paths**

Build `hand_interaction_demo` and `test_hand_interaction_demo_ui` with both hand-demo presets. Run the focused UI tests, the full CPU CTest preset, then a bounded or user-controlled TensorRT camera smoke test.

- [x] **Step 5: Document and commit**

Update the README to define `Raw`, `Derived`, `Motion`, graph-state, and 1500 ms action-retention semantics. Run `git diff --check`, commit the code and tests, and push `feature/thig-hand-interaction-demo` so PR #19 updates.
