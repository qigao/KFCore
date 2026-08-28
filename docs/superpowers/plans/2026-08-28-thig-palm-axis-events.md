# THIG Palm-Axis Events Implementation Plan

> **For Codex:** Execute this plan task by task with red-green-refactor and verify every claimed result from command output.

**Goal:** Expose generic palm-axis angle states from hand geometry, allow Wave to require a horizontal palm axis, and surface the state in the hand-interaction demo.

**Architecture:** `HandPrimitiveExtractor` remains the single source of hand geometry facts and emits one mutually exclusive palm-axis observation per reliable hand. `GestureGraphBuilder` consumes only symbolic observations and conditionally composes the Wave graph, so the THIG temporal core remains generic and unchanged. The library keeps the current Wave semantics by default; the demo explicitly opts into the stricter angle-aware rule.

**Tech Stack:** C++17, THIG temporal graph, TinyTest, CMake Presets, TensorRT demo UI.

---

### Task 1: Publish generic palm-axis observations

**Files:**
- Modify: `hand_interaction/include/kfcore/hand_interaction/types.hpp`
- Modify: `hand_interaction/src/primitive_extractor.cpp`
- Modify: `hand_interaction/src/gesture_graph.cpp`
- Test: `hand_interaction/tests/test_primitive_extractor.cpp`

1. Add failing tests for horizontal, diagonal, vertical, and modulo-180 axis classification, plus invalid threshold configuration.
2. Run `test_hand_primitive_extractor` and confirm the new tests fail for the missing behavior.
3. Add configurable named thresholds, validate their ordering and finite range, normalize orientation as an undirected axis, and emit exactly one axis relation.
4. Register the three relations in the gesture graph catalogue.
5. Re-run the focused test and confirm it passes.

### Task 2: Make the Wave angle condition optional

**Files:**
- Modify: `hand_interaction/include/kfcore/hand_interaction/hand_interaction.hpp`
- Modify: `hand_interaction/src/gesture_graph.cpp`
- Test: `hand_interaction/tests/test_hand_interaction.cpp`

1. Add failing tests proving an angle-aware Wave accepts a horizontal axis and rejects a missing/non-horizontal axis.
2. Run `test_hand_interaction` and confirm the tests fail for the missing option/condition.
3. Add an additive `wave_require_horizontal_palm_axis` setting defaulted to `false`.
4. Compose `Palm Axis Horizontal` with the existing open-palm Wave state only when the option is enabled.
5. Re-run the focused test and preserve the existing default Wave test.

### Task 3: Wire and display the angle-aware demo

**Files:**
- Modify: `hand_interaction/examples/hand_interaction_demo.cpp`
- Modify: `hand_interaction/examples/hand_interaction_demo_ui.cpp`
- Modify: `hand_interaction/README.md`
- Test: `hand_interaction/tests/test_hand_interaction_demo_ui.cpp`

1. Add a failing UI test for `Axis:Horizontal`.
2. Show the current palm-axis state in each hand overlay.
3. Construct the demo pipeline with `wave_require_horizontal_palm_axis = true`.
4. Document the axis display and the demo-specific Wave condition.
5. Run focused CPU tests, the full CPU suite, the TensorRT demo build/tests, then launch the updated demo without stopping it.
6. Review the diff, commit, push the feature branch, and update the existing pull request.
