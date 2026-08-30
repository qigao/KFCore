# Hand Wave Runtime Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make camera-derived open-palm waves reach THIG reliably, expose the primitive evidence needed to diagnose gestures, and bound repeated rotation actions.

**Architecture:** Keep landmark interpretation in `HandPrimitiveExtractor` and temporal composition in the existing THIG graph. Relax only the invalid wave exclusion, make the demo-only palm-axis gate explicit through CLI configuration, and expose existing primitive relations without adding a second state source.

**Tech Stack:** C++17, KFCore HandInteraction/THIG, TinyTest, CMake Presets, OpenCV Lite demo UI.

**Spec:** User-reported runtime behavior in this task; no separate specification file.

## Global Constraints

- Preserve the model, tracker, canonical hand identity, and THIG ownership boundaries.
- Use real `HandInteractionPipeline` frames for the Wave regression; do not mock primitive relations.
- Keep all thresholds bounded and configurable through existing option structures or demo CLI.
- Validate with `win-release-user` presets.

---

### Task 1: Camera-derived Wave regression

**Files:**
- Modify: `hand_interaction/tests/test_hand_interaction.cpp`
- Modify: `hand_interaction/src/gesture_graph.cpp`

**Interfaces:**
- Consumes: `HandInteractionPipeline::process(const HandFrame&, const GestureFrameContext&)`.
- Produces: Wave accepts a bounded Left/Neutral/Right/Neutral/Left reversal sequence while Open and Horizontal remain true.

- [x] Add a TinyTest case using translated landmark frames and the real primitive extractor.
- [x] Build and run `test_hand_interaction`; verify it fails because Neutral invalidates Wave.
- [x] Remove Neutral from the Wave forbidden-relation set while retaining vertical direction and non-open shape exclusions.
- [x] Rebuild and verify the focused test passes.

### Task 2: Explicit demo Wave orientation policy

**Files:**
- Modify: `hand_interaction/examples/hand_interaction_demo_cli.hpp`
- Modify: `hand_interaction/examples/hand_interaction_demo_cli.cpp`
- Modify: `hand_interaction/examples/hand_interaction_demo.cpp`
- Modify: `hand_interaction/tests/test_hand_interaction_demo_cli.cpp`

**Interfaces:**
- Produces: `--wave-require-horizontal-axis`; default remains permissive and the parsed value configures `HandInteractionSettings`.

- [x] Add failing CLI default/flag behavior tests.
- [x] Run the focused CLI test and confirm the new option is absent.
- [x] Add the typed argument and parser/help entry, then pass it to `HandInteractionPipeline`.
- [x] Rebuild and verify the CLI test passes.

### Task 3: Primitive diagnostic overlay

**Files:**
- Modify: `hand_interaction/examples/hand_interaction_demo_ui.cpp`
- Modify: `hand_interaction/tests/test_hand_interaction_demo_ui.cpp`

**Interfaces:**
- Produces: Per-hand derived text includes current Direction and Rotation from the same `PrimitiveFrame` used by THIG.

- [x] Add a failing UI formatting test with explicit Direction and Rotation observations.
- [x] Run it and confirm both fields are missing.
- [x] Extend the existing observation formatting path without adding duplicate state.
- [x] Rebuild and verify the UI test passes.

### Task 4: Rotation repeat bound

**Files:**
- Modify: `hand_interaction/include/kfcore/hand_interaction/hand_interaction.hpp`
- Modify: `hand_interaction/src/gesture_graph.cpp`
- Modify: `hand_interaction/tests/test_hand_interaction.cpp`

**Interfaces:**
- Produces: `HandInteractionSettings::rotation_cooldown_ms`; applied only to Rotate actions.

- [x] Add a failing temporal test proving a sustained rotation cannot emit repeatedly inside the configured cooldown.
- [x] Run it and confirm repeated Rotate is currently emitted.
- [x] Wire the setting to the two Rotate `ActionSpec::cooldownMs` fields.
- [x] Rebuild and verify the focused test passes.

### Task 5: Verification

**Files:**
- Modify only if verification exposes a directly related defect.

**Interfaces:**
- Produces: Reproducible build/test evidence for the changed HandInteraction and demo surfaces.

- [x] Build `test_hand_interaction`, `test_hand_interaction_demo_cli`, `test_hand_interaction_demo_ui`, and `hand_interaction_demo` with the documented build preset.
- [x] Run the focused CTest set through the TensorRT demo test preset.
- [x] Run the adjacent THIG and primitive-extractor tests.
- [x] Inspect the final diff and working-tree status for unrelated changes.
