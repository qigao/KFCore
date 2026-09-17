# Hand Interaction THIG Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce false activations and multi-hand interference in production THIG gesture recognition while preserving the current gesture set and generic THIG ABI.

**Architecture:** Keep primitive extraction as the observation source and the hand-interaction graph as the policy layer. Bind the single interaction lifecycle only on a completed Grasp, strengthen Swipe through existing observation-window settings, and compose two-hand shapes with per-source stationary evidence.

**Tech Stack:** C++17, KFCore THIG, TinyTest, CMake presets.

**Spec:** `docs/superpowers/specs/2026-09-15-hand-interaction-thig-robustness.md`

## Global Constraints

- Preserve Wave, Point, Click, and Pinch exclusion.
- Do not add a GRU dependency or modify the generic THIG public ABI.
- Keep one active grasp/drag lifecycle per pipeline.
- Preserve existing action names and public setting fields.
- Use the existing dirty feature branch in place because this work continues its uncommitted hand-interaction changes; do not isolate, reset, or overwrite unrelated changes.
- Run the smallest relevant test after each production change, then the adjacent and complete configured test suites.

---

### Task 1: Bind interaction ownership only on Grasp

**Files:**
- Modify: `hand_interaction/tests/test_hand_interaction.cpp`
- Modify: `hand_interaction/src/gesture_graph.cpp`

- [x] Add a regression test where source 1 remains open while source 2 changes from open to fist; require `Grasp` from source 2.
- [x] Run `test_hand_interaction` and verify the test fails because idle relation binding reserves source 1.
- [x] Remove idle `bindOnRelations`; retain `bindSource = true` on the Grasp transition.
- [x] Rebuild and verify `test_hand_interaction` passes.

### Task 2: Require sustained default Swipe evidence

**Files:**
- Modify: `hand_interaction/tests/test_hand_interaction.cpp`
- Modify: `hand_interaction/include/kfcore/hand_interaction/hand_interaction.hpp`

- [x] Add a default-settings test that rejects left-swipe activation through 66 ms and accepts sustained evidence at 100 ms.
- [x] Run the focused test and verify V3 defaults activate too early.
- [x] Set direction dwell/window/minimum observations/sample capacity to 67 ms, 167 ms, 3, and 5; preserve ratio and switch margin.
- [x] Bump the semantic graph version from V3 to V4 and verify the focused tests pass.

### Task 3: Require stationary two-hand shape participants

**Files:**
- Modify: `hand_interaction/tests/test_hand_interaction.cpp`
- Modify: `hand_interaction/src/gesture_graph.cpp`

- [x] Add a failing test that supplies stationary V plus unstable Fist and rejects `V Fist`.
- [x] Extend positive `V Fist` and two-hand V coverage with stationary evidence from both distinct sources.
- [x] Compose each shape with same-source `Motion Stationary`, then join the two composed hands with `Distinct` source semantics.
- [x] Rebuild and verify focused tests pass.

### Task 4: Align hand-tracking tests with the ByteTrack contract

**Files:**
- Modify: `vision/core/hand_models/tests/test_hand_tracking.cpp`
- Modify: `vision/core/hand_models/README.md`

- [x] Assert `track_id == -1` on a new track, non-negative identity after a matched frame, and stability thereafter.
- [x] Assert reset restores the unconfirmed first-frame state and later reconfirms.
- [x] Run the hand-tracking and underlying tracker tests together.

### Task 5: Synchronize documentation and verify regressions

**Files:**
- Modify: `hand_interaction/README.md`
- Modify: `docs/superpowers/specs/2026-09-15-hand-interaction-thig-restoration.md`
- Modify: `docs/superpowers/plans/2026-09-15-hand-interaction-thig-robustness.md`

- [x] Document V4 binding, Swipe confirmation, and stationary two-hand semantics.
- [x] Mark the restoration design's V3 timing semantics as superseded by this V4 hardening specification.
- [x] Format modified C++ files and run focused THIG, primitive, interaction, hand-model, and tracker tests.
- [x] Run the complete configured CTest suite and `git diff --check`.
- [x] Mark every completed plan checkbox only after its verification evidence exists.
