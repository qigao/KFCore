# Hand Tracking Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce the current ByteTrack-plus-canonical-identity resilience contract before considering a tracker replacement.

**Architecture:** ByteTrack remains the raw palm-box association layer, while `HandPrimitiveExtractor` remains the semantic identity owner used by THIG. The change adds executable boundary and integration contracts without changing public APIs, defaults, or dependencies.

**Tech Stack:** C++17, ByteTrack C API, KFCore hand models, KFCore hand interaction, THIG, TinyTest, CMake presets.

**Spec:** `docs/superpowers/specs/2026-09-15-hand-tracking-resilience.md`

## Global Constraints

- Do not replace ByteTrack or add another tracker dependency.
- Do not change public APIs, option defaults, action names, or ABI.
- Keep appearance opt-in and preserve source-free `HandTracker::update(HandFrame)`.
- Treat `canonical_id == 0` as deliberate fail-closed ambiguity.
- Preserve all existing dirty-worktree changes and do not commit or reset them.

---

### Task 1: Lock the raw ByteTrack dropout contract

**Files:**
- Modify: `vision/core/hand_models/tests/test_hand_tracking.cpp`
- Modify only if a contract test fails: `vision/core/hand_models/src/tracker.cpp`

**Interfaces:**
- Consumes: `HandTracker::create(const HandTrackingOptions&)` and `HandTracker::update(HandFrame)`.
- Produces: executable confirmation and lost-buffer behavior at the KFCore wrapper boundary.

- [x] Add a contract test that confirms one hand, submits one empty frame, then verifies the same non-negative raw ID on spatial reacquisition.
- [x] Run `test_hand_tracking`; if the test already passes, record it as characterization evidence rather than claiming a production fix.
- [x] Add a contract test using `lost_track_buffer = 1` that advances empty frames beyond expiry and requires the next detection to return `track_id == -1`.
- [x] Run the focused test. If behavior differs from the upstream tracker contract, inspect the C tracker tests before making a minimal wrapper fix.

### Task 2: Lock canonical gesture continuity across raw-ID replacement

**Files:**
- Modify: `hand_interaction/tests/test_hand_interaction.cpp`
- Modify only if the integration contract fails: `hand_interaction/src/hand_identity.cpp`

**Interfaces:**
- Consumes: `HandInteractionPipeline::process`, tracked `HandFrame` values, and the existing Open-to-Fist Grasp graph.
- Produces: a test proving THIG state follows canonical identity rather than raw tracker identity.

- [x] Add an integration test that establishes stationary Open evidence under raw ID 7, changes the reliable hand to raw ID 71, then supplies stationary Fist evidence.
- [x] Assert that every reliable frame resolves to the original positive canonical ID and that Grasp is emitted from that canonical source.
- [x] Run `test_hand_interaction`; if it already passes, retain it as an architectural regression test and do not modify production identity logic.

### Task 3: Document the replacement gate and verify

**Files:**
- Modify: `vision/core/hand_models/README.md`
- Modify: `hand_interaction/README.md`
- Modify: `docs/superpowers/plans/2026-09-15-hand-tracking-resilience.md`

**Interfaces:**
- Consumes: the raw tracking and canonical identity contracts proven in Tasks 1-2.
- Produces: a documented decision boundary for enabling appearance or evaluating an alternative tracker.

- [x] Document ByteTrack as a raw-ID layer and canonical identity as the THIG identity fact source.
- [x] Document the recorded-sequence metrics required before replacing ByteTrack.
- [x] Read the active CMake preset inheritance, format modified C++ tests, and build the two focused targets through `win-release-user` in the Visual Studio environment.
- [x] Run focused tracker/primitive/interaction/THIG tests, then the complete configured CTest suite.
- [x] Run `git diff --check`, verify no task checkboxes remain open, and mark the plan complete only after evidence exists.
