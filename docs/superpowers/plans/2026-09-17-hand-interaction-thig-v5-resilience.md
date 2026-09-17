# Hand Interaction THIG V5 Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent ghost two-hand activations and make default Swipe confirmation robust across common frame rates without adding learned-model dependencies or public API fields.

**Architecture:** Preserve same-source THIG dropout semantics and change only the recency aggregation of concurrent distinct-source matches. Keep the existing direction stabilizer, change the default sample floor from three to two while retaining the 67 ms duration gate, and lock behavior with real-engine TinyTest scenarios.

**Tech Stack:** C++17, KFCore THIG, TinyTest, CMake presets.

**Spec:** `docs/superpowers/specs/2026-09-17-hand-interaction-thig-v5-resilience.md`

## Global Constraints

- Do not add or remove public fields, actions, dependencies, or configuration formats.
- Preserve same-source `Both` bounded-dropout behavior.
- Preserve the 67 ms direction dwell, 167 ms window, 0.60 support ratio, 0.15 switch margin, and five-sample bound.
- Keep production recognition deterministic and independent from the GRU experiment.
- Use the existing dirty feature branch in place; do not reset or overwrite unrelated changes.
- Run the smallest relevant test first, then adjacent hand-interaction tests, then the complete configured suite.

---

### Task 1: Require current evidence from both distinct sources

**Files:**
- Modify: `thig/tests/test_temporal_graph.cpp`
- Modify: `thig/src/temporal_graph.cpp`
- Modify: `hand_interaction/tests/test_hand_interaction.cpp`

**Interfaces:**
- Consumes: existing `PatternSourceJoin::Distinct` and `PatternOperator::Both` graph semantics.
- Produces: distinct concurrent matches whose `latestObservedMs` is the older participant timestamp, so `CurrentMatches()` accepts them only when both participants are current.

- [x] **Step 1: Add a failing generic THIG regression**

  Feed source 4's first relation at 0 ms and source 8's second relation alone at
  20 ms. Assert that the distinct-source action list is empty.

- [x] **Step 2: Run the THIG test and verify RED**

  Run `ctest --preset win-release-user -R "^test_thig_temporal_graph$" --output-on-failure`.
  Expected: the new case fails because the historical source 4 match is treated
  as current through source 8.

- [x] **Step 3: Apply the minimal freshness fix**

  In `MergeDistinctMatches`, retain interval merging and source ordering but set
  concurrent `latestObservedMs` to:

  ```cpp
  merged.latestObservedMs =
      std::min(left.latestObservedMs, right.latestObservedMs);
  ```

- [x] **Step 4: Verify generic semantics**

  Re-run `test_thig_temporal_graph`. Confirm the new regression and the existing
  same-source observed-dropout case both pass.

- [x] **Step 5: Add and run the hand-interaction ghost regression**

  Feed stationary `Shape V` for source 1, then stationary `Shape Fist` for source
  2 without refreshing source 1. Assert that `V Fist` is absent, then refresh
  both sources and assert that it is present.

### Task 2: Make Swipe confirmation time-based across frame rates

**Files:**
- Modify: `hand_interaction/tests/test_hand_interaction.cpp`
- Modify: `hand_interaction/include/kfcore/hand_interaction/hand_interaction.hpp`

**Interfaces:**
- Consumes: `HandInteractionSettings::direction_minimum_supporting_observations`.
- Produces: default value `2`; all existing caller overrides retain their behavior.

- [x] **Step 1: Add a failing jittered 12 FPS case**

  Feed default open-left observations at literal timestamps 0 ms and 86 ms.
  Assert no Swipe before 67 ms and `Swipe Left` at 86 ms.

- [x] **Step 2: Run the interaction test and verify RED**

  Run `ctest --preset win-release-user -R "^test_hand_interaction$" --output-on-failure`.
  Expected: the 12 FPS case fails because the current three-sample floor cannot
  be met with two sustained samples.

- [x] **Step 3: Change only the default sample floor**

  Set:

  ```cpp
  int direction_minimum_supporting_observations = 2;
  ```

  Keep all time, ratio, margin, and capacity defaults unchanged.

- [x] **Step 4: Cover 30, 60, and 120 FPS plus an early negative**

  Add literal jittered timestamp sequences ending at or after 67 ms and assert
  each recognizes Swipe only at its final timestamp. Retain the existing default
  sustained-evidence test as a separate regression.

- [x] **Step 5: Verify focused tests**

  Rebuild and run `test_hand_interaction` and `test_thig_temporal_graph`.

### Task 3: Lock direction conflict hysteresis and publish V5 semantics

**Files:**
- Modify: `hand_interaction/tests/test_hand_interaction.cpp`
- Modify: `hand_interaction/include/kfcore/hand_interaction/hand_interaction.hpp`
- Modify: `hand_interaction/README.md`
- Modify: `docs/superpowers/plans/2026-09-17-hand-interaction-thig-v5-resilience.md`

**Interfaces:**
- Consumes: existing `rejectStableOnConflict`, support-ratio, and switch-margin behavior.
- Produces: version string `kfcore-hand-interaction-v5` and documented compatibility contract.

- [x] **Step 1: Add a direction conflict regression**

  Stabilize left direction, inject one right observation, assert no right action,
  then return left and assert left recovers. This test must exercise the real
  `TemporalGraphEngine` built by `build_hand_interaction_graph()`.

- [x] **Step 2: Run the regression**

  Confirm it passes with the existing stabilizer; if it fails, diagnose before
  altering the algorithm.

- [x] **Step 3: Bump the semantic graph version and update documentation**

  Change the version constant to `kfcore-hand-interaction-v5` and document the
  distinct-source freshness rule, two-sample/67 ms Swipe contract, and risks.

- [x] **Step 4: Run adjacent and complete verification**

  Build with `win-release-user`; run THIG, primitive extractor, hand interaction,
  hand tracking, and ByteTrack status tests, then the full preset suite and
  `git diff --check`.

- [x] **Step 5: Record completion evidence**

  Mark checkboxes only after the corresponding command output has been observed.

### Task 4: Resolve review findings

**Files:**
- Modify: `thig/src/temporal_graph.cpp`
- Modify: `thig/tests/test_temporal_graph.cpp`
- Modify: `hand_interaction/tests/test_hand_interaction.cpp`
- Modify: `hand_interaction/README.md`
- Modify: `docs/superpowers/specs/2026-09-17-hand-interaction-thig-v5-resilience.md`

**Interfaces:**
- Consumes: concurrent-match `endMs`, shared direction observation window, and interaction state graph.
- Produces: preserved overlap-recency upper bound and explicit Drag/Neutral compatibility coverage.

- [x] **Step 1: Reproduce nested distinct `During` revival**

  Add a state-gated graph where current containers surround ended contained
  intervals; verify the current implementation incorrectly emits one action.

- [x] **Step 2: Preserve both distinct freshness and overlap recency**

  Set concurrent distinct currentness to the minimum of `merged.endMs`, the left
  latest observation, and the right latest observation; verify the new regression
  and generic THIG suite pass.

- [x] **Step 3: Cover shared direction consumers and exact boundaries**

  Verify 67 ms and 167 ms are inclusive, 168 ms is outside the window, Drag Start
  uses two samples plus 67 ms, and Neutral rearm uses the shared two-sample window.

- [x] **Step 4: Correct compatibility documentation**

  State that the sample-floor change affects Swipe, Drag, switching, and Neutral
  rearm while ratio, margin, state binding, and relation-specific guards remain.
