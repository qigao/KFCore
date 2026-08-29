# Hand Appearance Re-ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the palm-only experiment with a bounded six-part full-hand
appearance signal that can use palm and individual finger evidence during hand
crossings. V1 compatibility is explicitly out of scope.

**Architecture:** `HandPipeline` produces an owned 256-float descriptor partitioned
into palm and five fingers from Host BGR/RGB input. `HandTrackIdentityRegistry`
owns bounded per-part EMA prototypes and combines comparable regions with existing
shape, motion, and tracker evidence. THIG consumes confirmed canonical IDs only.

**Tech Stack:** C++17, KFCore ImageView and vision model types, TinyTest, CMake Presets.

**Spec:** `docs/superpowers/specs/2026-08-29-hand-appearance-reid-design.md`

## Constraints

- [x] Remove the V1 kind/layout instead of maintaining a compatibility adapter.
- [x] Preserve shape-only matching when fewer than the configured common parts exist.
- [x] Fail fast for unsupported built-in extraction inputs and malformed descriptors.
- [x] Keep descriptor/state storage fixed and bounded; retain no image pointer.
- [x] Add no OpenCV, CUDA, ONNX, or unvalidated learned-model dependency.
- [x] Keep information-free exact occlusion ambiguous.

## Task 1: Lock the six-part contracts with failing tests

**Files:**
- Modify: `vision_models/tests/test_hand_pipeline.cpp`
- Modify: `hand_interaction/tests/test_primitive_extractor.cpp`
- Modify: `hand_interaction/tests/test_hand_interaction_demo_ui.cpp`

- [x] Assert palm plus five finger parts are produced with masks and quality.
- [x] Assert RGB/BGR equivalence and brightness stability.
- [x] Assert index-finger texture can change without changing the palm part.
- [x] Assert invalid image/config contracts fail fast.
- [x] Assert one distinctive finger rejects a crossing swap.
- [x] Assert an occluded part remains in the retained prototype.
- [x] Preserve exact-overlap ambiguity and UI/timing coverage.

## Task 2: Replace extraction and public descriptor layout

**Files:**
- Create: `vision_models/src/hand_appearance.hpp`
- Create: `vision_models/src/hand_appearance.cpp`
- Modify: `vision_models/include/kfcore/vision_models/types.hpp`
- Modify: `vision_models/src/pipeline.cpp`
- Modify: `vision_models/CMakeLists.txt`

- [x] Replace the kind-tagged 96-float layout with six fixed partitions totaling 256 floats.
- [x] Add valid-part mask and per-part coverage quality.
- [x] Sample a 6x8 palm grid and five 8x2 landmark-aligned strips.
- [x] Validate input before inference and extract before ByteTrack commit.
- [x] Pass focused vision pipeline tests.

## Task 3: Replace matching and retained prototype updates

**Files:**
- Modify: `hand_interaction/include/kfcore/hand_interaction/types.hpp`
- Modify: `hand_interaction/src/hand_identity.hpp`
- Modify: `hand_interaction/src/hand_identity.cpp`
- Modify: `hand_interaction/src/primitive_extractor.cpp`

- [x] Compare only common valid parts with quality-weighted mean distance.
- [x] Add worst-part compatibility evidence so a distinctive finger is not averaged away.
- [x] Keep gesture-driven appearance mismatch as bounded cost rather than vetoing shape/motion reacquisition.
- [x] Require a configurable minimum number of common parts.
- [x] Update accepted prototypes per part; retain missing parts unchanged.
- [x] Validate all new public options and malformed descriptor fields.
- [x] Pass focused canonical identity tests.

## Task 4: Demo and documentation

- [x] Keep demo extraction enabled for Host BGR capture on CPU/TensorRT backends.
- [x] Keep `appearance_ms` and `Match:Appearance` in the high-contrast overlay.
- [x] Document partition layout, CPU extraction, limits, and no-V1 migration.
- [ ] Rebuild and restart the local TensorRT demo after the old process is stopped.

## Task 5: Verification

- [x] Build and run focused CPU vision/identity tests.
- [x] Run the complete CPU preset suite (22/22).
- [x] Build and run the complete TensorRT preset suite (35/35).
- [ ] Record runtime `appearance_ms` and manually exercise a two-hand crossing.
- [ ] Run diff/CodeGraph checks and inspect public-layout impact before completion.
