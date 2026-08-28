# Hand-Shape Canonical Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep THIG canonical Hand IDs stable by matching a persistent, invariant hand-shape descriptor instead of treating frame age and position as identity.

**Architecture:** A small internal descriptor component converts the 21 labelled hand landmarks into 20 normalized 3D bone-length ratios. `HandTrackIdentityRegistry` owns persistent descriptor prototypes and uses shape as the candidate gate; recent spatial and ByteTrack evidence only rank compatible candidates.

**Tech Stack:** C++17, `std::array`, KFCore vision model types, TinyTest, CMake Presets.

**Spec:** `docs/superpowers/specs/2026-08-28-hand-shape-identity-design.md`

## Global Constraints

- `KFCore::hand_interaction` remains independent of OpenCV, CUDA, TensorRT, and ONNX Runtime.
- Invalid landmark or configuration input fails explicitly; there is no position-only fallback when the shape descriptor is invalid.
- Descriptor and identity state are bounded and single-thread-owned.
- Public structure layout changes require downstream consumers to rebuild.
- Existing raw Track ID and THIG observation formats remain unchanged.

---

### Task 1: Specify persistent shape-based identity behavior

**Files:**
- Modify: `hand_interaction/tests/test_primitive_extractor.cpp`

**Interfaces:**
- Consumes: existing `HandPrimitiveExtractor::process(const HandFrame&, const GestureFrameContext&)`
- Produces: behavioral contracts for invariant reacquisition, different-shape rejection, and persistent two-hand identity

- [ ] **Step 1: Add a complete labelled-hand fixture**

  Add a test-only `identity_hand()` fixture with all 21 finite landmarks and a
  `stretch_finger()` helper that changes literal bone proportions without using
  production descriptor code.

- [ ] **Step 2: Add the failing long-gap invariant test**

  ```cpp
  it("reacquires the same hand by shape after spatial evidence expires")
  {
      HandPrimitiveOptions options;
      options.identity.reacquire_frames = 2;
      HandPrimitiveExtractor extractor(options);
      // Establish one hand, consume three empty frames, then translate, scale,
      // rotate, and recreate its raw ID. The final canonical ID must equal the
      // literal ID returned by the first frame.
  }
  ```

  Run:

  ```powershell
  cmake --build --preset win-hand-interaction-demo-cpu-release-user --target test_hand_primitive_extractor
  ctest --preset win-hand-interaction-demo-cpu-release-user -C Release -R test_hand_primitive_extractor --output-on-failure
  ```

  Expected: FAIL because the current registry deletes the first identity after
  the configured frame horizon.

- [ ] **Step 3: Add different-shape and two-hand swap tests**

  Assert that a materially stretched finger creates a new canonical ID, and
  that two hands with different literal bone proportions recover their original
  IDs after the spatial horizon even when their screen positions swap.

- [ ] **Step 4: Add invalid-shape behavior tests**

  Assert that a non-finite non-palm landmark and a fully degenerate landmark set
  both produce canonical ID `0` rather than falling back to location.

---

### Task 2: Add the invariant hand-shape descriptor

**Files:**
- Create: `hand_interaction/src/hand_shape_descriptor.hpp`
- Create: `hand_interaction/src/hand_shape_descriptor.cpp`
- Modify: `hand_interaction/CMakeLists.txt`
- Modify: `hand_interaction/src/hand_identity.hpp`
- Modify: `hand_interaction/src/primitive_extractor.cpp`

**Interfaces:**
- Consumes: `std::array<vision_models::HandLandmark, vision_models::kHandLandmarkCount>`
- Produces: `std::optional<HandShapeDescriptor> MakeHandShapeDescriptor(...)` and `float HandShapeDistance(const HandShapeDescriptor&, const HandShapeDescriptor&)`

- [ ] **Step 1: Define the fixed descriptor value**

  ```cpp
  inline constexpr std::size_t kHandShapeFeatureCount = 20;

  struct HandShapeDescriptor {
    std::array<float, kHandShapeFeatureCount> values {};
  };
  ```

- [ ] **Step 2: Implement validated descriptor construction**

  Use the five literal MediaPipe chains, compute each 3D edge length, reject
  non-finite inputs or a non-positive total, and normalize all values by the
  total length. Implement L1 distance without allocation.

- [ ] **Step 3: Pass shape and handedness into identity observations**

  Extend the internal observation value with an optional descriptor and
  `vision_models::Handedness`. `primitive_extractor.cpp` computes it once per
  hand and passes it with the existing palm geometry.

- [ ] **Step 4: Build and run the focused test**

  Expected: the test compiles but the long-gap case still fails until Task 3;
  invalid descriptor cases now return ID `0`.

---

### Task 3: Make feature identity persistent and configurable

**Files:**
- Modify: `hand_interaction/include/kfcore/hand_interaction/types.hpp`
- Modify: `hand_interaction/src/hand_identity.hpp`
- Modify: `hand_interaction/src/hand_identity.cpp`
- Modify: `hand_interaction/src/primitive_extractor.cpp`
- Modify: `hand_interaction/tests/test_primitive_extractor.cpp`

**Interfaces:**
- Consumes: `HandShapeDescriptor`, handedness, and the four appended `HandIdentityOptions` fields
- Produces: persistent canonical identity with bounded descriptor prototypes

- [ ] **Step 1: Add and validate public configuration**

  Append `maximum_shape_distance = 0.35F`, `shape_cost_weight = 2.0F`,
  `shape_update_weight = 0.20F`, and
  `handedness_mismatch_penalty = 0.35F`. Change the default
  `maximum_identities` to `32`. Add constructor tests for zero, negative,
  non-finite, and out-of-range values.

- [ ] **Step 2: Store and update descriptor prototypes**

  Add shape and handedness to `IdentityState`. On every accepted assignment,
  update the shape using the configured EMA weight and renormalize it; keep the
  first known handedness as stable categorical evidence.

- [ ] **Step 3: Replace time deletion with feature candidate gating**

  Remove identity pruning. Reject candidates only when shape distance exceeds
  `maximum_shape_distance`. Apply recent spatial/scale cost only while age is at
  most `reacquire_frames`; add shape, handedness, capped age, and raw-ID evidence
  to the existing deterministic assignment cost.

- [ ] **Step 4: Verify GREEN and refactor**

  Run the focused test and confirm every new behavior passes. Remove duplicated
  normalization or distance code while keeping descriptor responsibilities in
  the new internal component.

---

### Task 4: Document, verify, review, and publish

**Files:**
- Modify: `hand_interaction/README.md`
- Modify: `docs/design/2026-08-28-thig-hand-interaction.md`

**Interfaces:**
- Consumes: completed public configuration and tested behavior
- Produces: migration notes, reproducible verification, and the updated PR branch

- [ ] **Step 1: Document behavior and compatibility**

  Describe the descriptor invariants, persistent-until-reset lifetime,
  `reacquire_frames` spatial-horizon semantics, capacity failure, new defaults,
  and the downstream rebuild requirement.

- [ ] **Step 2: Run focused and full CPU verification**

  Build `test_hand_primitive_extractor`, run its CTest filter, build the CPU
  preset, then run all CPU tests with `--output-on-failure`.

- [ ] **Step 3: Run focused and full TensorRT verification**

  Set `TENSORRT_ROOT=C:\projects\TensorRT-11.2.1.2`, build the TensorRT preset,
  run the focused identity test, then run the complete TensorRT CTest preset.

- [ ] **Step 4: Review performance and correctness evidence**

  Confirm `git diff --check`, inspect the public API diff, verify the descriptor
  adds no heap allocation or OpenCV dependency, and request a read-only code
  review focused on identity swaps, ambiguity, capacity, and invalid input.

- [ ] **Step 5: Commit and push**

  Commit the implementation and documentation, push
  `feature/thig-hand-interaction-demo`, and confirm PR #19 targets `master`.
