# Hand-Shape Canonical Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep THIG canonical Hand IDs continuous only within a bounded temporal
retention window by matching invariant hand shape instead of treating raw tracker
IDs alone as identity.

**Architecture:** A small internal descriptor component converts the 21 labelled hand landmarks into 20 normalized 3D bone-length ratios. `HandTrackIdentityRegistry` owns retention-limited descriptor prototypes and uses shape plus known handedness as candidate gates; normalized spatial and ByteTrack evidence rank compatible candidates.

**Tech Stack:** C++17, `std::array`, KFCore vision model types, TinyTest, CMake Presets.

**Spec:** `docs/superpowers/specs/2026-08-28-hand-shape-identity-design.md`

## Final review correction (2026-08-29)

This correction supersedes every earlier statement in this plan that describes
identity as persistent until `reset()` or permits shape-only matching after the
reacquisition horizon. Canonical IDs are now retained only through
`reacquire_frames`; expiry is applied before matching so a later person cannot
inherit old THIG state. Within that window, normalized 3D shape and known
handedness are candidate gates; distance and scale are normalized, saturated
soft evidence. Same-top cross-observation competition below the ambiguity margin
returns `0` for every contender, independently of input order.

The default `maximum_shape_distance` is `0.20F` (calibrate it for the deployed
landmark noise). Final verification covers the bounded horizon, 3D-only shape
change, handedness gate, candidate competition, both legacy saturation ratios,
prototype update weight, and transactional capacity retry. Matching complexity
is `O(H * (20I + I log I))`, with staged copy and expiry pruning `O(I)`.
Candidate cost and ambiguity arithmetic use `double` so finite extreme public
weights do not overflow into order-dependent non-finite comparisons.

## Global Constraints

- `KFCore::hand_interaction` remains independent of OpenCV, CUDA, TensorRT, and ONNX Runtime.
- Invalid landmark or configuration input fails explicitly; there is no position-only fallback when the shape descriptor is invalid.
- Descriptor and identity state are bounded and single-thread-owned.
- Public structure layout changes require downstream consumers to rebuild.
- Existing raw Track ID and THIG observation formats remain unchanged.

---

### Task 1: Specify bounded shape-based identity behavior

**Files:**
- Modify: `hand_interaction/tests/test_primitive_extractor.cpp`

**Interfaces:**
- Consumes: existing `HandPrimitiveExtractor::process(const HandFrame&, const GestureFrameContext&)`
- Produces: behavioral contracts for in-window reacquisition, expiry, different-shape rejection, and distinct two-hand identity

- [ ] **Step 1: Add a complete labelled-hand fixture**

  Add a test-only `identity_hand()` fixture with all 21 finite landmarks and a
  `stretch_finger()` helper that changes literal bone proportions without using
  production descriptor code.

- [ ] **Step 2: Add bounded-horizon identity tests**

  ```cpp
  it("allocates a new identity after the retention horizon")
  {
      HandPrimitiveOptions options;
      options.identity.reacquire_frames = 2;
      HandPrimitiveExtractor extractor(options);
      // Establish one hand, consume more than the retention horizon, then
      // recreate its shape and raw ID. The final canonical ID must differ from
      // the literal ID returned by the first frame.
  }
  ```

  Run:

  ```powershell
  cmake --build --preset win-release-user --target test_hand_primitive_extractor
  ctest --preset win-release-user -C Release -R test_hand_primitive_extractor --output-on-failure
  ```

  Expected final behavior: the long-gap case receives a new ID; a separate
  in-window test retains the original ID.

- [ ] **Step 3: Add different-shape and two-hand swap tests**

  Assert that a materially stretched finger creates a new canonical ID, and
  that two hands with different literal bone proportions recover their original
  IDs inside the retention horizon even when their screen positions swap.

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

### Task 3: Make feature identity bounded and configurable

**Files:**
- Modify: `hand_interaction/include/kfcore/hand_interaction/types.hpp`
- Modify: `hand_interaction/src/hand_identity.hpp`
- Modify: `hand_interaction/src/hand_identity.cpp`
- Modify: `hand_interaction/src/primitive_extractor.cpp`
- Modify: `hand_interaction/tests/test_primitive_extractor.cpp`

**Interfaces:**
- Consumes: `HandShapeDescriptor`, handedness, and the three appended `HandIdentityOptions` fields
- Produces: retention-window canonical identity with bounded descriptor prototypes

- [ ] **Step 1: Add and validate public configuration**

  Append `maximum_shape_distance = 0.20F`, `shape_cost_weight = 2.0F`,
  `shape_update_weight = 0.20F`. Change the default `maximum_identities` to
  `32`. Add constructor tests for zero, negative,
  non-finite, and out-of-range values.

- [ ] **Step 2: Store and update descriptor prototypes**

  Add shape and handedness to `IdentityState`. On every accepted assignment,
  update the shape using the configured EMA weight and renormalize it; keep the
  first known handedness as stable categorical evidence.

- [ ] **Step 3: Use bounded time-and-feature candidate gating**

  Prune identities older than `reacquire_frames` before matching. Reject
  candidates when shape distance exceeds `maximum_shape_distance` or known
  handedness conflicts. Normalize and saturate spatial/scale costs within the
  retention window, then apply frame-wide ambiguity withholding.

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

  Describe descriptor invariants, the time-bounded `reacquire_frames` lifetime,
  saturation semantics, capacity failure, new defaults, calibration, and the
  downstream rebuild requirement.

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

- [ ] **Step 5: Commit implementation and documentation**

  Commit the implementation and documentation. Pushing and confirming a PR are
  separate publishing steps that require explicit authorization and are not
  performed by Task 4.
