# THIG Hand Interaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add backend-neutral complex hand-gesture interpretation to KFCore by migrating Retro THIG and adapting tracked Palm/21-point results into temporal actions.

**Architecture:** `KFCore::thig` is a standalone C++17 temporal graph. `KFCore::hand_interaction` owns bounded primitive histories and composes their observations through THIG; existing CPU/TensorRT inference and ByteTrack/Kalman remain unchanged.

**Tech Stack:** C++17, CMake package targets, TurboUtils TinyTest, existing KFCore vision model types.

**Spec:** `docs/design/2026-08-28-thig-hand-interaction.md`

## Global Constraints

- Preserve all existing `vision_models` public behavior and backend contracts.
- Treat KFCore `track_id == 0` as valid and `track_id < 0` as untracked.
- All retained temporal state is single-owner and bounded by time and count.
- CPU and TensorRT paths share exactly one primitive/THIG implementation.
- No OpenCV, ONNX Runtime, TensorRT, CUDA, Camera, Gallery, or UI dependency may enter `KFCore::thig`.
- Invalid configuration, non-monotonic input, and exhausted capacity fail explicitly without partial frame commit.

---

### Task 1: Migrate the generic temporal graph

**Files:**
- Create: `thig/include/kfcore/thig/temporal_graph.hpp`
- Create: `thig/src/temporal_graph.cpp`
- Create: `thig/tests/test_temporal_graph.cpp`
- Create: `thig/CMakeLists.txt`
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`
- Modify: `cmake/KFCoreConfig.cmake.in`

**Interfaces:**
- Produces: `kfcore::thig::TemporalGraphEngine`, `EngineSpec`, `Observation`, `ActionEvent`, pattern and state-graph specifications.
- Contract: `ProcessFrame(const std::vector<Observation>&, steady_clock::time_point)` validates monotonic time and copies returned evidence.

```cpp
kfcore::thig::TemporalGraphEngine engine(spec);
const auto actions = engine.ProcessFrame(
    {kfcore::thig::Observation{1, {"hand", 1}, std::nullopt,
                               "Shape Open", kfcore::thig::TruthValue::True}},
    std::chrono::steady_clock::time_point{});
```

- [ ] **Step 1: Add the migrated TinyTest behavior suite and CMake test target before production files exist.**
- [ ] **Step 2: Configure/build `test_thig_temporal_graph` and verify RED because the public header/target implementation is absent.**
- [ ] **Step 3: Port the Retro header and implementation under namespace `kfcore::thig`, preserving validation and bounded stores.**
- [ ] **Step 4: Build and run `ctest -R '^test_thig_temporal_graph$'`; verify all migrated cases pass.**
- [ ] **Step 5: Commit the independently usable `KFCore::thig` target.**

### Task 2: Extract bounded hand primitives

**Files:**
- Create: `hand_interaction/include/kfcore/hand_interaction/types.hpp`
- Create: `hand_interaction/include/kfcore/hand_interaction/primitive_extractor.hpp`
- Create: `hand_interaction/src/primitive_extractor.cpp`
- Create: `hand_interaction/tests/test_primitive_extractor.cpp`
- Create: `hand_interaction/CMakeLists.txt`

**Interfaces:**
- Consumes: `const vision_models::HandFrame&`, `GestureFrameContext { serial, timestamp, image_width, image_height }`.
- Produces: `PrimitiveFrame { canonical_hands, std::vector<thig::Observation> observations }`.
- State owner: each `HandPrimitiveExtractor` instance owns identity and geometry histories; `reset()` clears them atomically.

```cpp
struct GestureFrameContext {
  std::uint64_t serial;
  std::chrono::steady_clock::time_point timestamp;
  int image_width;
  int image_height;
};

PrimitiveFrame HandPrimitiveExtractor::process(
    const vision_models::HandFrame& frame,
    const GestureFrameContext& context);
```

The first geometry test uses a hand with `track_id = 0` and
`gesture = Gesture::Closed`, then asserts the literal relation
`"Shape Fist"` has source entity ID `1`.

- [ ] **Step 1: Write literal landmark tests for track ID zero, Closed-to-Fist mapping, V, OK, and index articulation.**
- [ ] **Step 2: Build the test and verify RED because `HandPrimitiveExtractor` is absent.**
- [ ] **Step 3: Implement stateless shape/pose geometry and positive canonical-ID mapping; run the focused tests GREEN.**
- [ ] **Step 4: Write failing timestamped tests for direction, stationarity, scale, rotation, two-hand distance, pruning, reset, and capacity rejection.**
- [ ] **Step 5: Implement bounded per-identity and pair histories with monotonic validation; run focused and vision pipeline regression tests GREEN.**
- [ ] **Step 6: Add ambiguity tests for identity reassociation and implement conservative canonical rekeying that emits no evidence when candidates cannot be distinguished.**
- [ ] **Step 7: Commit the backend-neutral primitive extractor.**

### Task 3: Compose semantic hand actions

**Files:**
- Create: `hand_interaction/include/kfcore/hand_interaction/hand_interaction.hpp`
- Create: `hand_interaction/src/gesture_graph.cpp`
- Create: `hand_interaction/src/hand_interaction.cpp`
- Create: `hand_interaction/tests/test_hand_interaction.cpp`

**Interfaces:**
- Consumes: Task 2 `PrimitiveFrame` and optional externally owned observations copied at the call boundary.
- Produces: `HandInteractionFrame { primitives, actions }` through `HandInteractionPipeline::process()`.
- Graph actions: `Wave`, `Grasp`, `Release`, directional drag, `OK`, `Single Hand V`, `Two Hand V`, `Zoom In`, `Zoom Out`, and experimental rotation events.

```cpp
HandInteractionFrame HandInteractionPipeline::process(
    const vision_models::HandFrame& frame,
    const GestureFrameContext& context,
    const std::vector<thig::Observation>& external_observations = {});
```

The Wave fixture supplies one canonical hand with stable `Shape Open` and the
literal direction sequence Left, Right, Left at 100 ms intervals; it expects
exactly one action named `"Wave"` and no navigation command.

- [ ] **Step 1: Write failing deterministic sequence tests for Wave, Grasp/Release, OK, dual-hand V, Zoom, and Rotate.**
- [ ] **Step 2: Implement `BuildHandInteractionGraph()` using THIG patterns/state graphs and make the focused tests GREEN.**
- [ ] **Step 3: Write failing facade tests for monotonic context, external evidence copy, action evidence lifetime, and whole-session reset.**
- [ ] **Step 4: Implement `HandInteractionPipeline` as the single owner of extractor and graph state; run all hand-interaction tests GREEN.**
- [ ] **Step 5: Commit semantic graph composition and facade.**

### Task 4: Package, document, and verify both backends

**Files:**
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`
- Modify: `CMakeUserPresets.json`
- Modify: `cmake/KFCoreConfig.cmake.in`
- Modify: `vision_models/README.md`
- Create: `hand_interaction/README.md`
- Modify: `cmake/tests/vision_models_consumer/CMakeLists.txt`

**Interfaces:**
- Produces: installed `KFCore::thig` and `KFCore::hand_interaction` targets and `KFCore_HAS_THIG` / `KFCore_HAS_HAND_INTERACTION` package flags.

```cmake
find_package(KFCore CONFIG REQUIRED)
target_link_libraries(consumer PRIVATE
  KFCore::thig
  KFCore::hand_interaction)
```

- [ ] **Step 1: Add package/preset consumer assertions and verify RED against the not-yet-exported targets.**
- [ ] **Step 2: Add options, dependency validation, exports, installed headers, and CPU/TensorRT preset enablement.**
- [ ] **Step 3: Document the ownership contract, supported actions, unsupported application mappings, configuration, and CPU/GPU shared flow.**
- [ ] **Step 4: Run focused THIG/hand-interaction tests, existing vision core tests, CPU real-model integration, TensorRT real-engine integration, and installed-consumer tests.**
- [ ] **Step 5: Inspect `git diff --check`, CodeGraph affected tests, and clean status; commit packaging/documentation.**
