# Shared Vision Frame Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CPU and TensorRT vision orchestration submit each frame once and share immutable image views across hand and FaceMesh computation.

**Architecture:** `VisionFrameView` carries the authoritative source view and the backend compute view without owning pixels. `TensorRtVisionInput` performs one upload per frame, while TensorRT backends borrow CUDA compute views directly instead of staging them again; legacy `ImageView` overloads remain source-compatible.

**Tech Stack:** C++17, KFCore ImageProcessor, ONNX Runtime, TensorRT/CUDA, TinyTest, CMake Presets.

**Spec:** `docs/superpowers/specs/2026-08-29-shared-vision-frame-input-design.md`

## Global Constraints

- Preserve all existing `process(ImageView)` entry points and result types.
- Borrowed views never outlive the synchronous pipeline call or the next `TensorRtVisionInput::prepare()`.
- Reject mismatched source/compute frame metadata; never fall back from CUDA to CPU.
- Do not share mutable preparer or pipeline state across threads.

---

### Task 1: Route source and compute views through vision pipelines

**Files:**
- Modify: `vision_models/include/kfcore/vision_models/core.hpp`
- Modify: `vision_models/src/pipeline.cpp`
- Modify: `vision_models/tests/test_hand_pipeline.cpp`

**Interfaces:**
- Produces: `VisionFrameView::borrow(const image::ImageView&)`, `HandPipeline::process(const VisionFrameView&)`, and `FaceMeshPipeline::process(const VisionFrameView&)`.

- [x] **Step 1: Write failing TinyTest cases** proving hand inference receives `compute`, appearance reads `source`, both FaceMesh stages receive the same `compute`, and mismatched metadata is rejected.
- [x] **Step 2: Build and run `test_hand_pipeline`** and confirm compilation fails because `VisionFrameView` and overloads are absent.
- [x] **Step 3: Add the borrowed view type, validation, overloads, and legacy delegating entry points.**
- [x] **Step 4: Rebuild and run `test_hand_pipeline`** and confirm all cases pass.

### Task 2: Prepare TensorRT input once and avoid CUDA restaging

**Files:**
- Modify: `vision_models/include/kfcore/vision_models/tensorrt.hpp`
- Modify: `vision_models/src/tensorrt.cpp`
- Create: `vision_models/src/tensorrt_image_source.hpp`
- Modify: `vision_models/tests/test_tensorrt_backend_validation.cpp`
- Modify: `vision_models/CMakeLists.txt`

**Interfaces:**
- Consumes: `VisionFrameView` from Task 1.
- Produces: `TensorRtVisionInput::create(options)` and `prepare(source)`; internal `stage_host_or_borrow_cuda()`.

- [x] **Step 1: Write failing TinyTest cases** proving CUDA input bypasses the stage callback and host input invokes it exactly once.
- [x] **Step 2: Build and run the TensorRT validation test** and confirm compilation fails because the source-selection helper is absent.
- [x] **Step 3: Implement the helper and `TensorRtVisionInput`, then use the helper in hand and face-landmark backends.**
- [x] **Step 4: Rebuild and run TensorRT validation and integration tests** with the configured TensorRT/CUDA roots.

### Task 3: Use one prepared frame in KFCore and Retro orchestration

**Files:**
- Modify: `hand_interaction/examples/hand_interaction_demo.cpp`
- Modify: `hand_interaction/README.md`
- Modify: `Retro/vision/hand_gesture/src/interaction_runtime.cpp`
- Modify: `Retro/vision/detection/src/detection_runtime.cpp`

**Interfaces:**
- Consumes: `VisionFrameView` and `TensorRtVisionInput` from Tasks 1-2.
- Produces: one frame preparation per synchronous CPU/TensorRT runtime invocation.

- [x] **Step 1: Change the demo and Retro runtimes** so CPU calls `VisionFrameView::borrow()` and TensorRT calls one owner-local `TensorRtVisionInput::prepare()` before all same-frame model stages.
- [x] **Step 2: Update the ownership documentation** to state the exact invalidation point and remove the obsolete claim that TensorRT detector and landmarker cannot share the uploaded image.
- [x] **Step 3: Build the KFCore combined CPU/TensorRT/THIG preset and the main Retro Release target.**
- [x] **Step 4: Run focused tests, then the full KFCore and main-repository CTest suites; run `git diff --check` and sync both CodeGraph indexes.**
