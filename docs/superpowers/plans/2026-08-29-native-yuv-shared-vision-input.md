# Native YUV Shared Vision Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve captured NV12/I420 through shared CPU/TensorRT vision inference and materialize BGR only for display.

**Architecture:** `CapturedFrame` remains the sole host owner. A borrowed `ImageView` feeds CPU directly or is packed/uploaded once by `TensorRtHandInput`; all same-frame TensorRT stages borrow that CUDA view. Hand appearance samples host YUV directly, while the demo performs an explicit post-inference BGR conversion for UI only.

**Tech Stack:** C++17, CUDA, TensorRT, Turbo Capture, KFCore ImageProcessor, TinyTest, CMake Presets.

**Spec:** `docs/superpowers/specs/2026-08-29-shared-vision-frame-input-design.md`

### Task 1: Support packed YUV in CUDA shared staging and affine preprocessing

**Files:** `image_processor/src/image_processor_cuda.cu`, `image_processor/tests/test_image_processor_cuda.cu`

- [x] Add failing NV12/I420 stage and affine equivalence tests.
- [x] Implement validated plane-aware packing for host/device inputs.
- [x] Run the focused CUDA image-processor test.

### Task 2: Preserve YUV through TensorRT models and hand appearance

**Files:** `vision_models/src/tensorrt_face.cpp`, `vision_models/src/hand_appearance.cpp`, `vision_models/tests/test_hand_pipeline.cpp`, `vision_models/tests/test_tensorrt_backend_validation.cpp`

- [x] Add failing tests for YUV appearance and shared TensorRT input preparation.
- [x] Map NV12/I420 to TensorRT-YOLO and sample appearance directly from host YUV.
- [x] Run focused vision model tests.

### Task 3: Make capture frames the direct inference source

**Files:** `hand_interaction/examples/hand_interaction_demo_capture.*`, `hand_interaction/examples/hand_interaction_demo.cpp`, `hand_interaction/tests/test_hand_interaction_demo_capture.cpp`

- [x] Add failing capture-to-ImageView mapping tests.
- [x] Infer from RGB24/NV12/I420 borrowed capture storage and defer BGR conversion until display.
- [x] Build the demo and run focused capture/UI tests.

### Task 4: Verify behavior and performance boundary

- [x] Build the combined CPU/TensorRT/THIG preset.
- [x] Run focused tests and the full configured CTest suite.
- [x] Run `git diff --check`, sync CodeGraph, and report measured timing scope without extrapolation.
