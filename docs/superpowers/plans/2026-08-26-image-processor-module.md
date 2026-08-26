# ImageProcessor Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract KFCore's CUDA image-to-tensor preprocessing into a TensorRT-independent `KFCore::image_processor` module and migrate TensorRT YOLO to it without changing public YOLO behavior.

**Architecture:** Expose borrowed image/tensor/buffer views plus a stateless planner, host row packer, and CUDA enqueue facade. Callers own all memory and stream lifetimes. Keep a thin TensorRT adapter that maps existing YOLO types and errors to the generic module; remove the old duplicate letterbox implementation after parity tests pass.

**Tech Stack:** C++17, CUDA Runtime/CUDA C++17, CMake 3.20+, TurboUtils TinyTest, TensorRT 11.2.1.2 for downstream integration only.

**Spec:** `docs/superpowers/specs/2026-08-26-image-processor-design.md`

## Task 1: Define the public CPU contract with red tests

**Files:**
- Create: `image_processor/tests/test_image_processor.cpp`
- Create: `image_processor/CMakeLists.txt`
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`

- [x] Add TinyTest cases for letterbox transforms, batch planning, resource limits, enum validation, row-stride validation, output capacity, and padded host-row packing.
- [x] Add the smallest CMake target needed to compile the test against the not-yet-existing public API.
- [x] Configure/build the focused target and confirm it fails because the API/implementation is absent.

## Task 2: Implement the generic CPU contract

**Files:**
- Create: `image_processor/include/kfcore/image_processor/types.hpp`
- Create: `image_processor/include/kfcore/image_processor/error.hpp`
- Create: `image_processor/include/kfcore/image_processor/image_processor.hpp`
- Create: `image_processor/src/image_processor.cpp`
- Modify: `image_processor/CMakeLists.txt`

- [x] Implement checked byte arithmetic, exact tensor layout validation, per-image transforms and bounded workspace planning.
- [x] Implement padded host-row packing into caller-owned pinned workspace without retaining views.
- [x] Run the focused CPU test to green and then the adjacent non-GPU YOLO helper tests.

## Task 3: Specify and implement CUDA enqueue

**Files:**
- Create: `image_processor/tests/test_image_processor_cuda.cu`
- Create: `image_processor/src/image_processor_cuda.cu`
- Modify: `image_processor/CMakeLists.txt`

- [x] Port the existing CUDA parity cases as public ImageProcessor tests and add BGR Tensor output order plus mixed Host/CUDA batch coverage.
- [x] Build/run the red CUDA test before connecting the implementation.
- [x] Implement async host upload and fused bilinear letterbox, channel conversion, normalization and NCHW FP16/FP32 output on the caller stream.
- [x] Run CPU and CUDA ImageProcessor tests to green.

## Task 4: Migrate TensorRT YOLO through the adapter

**Files:**
- Modify: `tensorrt_yolo/src/detector.cpp`
- Modify: `tensorrt_yolo/src/detector_helpers.hpp`
- Modify: `tensorrt_yolo/src/detector_helpers.cpp`
- Modify: `tensorrt_yolo/src/tensorrt_raii.hpp`
- Modify: `tensorrt_yolo/src/engine.cpp`
- Modify: `tensorrt_yolo/CMakeLists.txt`
- Delete: `tensorrt_yolo/src/letterbox.hpp`
- Delete: `tensorrt_yolo/src/letterbox.cu`
- Delete: `tensorrt_yolo/tests/test_letterbox_cuda.cu`

- [x] Add conversion and error mapping at the TensorRT adapter boundary.
- [x] Replace YOLO planning/staging/kernel calls with the generic processor while retaining YOLO batch/profile/engine limits.
- [x] Preserve transform-based NMS coordinate restoration and the existing detector call-lifetime contract.
- [ ] Run YOLO helper, CUDA, tracking, OpenCV and TensorRT integration tests.

The helper, CUDA, tracking, OpenCV, package-consumer and non-opt-in TensorRT tests passed (19/19). The
engine-backed integration executable was not registered because no local trusted `.engine` remains
after the requested reference-directory cleanup.

## Task 5: Package, document, and verify

**Files:**
- Modify: `cmake/KFCoreConfig.cmake.in`
- Modify: `CMakeUserPresets.json`
- Modify: `tensorrt_yolo/README.md`
- Modify: `docs/superpowers/specs/2026-08-25-tensorrt-yolo-bytetrack-design.md`

- [x] Export/install `KFCore::image_processor`, advertise package capability, and make CUDAToolkit a package dependency whenever the module is present.
- [x] Add a standalone Windows user preset while keeping the existing TensorRT preset functional.
- [x] Document API/lifetime rules, TensorRT integration, supported formats, and AprilTag extension boundary.
- [x] Add and run an isolated installed-package consumer for `KFCore::image_processor`.
- [x] Sync CodeGraph, inspect affected symbols, run focused tests followed by the relevant release CTest suite, and review the final diff.
