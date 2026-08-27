# GPU Image Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox status for resumable execution.

**Goal:** Keep the TensorRT InSwapper and optional GFPGAN post-processing chain on one CUDA device and download only the final packed BGR image, while exposing reusable image-processing primitives rather than face-model-specific CUDA code.

**Architecture:** `CudaImageProcessor` owns bounded reusable device buffers for preprocessing tensors, caller-writable inference output tensors, masks, and one composited image. Face-model adapters keep validating their model contracts and gain `infer_into` overloads that write into caller-owned host or CUDA views through the existing TensorRT executor. The face application creates the existing OpenCV static masks, passes them as host FP32 tensor views, performs generic affine alpha composition on the GPU, and downloads once into the final `cv::Mat`. Existing owning host-returning inference and CPU composition APIs remain unchanged.

**Tech Stack:** C++17, CUDA Runtime/CUDA C++, TensorRT 11, OpenCV Lite, CMake Presets, TurboUtils TinyTest.

**Spec:** User-approved design: generic GPU image-processing path plus device-resident InSwapper/GFPGAN final stages; CPU and public host paths remain available.

## Global Constraints

- Preserve existing public behavior and keep current `infer()` methods source-compatible.
- Fail fast on malformed tensors, images, transforms, device mismatches, aliasing, and configured capacity overruns.
- One `CudaImageProcessor` is synchronous and non-reentrant. Returned CUDA views borrow processor-owned storage with explicitly documented invalidation.
- Do not expose OpenCV or face-model types from `image_processor`.
- Do not add implicit CPU fallback. The GPU application requires all chained TensorRT models to use the same CUDA device.
- Use the existing TensorRT `Executor::run(inputs, outputs)` device-output support; do not duplicate runtime execution logic.

---

## Task 1: Define failing contracts for reusable CUDA output and composition

- [x] Add TinyTest cases in `image_processor/tests/test_image_processor_cuda.cu` for bounded writable FP32 NCHW output allocation, identity affine alpha composition, in-place second composition, final packed BGR download, and malformed/capacity errors.
- [x] Add adapter API/contract tests in `face_models/tests/test_swap_adapter_api.cpp` for the new `infer_into` signatures and output validation helpers where testable without loading an engine.
- [x] Configure/build the narrow test targets and record the expected compile/test failure before implementation.

## Task 2: Implement generic `CudaImageProcessor` device primitives

- [x] Extend `image_processor` types with generic composite options only where required.
- [x] Add a reusable writable tensor acquisition API, generic affine RGB-CHW + FP32 alpha composition API, and caller-buffer download API.
- [x] Implement validation, checked byte arithmetic, same-device checks, host-mask staging, CUDA bilinear sampling, fused alpha/strength blend, and synchronized error cleanup.
- [x] Keep source, preprocess tensor, inference tensor, mask, and composited image in separately owned bounded allocations.
- [x] Run the CUDA image-processor tests.

## Task 3: Add caller-owned output inference to face-model adapters

- [x] Add `TensorRtInSwapper::infer_into` and `TensorRtGfpGan::infer_into` without changing existing `infer()` behavior.
- [x] Reuse adapter input validation, validate exact output names/types/shapes/capacities, and call the existing executor with the caller view.
- [x] Run face-model contract/API tests and real integration tests when configured models are available.

## Task 4: Integrate the one-download TensorRT face-swap chain

- [x] Validate at application load that InSwapper and optional GFPGAN use the same CUDA device.
- [x] Cache the existing OpenCV-generated 128 and optional 512 masks in application state.
- [x] Replace host decode/paste/restage with writable CUDA output tensors and generic GPU affine composition.
- [x] Feed the first composited CUDA image directly to GFPGAN preprocessing and download only the final BGR image into its owning `cv::Mat`.
- [x] Preserve input immutability, output dimensions, enhancer strength semantics, and error translation.

## Task 5: Verify behavior and performance reporting

- [x] Build the narrow CUDA, face-model, face-application, example, and benchmark targets using the repository preset environment.
- [x] Run adjacent CTest suites plus real-model face swap with and without GFPGAN.
- [x] Run the face-swap benchmark and update the existing performance document with measured transfer/composition/total timing and explicit hardware/model inputs.
- [x] Inspect the final diff and status; report verified scope and residual risks without committing or pushing unless requested.
