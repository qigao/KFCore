# Packed YUV Pixel Formats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add decoded `Nv21`, `Yuy2`, and `Uyvy` frame input across CPU/CUDA ImageProcessor and its TensorRT application adapters.

**Architecture:** Append formats to the existing single-pointer `ImageView` contract. Central layout validation packs padded sources, while CPU and CUDA samplers decode the three layouts with the existing BT.601 limited-range transform. Downstream modules only map and forward the formats.

**Tech Stack:** C++17, CUDA, TensorRT, CMake Presets, TinyTest

**Spec:** `docs/superpowers/specs/2026-08-29-packed-yuv-pixel-formats-design.md`

## Global Constraints

- Do not add MJPEG, H.264, containers, timestamps, multi-plane pointers, or OpenCV core dependencies.
- Preserve existing `ImageView` field layout and append enum values only.
- Preserve byte-for-byte NV12/I420 conversion behavior.
- Use literal expected pixels and observe each new test fail before production edits.
- Continue in the current staged master workspace by explicit user direction; do not commit automatically.

---

### Task 1: CPU format contract and conversion

**Files:**
- Modify: `image_processor/tests/test_cpu_image_processor.cpp`
- Modify: `image_processor/tests/test_image_processor.cpp`
- Modify: `image_processor/include/kfcore/image_processor/types.hpp`
- Modify: `image_processor/src/image_layout.hpp`
- Modify: `image_processor/src/cpu.cpp`
- Modify: `image_processor/src/image_processor.cpp`

**Interfaces:**
- Produces: `PixelFormat::Nv21`, `PixelFormat::Yuy2`, `PixelFormat::Uyvy`
- Preserves: `CpuImageProcessor::copy_bgr()` and `letterbox_nchw()` signatures

- [x] Add literal red-pixel fixtures: NV21 `{81,81,81,81,240,90}`, YUY2
  `{81,90,81,240}`, and UYVY `{90,81,240,81}`; expect BGR `{0,0,255}` pixels.
- [x] Run `test_cpu_image_processor` and confirm compilation fails because the enum values do not exist.
- [x] Append the enum values and implement even-dimension/stride/capacity validation and CPU sampling.
- [x] Run `test_cpu_image_processor` and `test_image_processor` and confirm both pass.

### Task 2: CUDA staging and fused preprocessing

**Files:**
- Modify: `image_processor/tests/test_image_processor_cuda.cu`
- Modify: `image_processor/src/image_processor_cuda.cu`

**Interfaces:**
- Consumes: the three new `PixelFormat` values and contiguous layout rules from Task 1
- Produces: Host/CUDA staging and FP16/FP32 affine preprocessing for all three formats

- [x] Add literal padded Host staging fixtures and CUDA/CPU tensor-parity cases for NV21, YUY2,
  and UYVY.
- [x] Build `test_image_processor_cuda` and confirm failure in unsupported-format validation.
- [x] Extend staging row sizes, YUV sampling, affine processing, and composition-base validation.
- [x] Run `test_image_processor_cuda` and confirm the new fixtures and prior tests pass.

### Task 3: TensorRT and application adapters

**Files:**
- Modify: `tensorrt_yolo/include/kfcore/yolo/types.hpp`
- Modify: `tensorrt_yolo/src/detector_helpers.cpp`
- Modify: `tensorrt_yolo/tests/test_tensorrt_detection_helpers.cpp`
- Modify: `face_applications/src/tensorrt.cpp`
- Modify: `vision_models/src/tensorrt.cpp`
- Modify: `vision_models/src/hand_appearance.cpp`
- Modify: corresponding Vision Models validation tests

**Interfaces:**
- Consumes: packed single-frame layouts from Tasks 1-2
- Produces: format-preserving TensorRT-YOLO plans and Face/Vision model acceptance

- [x] Add planning and validation tests using the three new format values.
- [x] Build the tests and confirm adapter switches reject or fail to compile for the new formats.
- [x] Append YOLO enum values, calculate legacy capacities, and update all forwarding switches.
- [x] Run TensorRT helper, Vision Models, and Face Application validation tests.

### Task 4: Documentation and full verification

**Files:**
- Modify: `image_processor/README.md`
- Modify: `face_applications/README.md`
- Modify: `vision_models/README.md`

**Interfaces:**
- Documents: decoded pixel formats only; codecs remain outside ImageProcessor

- [x] Document layout, even-width constraints, Host/CUDA support, and the fixed BT.601 conversion.
- [x] Run the CPU preset tests, then `win-face-applications-release-user` build and CTest suite.
- [x] Run `git diff --cached --check`, verify no unintended unstaged files, and stage only this task's files.
