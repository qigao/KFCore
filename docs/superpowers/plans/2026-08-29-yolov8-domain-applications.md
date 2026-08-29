# YOLOv8 Domain Applications Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement CPU/GPU applications for the local YOLOv8 drone, football, and parking models with Turbo Capture, ByteTrack/Kalman, readable domain overlays, and bounded performance metrics.

**Architecture:** Application-private adapters feed the existing YOLO detection/tracking contract. ONNX Runtime CPU and the existing TensorRT detector share one internal compact-NMS geometry/decoder target; source capture, domain policy, UI, and orchestration stay separate.

**Tech Stack:** C++17, ONNX Runtime CPU, TensorRT 11.2, CUDA 12.8, KFCore ImageProcessor and ByteTrack/Kalman, TurboParser Capture, OpenCV Lite, TurboUtils TinyTest, CMake Presets.

**Spec:** `docs/superpowers/specs/2026-08-29-yolov8-domain-applications-design.md`

## Constraints

- Preserve all existing public APIs and user-visible detector/tracker behavior.
- No automatic backend fallback, model discovery, MJPEG decode, or invented drone class names.
- Model and engine files remain external and ignored.
- Every variable-size frame/model/tensor/path/statistic collection has a configured bound.
- Implement and verify inline; the user explicitly selected inline mode.

### Task 1: Domain profiles and summaries

**Files:**
- Create: `tensorrt_yolo/examples/yolo_domain_profile.hpp`
- Create: `tensorrt_yolo/examples/yolo_domain_profile.cpp`
- Create: `tensorrt_yolo/tests/test_yolo_domain_profile.cpp`
- Modify: `tensorrt_yolo/CMakeLists.txt`

- [x] Write tests for three exact label contracts, parsing, score filtering, unknown-class failure,
  football counts, and parking occupancy.
- [x] Build the focused test and confirm RED because the profile module is absent.
- [x] Implement the minimum profile/filter/summary logic and confirm GREEN.

### Task 2: Strict ONNX Runtime CPU detector

**Files:**
- Create: `tensorrt_yolo/examples/yolo_domain_onnx.hpp`
- Create: `tensorrt_yolo/examples/yolo_domain_onnx.cpp`
- Create: `tensorrt_yolo/tests/test_yolo_domain_onnx.cpp`
- Create: `tensorrt_yolo/tests/test_yolo_domain_onnx_integration.cpp`
- Modify: `tensorrt_yolo/CMakeLists.txt`

- [x] Move `detector_helpers.cpp` into one internal static target and keep existing tests GREEN.
- [x] Write CPU adapter validation tests for paths, options, host-only images, and reentrancy.
- [x] Confirm RED, then implement exact metadata validation, bounded preprocessing/run/output decode.
- [x] Run all three real ONNX models and validate finite, bounded, legal-class results.

### Task 3: CLI and camera source boundary

**Files:**
- Create: `tensorrt_yolo/examples/yolo_domain_cli.hpp`
- Create: `tensorrt_yolo/examples/yolo_domain_cli.cpp`
- Create: `tensorrt_yolo/examples/yolo_domain_capture.hpp`
- Create: `tensorrt_yolo/examples/yolo_domain_capture.cpp`
- Create: `tensorrt_yolo/tests/test_yolo_domain_cli.cpp`
- Create: `tensorrt_yolo/tests/test_yolo_domain_capture.cpp`
- Modify: `tensorrt_yolo/CMakeLists.txt`

- [x] Test mutually exclusive sources, compiled backend availability, explicit model paths, and
  bounded numeric options.
- [x] Test exact 640x480@30 NV12 selection, unsupported/MJPEG rejection, packed-byte overflow,
  latest-frame coalescing, and close wakeup.
- [x] Confirm RED, implement CLI and capture lifecycle, then confirm GREEN without a camera.

### Task 4: Frame conversion, readable UI, and metrics

**Files:**
- Create: `tensorrt_yolo/examples/yolo_domain_frame.hpp`
- Create: `tensorrt_yolo/examples/yolo_domain_frame.cpp`
- Create: `tensorrt_yolo/examples/yolo_domain_ui.hpp`
- Create: `tensorrt_yolo/examples/yolo_domain_ui.cpp`
- Create: `tensorrt_yolo/tests/test_yolo_domain_ui.cpp`
- Modify: `tensorrt_yolo/CMakeLists.txt`

- [x] Test exact NV12/I420/RGB24/BGRA conversion contracts and malformed-frame failures.
- [x] Test deterministic overlay text, high-contrast style, key decoding, bounded P50/P95 metrics,
  and parking summary formatting.
- [x] Confirm RED, implement conversion/drawing/statistics, and confirm GREEN.

### Task 5: Application orchestration and build preset

**Files:**
- Create: `tensorrt_yolo/examples/yolov8_domain_demo.cpp`
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`
- Modify: `tensorrt_yolo/CMakeLists.txt`
- Modify: `CMakeUserPresets.json`
- Modify: `tensorrt_yolo/README.md`

- [x] Add opt-in CPU/application options and fail-fast dependency validation.
- [x] Add a Windows preset enabling CPU+TensorRT, OpenCV Lite, Turbo Capture, and local integration
  model cache paths without compiling paths into source.
- [x] Implement image-directory and camera loops with one detector/tracker state owner and explicit
  reset/shutdown behavior.
- [x] Document commands, class contracts, pipeline timings, NV12 behavior, and engine generation.

### Task 6: Verification and measured smoke tests

- [x] Configure/build the application preset and run focused tests.
- [x] Run the full preset CTest suite.
- [x] Generate ignored TensorRT engines for the three ONNX files with TensorRT 11.2.
- [x] Smoke-test CPU and GPU image/camera paths as available, recording exact commands and measured
  load/detect/track/render/total timing output.
- [x] Sync CodeGraph, inspect affected files, verify the worktree diff contains no models, engines,
  local indexes, or dependency junction entries, and report remaining hardware/data risks.
