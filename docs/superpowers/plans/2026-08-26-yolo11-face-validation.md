# YOLO11 Face Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible, opt-in YOLO11-face TensorRT integration-test line that reuses the existing KFCore detector and KFCore ByteTrack/Kalman implementation without committing third-party source, weights, or generated engines.

**Architecture:** Keep the five-tensor EfficientNMS runtime contract as the single detector path. Add one optional configure-time engine path which registers the existing integration executable under a second CTest name with an isolated environment value. Generate the face engine only as a local ignored artifact, then exercise both the model-independent TensorRT contract tests and the existing OpenCV image-sequence tracker.

**Tech Stack:** CMake 3.20+, CTest, C++17, TensorRT 11.2.1.2, CUDA 12.8, Ultralytics/ONNX local export tooling, opencv-lite 4.13, TurboUtils TinyTest.

**Spec:** `docs/superpowers/specs/2026-08-25-tensorrt-yolo-bytetrack-design.md` (validation-only extension; no runtime API or engine-contract change).

**Global Constraints:** Follow repository `AGENTS.md`; fail fast on missing or invalid configured engine paths; use one validated canonical path as the CTest environment fact source; do not copy or compile `akanametov/yolo-face` source; do not commit `.pt`, `.onnx`, `.engine`, downloaded images, or generated outputs; document upstream provenance and unresolved weight-redistribution licensing separately from source-code licensing.

## Task 1: Specify the optional YOLO11-face CTest contract

**Files:**
- Modify: `cmake/tests/TensorRtIntegrationEngineFixture/CMakeLists.txt`
- Modify: `cmake/tests/TensorRtIntegrationEngineFixture/check_engine_environment.cmake`
- Modify: `cmake/tests/test_tensorrt_integration_engine_config.cmake`

- [ ] Add a failing fixture case proving that a second validated engine path creates a distinct test and injects it as `KFCORE_TENSORRT_TEST_ENGINE` for the reused executable.
- [ ] Run `ctest -R test_tensorrt_integration_engine_config --output-on-failure` and confirm the new assertion fails for the intended missing behavior.
- [ ] Commit the red test separately.

## Task 2: Register the optional YOLO11-face engine

**Files:**
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`
- Modify: `tensorrt_yolo/CMakeLists.txt`

- [ ] Add cache `FILEPATH` `KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE`, defaulting from its same-named environment variable.
- [ ] Validate it only when non-empty and integration tests are enabled; an explicitly configured invalid path must fail at configure time.
- [ ] Register `test_tensorrt_integration_yolo11_face` using the existing `test_tensorrt_integration` executable and the validated face-engine path.
- [ ] Re-run the focused CMake fixture test and the default development CTest suite.
- [ ] Commit the green implementation.

## Task 3: Produce a local YOLO11-face EfficientNMS engine

**Local artifacts only:**
- `C:/projects/cpp/KFCore/TensorRT-YOLO/examples/detect/models/yolov11n-face.pt`
- derived raw ONNX, EfficientNMS ONNX, and TensorRT engine beside it

- [ ] Download `yolov11n-face.pt` from the upstream release linked by `akanametov/yolo-face` and record URL, byte size, and SHA-256 in the verification log.
- [ ] Export a dynamic raw ONNX in the existing isolated Python environment and inspect its input/output names and shapes.
- [ ] Convert the raw head to the required `images`, `num_dets`, `boxes`, `scores`, `labels` EfficientNMS contract, retaining a dynamic min/opt/max profile.
- [ ] Build the TensorRT 11.2 engine with `trtexec`, inspect its bindings/profile, and reject the artifact if any contract detail differs.

## Task 4: Verify TensorRT inference and KFCore tracking

**Tests/commands:**
- Existing binary: `build/Msvc-Release/bin/Release/test_tensorrt_integration.exe`
- Existing example: `build/Msvc-Release/bin/Release/track_image_sequence.exe`

- [ ] Run the integration TinyTest binary with `KFCORE_TENSORRT_TEST_ENGINE` set to the face engine, covering min/opt/max spatial sizes, batch, host/device equality, and context isolation.
- [ ] Configure the optional face-engine cache path and prove `ctest -R test_tensorrt_integration_yolo11_face` executes independently.
- [ ] Run a real image sequence through the face detector and KFCore ByteTrack/Kalman example; inspect at least one rendered output image and confirm stable face track IDs across repeated frames.
- [ ] Run adjacent TensorRT, tracking, OpenCV, CUDA buffer, engine-file, and CMake configuration tests.

## Task 5: Document, review, and publish evidence

**Files:**
- Modify: `tensorrt_yolo/README.md`
- Modify: `docs/superpowers/specs/2026-08-25-tensorrt-yolo-bytetrack-design.md`

- [ ] Document the YOLO11 and YOLO11-face validation matrix, exact opt-in CMake usage, output contract, upstream links, and local-only artifact/licensing boundary.
- [ ] Update the design verification section without changing the runtime scope or claiming redistribution rights for model weights.
- [ ] Run formatting/diff checks, focused tests, full relevant CTest, and a final code review.
- [ ] Commit, push to the existing branch/PR, and update the GitHub issue with reproducible commands and measured results.

