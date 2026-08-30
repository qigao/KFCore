# ONNX Runtime、Hand 与 Face 模块边界实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `runtime_onnx` 成为唯一 ONNX session 实现，将 FaceMesh/face detector 归入 `face_model_*`，并把剩余 `vision_model_*` 改为 `hand_model_*`。

**Architecture:** runtime 只负责通用模型契约、session、tensor 与 ORT 错误；face/hand adapter 在各自边界把 runtime 错误翻译为领域错误。Face 与 Hand 使用独立公开 namespace、头文件和 CMake target，不保留旧 vision target 或兼容 alias。

**Tech Stack:** C++17、ONNX Runtime、TensorRT 11/CUDA、CMake Presets、TurboUtils TinyTest

**Spec:** `docs/superpowers/specs/2026-08-30-face-model-cpu-cuda-backends-design.md`

## Global Constraints

- 只使用标准 `win-release-user` configure/build/test preset。
- 不保留旧 `vision_model*` target、header 或 namespace alias。
- 不允许 face/hand 模块直接包含 `onnxruntime_cxx_api.h`。
- runtime 不依赖 face/hand 领域类型或错误码。
- OpenCV 只允许存在于 demo/UI，不进入生产模型 target。

---

### Task 1: Public API contract tests

**Files:**
- Create: `face_model_core/tests/test_face_pipeline_api.cpp`
- Create: `hand_model_core/tests/test_hand_model_api.cpp`
- Modify: `face_model_core/tests/CMakeLists.txt`
- Modify: `hand_model_core/tests/CMakeLists.txt`

**Interfaces:**
- Produces: `kfcore/face_models/core.hpp`, `kfcore/hand_models/core.hpp` and explicit face/hand backend signatures.

- [x] Add compile-time tests for `face_models::FaceMeshPipeline`, `FaceDetectorBackend`, `FaceLandmarkBackend`, `hand_models::HandPipeline` and `HandInferenceBackend`.
- [x] Build the two new test targets and verify failure because the new headers do not exist.

### Task 2: Make runtime_onnx the only session implementation

**Files:**
- Delete: `face_models_cpu/src/cpu_session.hpp`
- Delete: `face_models_cpu/src/cpu_session.cpp`
- Delete: `hand_models_cpu/src/onnx_session.hpp`
- Delete: `hand_models_cpu/src/onnx_session.cpp`
- Modify: CPU face and hand implementation files and CMake source lists.

**Interfaces:**
- Consumes: `runtime_onnx::Environment`, `runtime_onnx::Session`, `ModelContract`, `FloatTensorView`, `HostTensor`.
- Produces: domain-local contract factories and error translation helpers without another session class.

- [x] Replace wrapper-owned sessions with direct runtime composition while preserving error codes and output validation.
- [x] Remove direct ORT use from integration tests.
- [x] Build runtime and CPU model tests; verify all pass.

### Task 3: Move face detector and FaceMesh into face_model targets

**Files:**
- Modify: `face_model_core`, `face_models_cpu`, `face_models_cuda` public headers, sources, tests and CMake files.
- Modify: `face_applications_cpu` and hand-interaction FaceMesh demo consumers.
- Modify: shared face result, geometry, decode and pipeline ownership.

**Interfaces:**
- Produces: `face_models::FaceDetectorBackend`, `FaceLandmarkBackend`, `FaceMeshPipeline`, `CpuFaceDetector`, `CpuFaceLandmarker`, `TensorRtFaceDetector`, `TensorRtFaceLandmarker`.

- [x] Move face-only types and pipeline behavior without changing inference semantics.
- [x] Move CPU and TensorRT adapters and their model contracts.
- [x] Build face core/CPU/CUDA/application tests; verify all pass.

### Task 4: Rename remaining vision modules to hand modules

**Files:**
- Rename: `vision_model_core` to `hand_model_core`.
- Rename: `vision_models_cpu` to `hand_models_cpu`.
- Rename: `vision_models_cuda` to `hand_models_cuda`.
- Modify: namespaces, public include paths, CMake targets, hand interaction consumers, README and tests.

**Interfaces:**
- Produces: `KFCore::hand_model_core`, `KFCore::hand_models_cpu`, `KFCore::hand_models_cuda` under `kfcore::hand_models`.

- [x] Rename hand-only types and implementation namespaces.
- [x] Update every repository consumer and delete old target/header references.
- [x] Build hand core/CPU/CUDA and hand-interaction tests; verify all pass.

### Task 5: Full verification

**Files:**
- Modify: root README and architecture spec/plan.

**Interfaces:**
- Produces: clean generated export with runtime, face, hand and YOLO targets.

- [x] Run standard configure and full build.
- [x] Run full `ctest --preset win-release-user`.
- [x] Verify generated exports contain no old vision targets.
- [x] Run old-name, direct-ORT, OpenCV-boundary, `git diff --check` and `.codegraph/` checks.
