# Hand and MediaPipe CPU/TensorRT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenCV-free CPU and TensorRT CUDA execution for the local palm, hand-landmark, keypoint-classifier, and MediaPipe 468-face-landmark models, with KFCore ByteTrack orchestration.

**Architecture:** A shared core owns public types, geometry, and per-instance tracking. ONNX Runtime CPU and TensorRT CUDA are separate adapters behind a narrow backend interface; the GPU backend stages each frame once and reuses that CUDA image for palm and ROI preprocessing.

**Tech Stack:** C++17, ONNX Runtime CPU, TensorRT 11.2, CUDA 12.8, KFCore ImageProcessor, KFCore trackers, TurboUtils TinyTest, CMake Presets.

**Spec:** `docs/superpowers/specs/2026-08-28-hand-mediapipe-cpu-tensorrt-design.md`

## Global Constraints

- Preserve YOLOv12-face as the existing face detector; do not add MediaPipe face detection.
- Do not add OpenCV to the new CPU or TensorRT model APIs.
- CPU accepts host images only; TensorRT accepts host or CUDA images and performs no implicit CPU inference fallback.
- Every mutable runtime/pipeline instance is synchronous, non-reentrant, and independently owned.
- Every image, tensor, model, output, and hand-count growth path has a configurable hard bound.
- Engine files and model binaries remain external deployment assets and are not committed.

---

### Task 1: Bounded TensorRT data-dependent outputs

**Files:**
- Modify: `tensorrt_runtime/include/kfcore/tensorrt/types.hpp`
- Modify: `tensorrt_runtime/include/kfcore/tensorrt/runtime.hpp`
- Modify: `tensorrt_runtime/src/tensor_validation.hpp`
- Modify: `tensorrt_runtime/src/tensor_validation.cpp`
- Modify: `tensorrt_runtime/src/executor.cpp`
- Modify: `tensorrt_runtime/tests/test_tensor_validation.cpp`
- Modify: `tensorrt_runtime/tests/test_tensorrt_runtime_integration.cpp`
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`
- Modify: `tensorrt_runtime/CMakeLists.txt`

**Interfaces:**
- Produces: `DynamicOutputRequest { name, data_type, max_byte_size }`.
- Produces: `HostTensor { name, data_type, shape, bytes }` with typed checked accessors.
- Produces: `Executor::run_dynamic(const std::vector<TensorView>&, const std::vector<DynamicOutputRequest>&)`.

- [x] **Step 1: Write failing unit tests for request completeness, duplicate names, zero capacity, aggregate limits, and checked typed access.**

```cpp
const DynamicOutputRequest request{"detections", DataType::Float32, 4096};
detail::validate_dynamic_output_requests({output_descriptor("detections")}, {request}, 4096);
check_error([&] { detail::validate_dynamic_output_requests(
    {output_descriptor("detections")}, {{"detections", DataType::Float32, 0}}, 4096); },
    TensorRtErrorCode::ResourceLimitExceeded, "positive");
```

- [x] **Step 2: Configure/build the TensorRT runtime tests and verify RED because the new types/API do not exist.**

Run: `cmake --fresh --preset win-tensorrt-models-release-user` then build `test_tensorrt_runtime_contract`.
Expected: compilation failure naming `DynamicOutputRequest` or `validate_dynamic_output_requests`.

- [x] **Step 3: Implement validation and `run_dynamic()` with bounded executor-owned CUDA/pinned staging and owning host results.**

```cpp
struct DynamicOutputRequest {
    std::string name;
    DataType data_type = DataType::Float32;
    std::size_t max_byte_size = 0;
};
struct HostTensor {
    std::string name;
    DataType data_type = DataType::Float32;
    TensorShape shape;
    std::vector<std::byte> bytes;
};
```

- [x] **Step 4: Run the unit test GREEN, then add and run a Palm-engine integration test that asserts `[N,8]`, `N <= 2016`, and finite FP32 values.**

Run: `ctest --preset win-tensorrt-models-release-user -R "tensorrt_(tensor_validation|runtime_dynamic_output)"`.
Expected: both tests pass when the Palm engine cache variable is supplied.

- [x] **Step 5: Commit the runtime capability.**

```bash
git add CMakeOptions.cmake CMakeLists.txt tensorrt_runtime
git commit -m "feat(tensorrt): support bounded data-dependent outputs"
```

### Task 2: Shared vision model contracts, geometry, and tracking

**Files:**
- Create: `vision_models/CMakeLists.txt`
- Create: `vision_models/include/kfcore/vision_models/error.hpp`
- Create: `vision_models/include/kfcore/vision_models/types.hpp`
- Create: `vision_models/include/kfcore/vision_models/core.hpp`
- Create: `vision_models/src/error.cpp`
- Create: `vision_models/src/geometry.hpp`
- Create: `vision_models/src/geometry.cpp`
- Create: `vision_models/src/decode.hpp`
- Create: `vision_models/src/decode.cpp`
- Create: `vision_models/src/pipeline.cpp`
- Create: `vision_models/tests/test_geometry.cpp`
- Create: `vision_models/tests/test_decode.cpp`
- Create: `vision_models/tests/test_hand_pipeline.cpp`
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `HandInferenceBackend::infer(const image::ImageView&) -> std::vector<HandResult>`.
- Produces: `HandPipeline::create(std::unique_ptr<HandInferenceBackend>, HandPipelineOptions)` and `process/reset`.
- Produces: common Palm `[N,8]`, hand `[N,63]/[N,1]/[N,1]`, classifier, and 468-point decoders.

- [x] **Step 1: Write failing tests with hand-derived transforms and literal decoded coordinates.**

```cpp
const auto roi = detail::make_hand_roi(/* normalized center */ 0.5F, 0.5F,
                                       /* normalized size */ 0.25F, 0.0F, 640, 480);
check_close(roi.rect.width, 160.0F, 1e-5F);
check_close(roi.rect.height, 160.0F, 1e-5F);
```

- [x] **Step 2: Build the new test target and verify RED because the module is absent.**

Run: build target `test_vision_model_geometry`.
Expected: configure or compile failure caused by missing production symbols.

- [x] **Step 3: Implement checked geometry/decode functions and immutable owning result types.**

The transform maps destination pixel coordinates directly into the source image, so CPU and CUDA use one factual geometry path.

- [x] **Step 4: Write a failing pipeline test using a deterministic real backend implementation that returns two observations and verifies ByteTrack ids and detection-index association.**

- [x] **Step 5: Implement `HandPipeline` with uniquely owned ByteTrack state, fail-fast non-reentrancy, bounded result counts, reset, and no model fallback.**

- [x] **Step 6: Run core tests GREEN and commit.**

```bash
git add CMakeOptions.cmake CMakeLists.txt vision_models
git commit -m "feat(vision): add hand geometry and tracking core"
```

### Task 3: OpenCV-free ONNX Runtime CPU backend

**Files:**
- Create: `vision_models/include/kfcore/vision_models/cpu.hpp`
- Create: `vision_models/src/onnx_session.hpp`
- Create: `vision_models/src/onnx_session.cpp`
- Create: `vision_models/src/cpu.cpp`
- Create: `vision_models/tests/test_cpu_backend_validation.cpp`
- Create: `vision_models/tests/test_cpu_integration.cpp`
- Modify: `vision_models/CMakeLists.txt`
- Modify: `CMakeUserPresets.json`

**Interfaces:**
- Produces: `CpuHandBackend::load(HandOnnxModelPaths, CpuVisionOptions)`.
- Produces: `CpuFaceLandmarker::load(path, CpuVisionOptions)` and `infer(ImageView, FaceBox)`.

- [x] **Step 1: Write failing public-validation tests for empty paths, host-only input, invalid boxes, zero limits, and non-copyable instance ownership.**

- [x] **Step 2: Build and verify RED because CPU classes do not exist.**

Run: build `test_vision_models_cpu_validation`.
Expected: compilation failure naming the new CPU API.

- [x] **Step 3: Implement strict ONNX tensor-name/type/shape validation and CPU preprocessing through `CpuImageProcessor`.**

Palm uses RGB unit-range letterbox; hand and face ROIs use RGB unit-range affine sampling. Keypoints are wrist-relative and normalized by maximum absolute coordinate.

- [x] **Step 4: Implement inference/decode and verify unit tests GREEN.**

- [x] **Step 5: Add opt-in real-model integration tests for all four ONNX files; assert bounded counts, exact output sizes, finite results, and successful pipeline tracking.**

Run: `ctest --preset win-vision-models-cpu-release-user -R vision_models_cpu`.
Expected: all CPU tests pass.

- [x] **Step 6: Commit the CPU backend.**

```bash
git add CMakeUserPresets.json vision_models
git commit -m "feat(vision): add ONNX Runtime CPU hand and face landmarks"
```

### Task 4: TensorRT CUDA backend sharing one staged image

**Files:**
- Create: `vision_models/include/kfcore/vision_models/tensorrt.hpp`
- Create: `vision_models/src/tensorrt_models.hpp`
- Create: `vision_models/src/tensorrt_models.cpp`
- Create: `vision_models/src/tensorrt.cpp`
- Create: `vision_models/tests/test_tensorrt_backend_validation.cpp`
- Create: `vision_models/tests/test_tensorrt_integration.cpp`
- Modify: `vision_models/CMakeLists.txt`
- Modify: `CMakeUserPresets.json`

**Interfaces:**
- Produces: `TensorRtHandBackend::load(HandTensorRtEnginePaths, TensorRtVisionOptions)`.
- Produces: `TensorRtFaceLandmarker::load(path, TensorRtVisionOptions)` and `infer(ImageView, FaceBox)`.

- [x] **Step 1: Write failing validation tests for engine paths, device/limits, strict tensor contracts, and non-copyable ownership.**

- [x] **Step 2: Build and verify RED because TensorRT classes do not exist.**

Run: build `test_vision_models_tensorrt_validation`.
Expected: compilation failure naming the new TensorRT API.

- [x] **Step 3: Implement strict adapters for Palm, HandLandmark, KeypointClassifier, and MediaPipeFaceLandmark engines.**

Palm calls `run_dynamic()` with a checked cap of `max_palm_candidates * 8 * sizeof(float)` (2016 candidates for the trusted model). It then confidence-sorts and truncates to the independent `max_hands` limit. Fixed outputs use existing executor views.

- [x] **Step 4: Implement the CUDA backend: stage once, preprocess Palm and each ROI with `process_affine()`, infer on GPU, then perform bounded CPU decode/tracking.**

- [x] **Step 5: Generate local TensorRT engines from the four trusted ONNX files and run CUDA integration tests.**

Run: `ctest --preset win-vision-models-tensorrt-release-user -R vision_models_tensorrt`.
Expected: Palm dynamic output and the three fixed-output engines execute on device 0 with finite results.

- [x] **Step 6: Commit the TensorRT backend.**

```bash
git add CMakeUserPresets.json vision_models
git commit -m "feat(vision): add TensorRT hand and MediaPipe landmarks"
```

### Task 5: Packaging, documentation, and regression verification

**Files:**
- Create: `vision_models/README.md`
- Modify: `cmake/KFCoreConfig.cmake.in`
- Modify: `CMakeLists.txt`
- Modify: `CMakeUserPresets.json`
- Modify: `README.md`

**Interfaces:**
- Produces exported targets `KFCore::vision_model_core`, `KFCore::vision_models_cpu`, and `KFCore::vision_models_tensorrt` when enabled.

- [x] **Step 1: Add installation/export dependency tests that fail before package metadata is updated.**

- [x] **Step 2: Update package flags/dependencies and document model contracts, engine conversion commands, ownership, CPU/GPU flow, and limitations.**

- [x] **Step 3: Run CodeGraph sync and affected analysis to confirm no existing face/YOLO dependency inversion.**

Run: `codegraph sync .` and `codegraph affected -p . <changed source files>`.

- [x] **Step 4: Run focused CPU, TensorRT, image processor, trackers, and face-model tests, then full relevant preset suites.**

Run: `ctest --preset win-vision-models-cpu-release-user`, `ctest --preset win-vision-models-tensorrt-release-user`, and the existing face/TensorRT preset tests.

- [x] **Step 5: Verify clean diff, scan for placeholder markers, and commit documentation/package integration.**

```bash
git diff --check
rg.exe -n "TODO|FIXME|HACK|Not implemented" vision_models tensorrt_runtime
git add README.md CMakeLists.txt CMakeUserPresets.json cmake/KFCoreConfig.cmake.in vision_models/README.md docs/superpowers
git commit -m "docs(vision): document CPU and TensorRT model pipelines"
```
