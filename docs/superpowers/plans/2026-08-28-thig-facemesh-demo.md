# THIG FaceMesh Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional CPU and TensorRT YOLOv12-face plus MediaPipe FaceMesh inference to the existing THIG camera window, with bounded configuration and per-stage timings.

**Architecture:** A backend-neutral `FaceMeshPipeline` composes narrow face detector and landmarker interfaces. CPU detection lives in the existing ONNX Runtime vision adapter; the TensorRT demo supplies a thin TensorRT-YOLO detector adapter, while the UI consumes only owned backend-neutral results.

**Tech Stack:** C++17, ONNX Runtime CPU, TensorRT 11.2, CUDA 12.8, KFCore TensorRT-YOLO, KFCore ImageProcessor, OpenCV Lite HighGUI, TurboUtils TinyTest, CMake Presets.

**Spec:** `docs/design/2026-08-28-thig-facemesh-demo.md`

## Global Constraints

- Preserve YOLOv12-face as the face detector; do not use `MediaPipeFaceDetector.onnx`.
- Existing hand-only CLI invocations and THIG behavior remain unchanged.
- FaceMesh is optional, but detector and landmark paths must be supplied together.
- No-face frames return an empty face result; dependency, contract, and runtime failures fail fast.
- CPU and TensorRT instances are synchronous, independently owned, and non-reentrant.
- Model and engine files remain ignored external assets.
- The first version draws 468 points and does not copy an unverified FaceMesh connectivity table.

---

### Task 1: Backend-neutral FaceMesh pipeline

**Files:**
- Modify: `vision_models/include/kfcore/hand_models/types.hpp`
- Modify: `vision_models/include/kfcore/hand_models/core.hpp`
- Modify: `vision_models/src/pipeline.cpp`
- Modify: `vision_models/tests/test_hand_pipeline.cpp`

**Interfaces:**
- Produces: `FaceDetection`, `FaceDetectionResult`, `FaceMeshTimings`, and `FaceMeshFrame` owned value types.
- Produces: `FaceDetectorBackend::infer(ImageView)`, `FaceLandmarkBackend::infer(ImageView, RectF)`, and `FaceMeshPipeline::process(ImageView)`.

- [x] **Step 1: Write failing tests for detector-to-landmarker composition, empty detection, and confidence filtering.**

```cpp
auto pipeline = FaceMeshPipeline::create(
    std::make_unique<FixtureDetector>(FaceDetection{{10, 20, 30, 40}, 0.9F}),
    std::make_unique<FixtureLandmarker>(0.8F),
    FaceMeshPipelineOptions{0.5F});
const FaceMeshFrame frame = pipeline->process(valid_image());
check_true(frame.detection.has_value());
check_true(frame.landmarks.has_value());
check_equal(frame.landmarks->landmarks.size(), kFaceLandmarkCount);
```

- [x] **Step 2: Build `test_vision_model_hand_pipeline` and verify RED because the FaceMesh interfaces do not exist.**

Run: `cmake --build --preset win-release-user --target test_vision_model_hand_pipeline`.
Expected: compilation fails on `FaceMeshPipeline` or `FaceDetectorBackend`.

- [x] **Step 3: Implement the minimal value types and synchronous composition pipeline with finite threshold validation and owned results.**

```cpp
class FaceMeshPipeline final {
public:
    static std::unique_ptr<FaceMeshPipeline> create(
        std::unique_ptr<FaceDetectorBackend> detector,
        std::unique_ptr<FaceLandmarkBackend> landmarker,
        const FaceMeshPipelineOptions& options = {});
    FaceMeshFrame process(const image::ImageView& image);
};
```

- [x] **Step 4: Rebuild and run `test_vision_model_hand_pipeline`; expect all FaceMesh and existing hand cases to pass.**

Run: `ctest --preset win-release-user -R test_vision_model_hand_pipeline --output-on-failure`.

### Task 2: CPU YOLOv12-face detector adapter

**Files:**
- Modify: `vision_models/include/kfcore/hand_models/cpu.hpp`
- Modify: `vision_models/src/cpu.cpp`
- Modify: `vision_models/src/decode.hpp`
- Modify: `vision_models/src/decode.cpp`
- Modify: `vision_models/tests/test_decode.cpp`
- Modify: `vision_models/tests/test_cpu_backend_validation.cpp`
- Modify: `vision_models/tests/test_cpu_integration.cpp`

**Interfaces:**
- Consumes: `FaceDetectorBackend` and `FaceDetectionResult` from Task 1.
- Produces: `CpuFaceDetector::load(path, CpuHandOptions)` and `infer(ImageView)` with YOLOv12-face `images -> output0 [1,300,6]` validation.
- Changes: `CpuFaceLandmarker` implements `FaceLandmarkBackend`.

- [x] **Step 1: Write failing literal decode tests for highest-score class 0 selection, letterbox restoration, empty detections, non-finite values, and invalid boxes.**

```cpp
std::array<float, 12> rows{100, 120, 300, 360, 0.75F, 0,
                           90, 110, 310, 370, 0.90F, 0};
const auto face = detail::decode_yolo12_face(
    rows.data(), rows.size(), 0, 0.5F, transform, 640, 480);
check_true(face.has_value());
check_close(face->confidence, 0.90F, 0.0001F);
```

- [x] **Step 2: Build `test_vision_model_decode` and verify RED on the absent decoder.**

Run: `cmake --build --preset win-release-user --target test_vision_model_decode`.

- [x] **Step 3: Implement decoder and CPU adapter using RGB unit-range 640x640 letterbox with border 114, bounded tensor/model/output storage, and ordinary empty no-face results.**

```cpp
class CpuFaceDetector final : public FaceDetectorBackend {
public:
    static std::unique_ptr<CpuFaceDetector> load(
        const std::filesystem::path& model_path,
        const CpuHandOptions& options = {});
    FaceDetectionResult infer(const image::ImageView& image) override;
};
```

- [x] **Step 4: Run decode, validation, and CPU integration tests with `yolov12n-face.onnx` plus `MediaPipeFaceLandmarkDetector.onnx`.**

Run: `ctest --preset win-release-user -R "vision_model_(decode|cpu|hand_pipeline)" --output-on-failure`.

### Task 3: Demo CLI, TensorRT adapter, and one-window overlay

**Files:**
- Create: `hand_interaction/examples/hand_interaction_demo_face.hpp`
- Create: `hand_interaction/examples/hand_interaction_demo_face.cpp`
- Modify: `hand_interaction/examples/hand_interaction_demo_cli.hpp`
- Modify: `hand_interaction/examples/hand_interaction_demo_cli.cpp`
- Modify: `hand_interaction/examples/hand_interaction_demo.cpp`
- Modify: `hand_interaction/examples/hand_interaction_demo_ui.hpp`
- Modify: `hand_interaction/examples/hand_interaction_demo_ui.cpp`
- Modify: `hand_interaction/tests/test_hand_interaction_demo_cli.cpp`
- Modify: `hand_interaction/tests/test_hand_interaction_demo_ui.cpp`
- Modify: `hand_interaction/CMakeLists.txt`
- Modify: `CMakeLists.txt`
- Modify: `CMakeUserPresets.json`

**Interfaces:**
- Consumes: Task 1 pipeline, CPU detector, existing CPU/TensorRT landmark adapters, and TensorRT-YOLO detector.
- Produces: optional `Arguments::face_detector_model` and `facemesh_model`, plus validated `face_score` and `facemesh_score`.
- Produces: `make_face_mesh_pipeline(arguments)` and overlay rendering for `std::optional<FaceMeshFrame>`.

- [x] **Step 1: Write failing CLI tests for omitted, complete, partial, duplicate, unreadable, and out-of-range FaceMesh options.**

```cpp
const Arguments args = parse_arguments(
    {"demo", "--backend", "cpu", "--model-dir", model_dir,
     "--face-detector", detector, "--facemesh", mesh,
     "--face-score", "0.6", "--facemesh-score", "0.7"},
    {true, true});
check_equal(args.face_score, 0.6F);
check_equal(args.facemesh_score, 0.7F);
```

- [x] **Step 2: Write a failing UI test that supplies a face box and literal landmarks, then asserts colored output pixels while the source pixels remain unchanged.**

Run: `cmake --build --preset win-release-user --target test_hand_interaction_demo_cli test_hand_interaction_demo_ui`.
Expected: compilation fails on the new arguments and overlay parameter.

- [x] **Step 3: Implement CLI validation, the CPU/TensorRT factory adapter, serial per-frame FaceMesh processing, one composed overlay, and face timing rows.**

```cpp
std::unique_ptr<vision_models::FaceMeshPipeline>
make_face_mesh_pipeline(const Arguments& arguments);

cv::Mat compose_overlay(const cv::Mat& source,
                        const vision_models::HandFrame& hands,
                        const HandInteractionFrame& interaction,
                        const std::optional<vision_models::FaceMeshFrame>& face,
                        const DemoMetrics& metrics);
```

- [x] **Step 4: Enable TensorRT-YOLO only in the TensorRT demo preset, link the adapter there, and run CLI/UI plus adjacent hand tests under both CPU and TensorRT presets.**

Run: `ctest --preset win-release-user -R "hand_interaction_demo|vision_model" --output-on-failure` and the matching TensorRT preset command.

### Task 4: Assets, camera verification, and documentation

**Files:**
- Modify: `hand_interaction/README.md`

**Interfaces:**
- Produces: documented CPU and TensorRT invocations with explicit face model paths and timing semantics.

- [x] **Step 1: Generate `yolov12n-face.engine` and `face_landmark.engine` from the ignored local ONNX assets using TensorRT 11.2 strongly typed mode, then smoke-test each engine with `trtexec`.**

Run: `trtexec --onnx=<model> --saveEngine=<engine> --builderOptimizationLevel=3 --skipInference`, followed by one bounded `--loadEngine` inference for each shape.

- [x] **Step 2: Run bounded CPU and TensorRT camera demos with the two face arguments and verify model loading, processed-frame summaries, and absence of runtime error logs.**

```powershell
hand_interaction_demo.exe --backend tensorrt --camera 0 --mode 385 --max-frames 100 `
  --palm palm_detection.engine --hand hand_landmark.engine `
  --classifier keypoint_classifier.engine --face-detector yolov12n-face.engine `
  --facemesh face_landmark.engine
```

- [x] **Step 3: Update README behavior, model requirements, metrics, GPU staging limitation, and exact invocations; run the full relevant CTest suites and verify a clean branch diff.**

Run: `ctest --preset win-release-user --output-on-failure` and `ctest --preset win-release-user --output-on-failure`.
