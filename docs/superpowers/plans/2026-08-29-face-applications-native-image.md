# Face Applications Native Image Implementation Plan

**Goal:** Accept raw BGR/RGB/NV12/I420 frames in both face-application backends while keeping
OpenCV at demo/test adapter boundaries.

**Architecture:** Replace the TensorRT facade's `cv::Mat` contract with borrowed `ImageView` and
owned `BgrImage`. Reuse the existing per-device `CudaImageProcessor`, extend CUDA composition to
sample YUV bases, and replace private OpenCV geometry/mask utilities with bounded project-owned
types. The CPU facade converts a Host `ImageView` once before entering its existing pipeline.

**Tech Stack:** C++17, CUDA, TensorRT, ONNX Runtime, KFCore ImageProcessor, TinyTest, CMake presets.

---

### Task 1: Protect YUV composition behavior

- Add CUDA tests with literal NV12/I420 fixtures and expected BGR output.
- Run the test and confirm the existing RGB/BGR-only validation fails.
- Extend layout validation and the composition kernel to sample YUV while returning BGR8.

### Task 2: Remove OpenCV from the TensorRT core contract

- Add project-owned point, affine and float-mask types and behavior tests.
- Replace OpenCV geometry/mask implementation using the existing CPU algorithms as the reference.
- Remove obsolete host preprocess/composer APIs and core OpenCV linkage.

### Task 3: Add native TensorRT ImageView flow

- Change public input/output signatures and compile-contract tests.
- Map all four formats to TensorRT-YOLO, stage once per device, and preprocess Age/Gender on CUDA.
- Allocate/download the final owned BGR result using target `ImageView` dimensions.

### Task 4: Add CPU ImageView flow

- Add overload contract and invalid-memory tests where model-free testing is possible.
- Convert a Host view once with `CpuImageProcessor::copy_bgr`, then reuse existing analysis/swap.
- Extend the opt-in integration test to exercise YUV input.

### Task 5: Move OpenCV adaptation to examples/tests and verify

- Adapt CLI/demo, benchmark and integration code at their decode/display boundaries.
- Update README and installed-consumer coverage.
- Configure/build the relevant presets, run focused tests, then the adjacent suite.
