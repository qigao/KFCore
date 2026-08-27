# Face Swap Stage Timings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add synchronous per-stage wall-time telemetry to the TensorRT face-swap application and display current, P50, and P95 timings in the demo UI.

**Architecture:** Keep `swap()` as the compatibility path and route it together with the new `swap_profiled()` API through one private implementation. The core records only `std::chrono::nanoseconds`; the demo owns bounded aggregation, percentile calculation, formatting, and rendering. Optional model stages remain explicitly absent when their model is disabled.

**Tech Stack:** C++20, TensorRT 11, CUDA, OpenCV Lite, CMake Presets, TurboUtils TinyTest.

---

### Task 1: Define the public timing contract with failing integration coverage

**Files:**
- Modify: `face_applications/include/kfcore/face_applications/tensorrt.hpp`
- Modify: `face_applications/tests/test_face_swap_integration.cpp`

**Steps:**
1. Add a test that calls `swap_profiled()`, verifies a valid image, verifies mandatory stages and total duration are positive, and verifies disabled optional stages are absent.
2. Add GFPGAN coverage that verifies its optional timing stages are present when configured.
3. Build the integration test and preserve the expected compile failure before adding the API.
4. Define `FaceAnalysisTimingReport`, `FaceSwapTimingReport`, and `ProfiledFaceSwapResult` using `std::chrono::nanoseconds` and `std::optional` for optional stages.

### Task 2: Implement synchronous stage timing without changing `swap()` behavior

**Files:**
- Modify: `face_applications/src/tensorrt.cpp`
- Modify: `face_applications/include/kfcore/face_applications/tensorrt.hpp`

**Steps:**
1. Add nullable timing-report plumbing to `Impl::analyze_internal()` and the shared private swap implementation.
2. Time detector, Face68, ArcFace, optional age/gender, projection, InSwapper, composition, and optional GFPGAN boundaries with `steady_clock`.
3. Preserve existing exception translation and input/output behavior in the shared implementation.
4. Ensure `swap()` passes no timing destination, so it performs no clock reads; make `swap_profiled()` return the image and populated report.
5. Build and run the real TensorRT integration test.

### Task 3: Add bounded percentile aggregation with unit tests

**Files:**
- Create: `face_applications/examples/face_swap_demo_metrics.hpp`
- Create: `face_applications/examples/face_swap_demo_metrics.cpp`
- Modify: `face_applications/tests/test_face_swap_demo_ui.cpp`
- Modify: `face_applications/CMakeLists.txt`

**Steps:**
1. Add failing tests for current, nearest-rank P50/P95, optional-stage absence, and fixed-capacity eviction.
2. Implement a 120-sample bounded history in the demo layer.
3. Keep percentile work out of the inference core and expose presentation-ready timing rows.
4. Run the demo UI unit test.

### Task 4: Render telemetry in the demo UI

**Files:**
- Modify: `face_applications/examples/face_swap_demo.cpp`
- Modify: `face_applications/examples/face_swap_demo_ui.hpp`
- Modify: `face_applications/examples/face_swap_demo_ui.cpp`
- Modify: `face_applications/tests/test_face_swap_demo_ui.cpp`

**Steps:**
1. Replace the demo's outer stopwatch with `swap_profiled()` and append each report to the bounded history.
2. Extend the canvas footer to render stage rows in two columns with `current / P50 / P95` milliseconds.
3. Keep the existing R/S/Q interaction and image validation behavior.
4. Run the demo UI unit test.

### Task 5: Document and verify the complete change

**Files:**
- Modify: `face_applications/README.md`

**Steps:**
1. Document timing semantics: synchronous wall time, included CPU adapter work, optional stages, and no FLOPS/GPU-utilization claim.
2. Configure/build with the repository presets and configured TensorRT/OpenCV Lite paths.
3. Run the focused tests, then the full CTest suite.
4. Run a real GPU demo smoke test and visually inspect its telemetry canvas.
5. Review the diff and commit the verified change without pushing unless requested.
