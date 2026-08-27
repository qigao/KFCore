# Face Swap Demo UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight OpenCV desktop demo that visualizes source, target, and TensorRT face-swap output and supports rerun, save, and quit actions.

**Architecture:** Keep UI code in the examples boundary and reuse the existing CLI parser and `TensorRtFaceSwapApplication`; the library API and synchronous, non-reentrant execution contract remain unchanged. Separate deterministic key decoding and canvas composition from the HighGUI event loop so they can be unit tested without opening a window.

**Tech Stack:** C++17, TensorRT face applications, OpenCV Lite core/imgproc/imgcodecs/highgui, CMake, TinyTest.

**Spec:** `face_applications/README.md`

## Global Constraints

- Reuse the existing explicit model and image CLI paths; do not search for models or convert ONNX implicitly.
- Keep inference synchronous and fail fast on invalid images, engines, matrices, or output writes.
- Add no dependency beyond the existing OpenCV Lite package.
- Do not change the installed `KFCore::face_applications` public interface.

---

### Task 1: OpenCV Lite HighGUI component

**Files:**
- Modify: `cmake/tests/test_find_opencv_lite.cmake`
- Modify: `cmake/FindOpenCVLite.cmake`

**Interfaces:**
- Consumes: `OPENCV_LITE_ROOT` containing version-matched OpenCV libraries.
- Produces: validated imported target `OpenCVLite::highgui` when explicitly requested.

- [x] **Step 1: Extend the finder fixture to request `highgui`**

Add `highgui` to the synthetic SDK component lists and successful discovery cases.

- [x] **Step 2: Run the finder test and verify it fails**

Run: `ctest --test-dir build/Msvc-Face -C Release -R ^test_find_opencv_lite$ --output-on-failure`

Expected: FAIL with `FindOpenCVLite does not support component: highgui`.

- [x] **Step 3: Add `highgui` to the validated component set**

Extend supported components, preseed rejection, and partial-target detection without changing default components.

- [x] **Step 4: Run the finder test and verify it passes**

Run the command from Step 2; expected: PASS.

### Task 2: Testable demo presentation logic

**Files:**
- Create: `face_applications/examples/face_swap_demo_ui.hpp`
- Create: `face_applications/examples/face_swap_demo_ui.cpp`
- Create: `face_applications/tests/test_face_swap_demo_ui.cpp`
- Modify: `face_applications/CMakeLists.txt`

**Interfaces:**
- Produces: `DemoAction action_from_key(int) noexcept` and `cv::Mat compose_canvas(...)` in `kfcore::face_applications::demo`.
- Consumes: three BGR images and a status string; an empty result image represents pending inference.

- [x] **Step 1: Write failing tests for keys, layout, input preservation, and invalid image types**

Test `Q/q/Escape`, `R/r`, `S/s`, unknown keys, pending-result composition, non-mutating input, and fail-fast validation.

- [x] **Step 2: Configure and build the test to verify it fails**

Run: `cmake --build --preset win-face-applications-release-user --target test_face_swap_demo_ui`

Expected: compile failure because `face_swap_demo_ui.hpp` does not exist.

- [x] **Step 3: Implement minimal key decoding and fixed-size letterboxed canvas composition**

Use named layout constants, preserve all inputs, accept only `CV_8UC3`, and render a pending placeholder when result is empty.

- [x] **Step 4: Build and run the focused test**

Run: `ctest --test-dir build/Msvc-Face -C Release -R ^test_face_swap_demo_ui$ --output-on-failure`

Expected: PASS.

### Task 3: Interactive demo executable and documentation

**Files:**
- Create: `face_applications/examples/face_swap_demo.cpp`
- Modify: `face_applications/CMakeLists.txt`
- Modify: `face_applications/README.md`

**Interfaces:**
- Consumes: the same arguments accepted by `face_swap_image`.
- Produces: `face_swap_demo`, with `R` rerun, `S` save to `--output`, and `Q`/Escape/window-close quit.

- [x] **Step 1: Add the executable target and require `OpenCVLite::highgui` only for examples**

Link the event-loop translation unit to `KFCore::face_applications`, `OpenCVLite::imgcodecs`, `OpenCVLite::highgui`, and `TurboUtils::Core`.

- [x] **Step 2: Implement the synchronous event loop**

Show a loading canvas before engine creation, run the initial swap, report elapsed milliseconds, handle explicit key actions, validate output, and surface errors at the executable boundary.

- [x] **Step 3: Document launch and controls**

Add a complete command using the existing model flags and explain that `--gfpgan` enables enhancement.

- [x] **Step 4: Build, test, and perform a launch smoke check**

Run the focused targets, the related CTest subset, then launch the demo with the configured integration assets and close it through the UI.

- [x] **Step 5: Commit**

Commit the tested UI, finder, tests, and documentation together as one user-visible example feature.
