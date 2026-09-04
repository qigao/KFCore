# THIG Hand Interaction Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in live USB-camera demo that runs KFCore hand inference and THIG interaction through Turbo Capture with CPU or TensorRT.

**Architecture:** Keep capture, frame conversion, CLI parsing, and UI rendering in the example boundary. Copy Turbo Capture's callback-borrowed frame into a bounded latest-frame mailbox, then let the single UI/inference thread share one owning BGR image across display, HandPipeline, and HandInteractionPipeline.

**Tech Stack:** C++17, Salts::Capture, OpenCV Lite core/imgproc/highgui, ONNX Runtime or TensorRT/CUDA, KFCore vision models/THIG, CMake presets, TinyTest.

**Spec:** `docs/superpowers/specs/2026-08-28-thig-hand-interaction-demo-design.md`

## Global Constraints

- Do not change any installed KFCore public library interface or make OpenCV/Capture a transitive library dependency.
- No callback-borrowed pointer may escape the Turbo Capture callback.
- Retained frame storage is exactly two vectors, each bounded by `max_frame_bytes`; unread frames are explicitly coalesced and counted.
- Support only exact packed NV12, I420, BGRA, and RGB24 frames; reject MJPEG and malformed lengths.
- CPU and TensorRT backend selection is explicit and has no runtime fallback.
- Keep model paths explicit/readable and do not search, download, convert, or generate models.
- Stop capture and drain callbacks before destroying capture/device/mailbox state.

---

### Task 1: CLI and deterministic native-mode selection

**Files:**
- Create: `hand_interaction/examples/hand_interaction_demo_cli.hpp`
- Create: `hand_interaction/examples/hand_interaction_demo_cli.cpp`
- Create: `hand_interaction/tests/test_hand_interaction_demo_cli.cpp`
- Modify: `hand_interaction/CMakeLists.txt`

**Interfaces:**
- Produces: `demo::Arguments parse_arguments(const std::vector<std::string>&, BackendAvailability)`.
- Produces: `std::size_t select_mode(const std::vector<salts_video_native_mode_t>&, const CaptureRequest&)`.
- Consumes: explicit backend availability, camera index, mode id or exact width/height/fps, and CPU/TensorRT model paths.

- [x] **Step 1: Write failing TinyTest cases.** Cover list-only operation without models, missing/duplicate/unknown options, numeric range errors, unavailable backend rejection, readable model validation, exact mode-id matching, exact geometry/fps matching, format preference, and MJPEG-only rejection.
- [x] **Step 2: Build `test_hand_interaction_demo_cli` and verify RED.** Run `cmake --build --preset win-release-user --target test_hand_interaction_demo_cli`; expect missing source/header failure.
- [x] **Step 3: Implement the minimal parser and selector.** Use named defaults (`1280x720@30`, camera 0, 64 MiB), `std::from_chars`, TurboUtils filesystem validation, and error messages prefixed with `hand_interaction_demo arguments:`.
- [x] **Step 4: Run focused tests GREEN.** Run `ctest --test-dir build/Msvc-Vision-CPU -C Release -R '^test_hand_interaction_demo_cli$' --output-on-failure`.
- [x] **Step 5: Commit.** Commit parser, selector, test, and CMake test target as one independently reviewable unit.

### Task 2: Bounded callback-to-consumer frame mailbox

**Files:**
- Create: `hand_interaction/examples/hand_interaction_demo_capture.hpp`
- Create: `hand_interaction/examples/hand_interaction_demo_capture.cpp`
- Create: `hand_interaction/tests/test_hand_interaction_demo_capture.cpp`
- Modify: `hand_interaction/CMakeLists.txt`

**Interfaces:**
- Produces: `CapturedFrame`, `CaptureCounters`, and `LatestFrameMailbox` with `publish(...) noexcept`, `take_latest(..., timeout)`, `close() noexcept`, and `counters() const noexcept`.
- Produces: RAII `CameraCapture` that enumerates/open modes, installs callbacks, starts, stops, and destroys in control-plane order.
- Consumes: Task 1 camera request and selected native mode.

- [x] **Step 1: Write failing mailbox tests.** Verify one-frame transfer, vector ownership swap, monotonically increasing serials, unread-frame coalescing, exact capacity rejection, malformed metadata rejection, timeout, close wakeup, and counters.
- [x] **Step 2: Build the test and verify RED.** Expect missing capture adapter symbols.
- [x] **Step 3: Implement the mutex-protected two-vector protocol.** Pre-reserve both sides before start, perform checked frame-length validation, never allocate above the configured limit, copy inside the callback, and hold the mutex only during copy/publish or swap/consume.
- [x] **Step 4: Implement RAII capture control.** Do not call stop/destroy from callbacks; map ERROR state to mailbox close; guarantee stop-before-destroy on every normal and exceptional exit.
- [x] **Step 5: Run focused tests GREEN and commit.** Use the CPU preset CTest directory; tests must not enumerate real hardware.

### Task 3: Frame conversion and deterministic overlay UI

**Files:**
- Create: `hand_interaction/examples/hand_interaction_demo_frame.hpp`
- Create: `hand_interaction/examples/hand_interaction_demo_frame.cpp`
- Create: `hand_interaction/examples/hand_interaction_demo_ui.hpp`
- Create: `hand_interaction/examples/hand_interaction_demo_ui.cpp`
- Create: `hand_interaction/tests/test_hand_interaction_demo_ui.cpp`
- Modify: `hand_interaction/CMakeLists.txt`

**Interfaces:**
- Produces: `cv::Mat to_bgr(const CapturedFrame&)` with owning packed `CV_8UC3` output.
- Produces: `DemoAction action_from_key(int) noexcept` and `compose_overlay(const cv::Mat&, const HandFrame&, const HandInteractionFrame&, const DemoMetrics&)`.
- Consumes: Task 2 owning frame plus existing public vision/hand-interaction value types.

- [x] **Step 1: Write failing conversion/UI tests.** Use literal 2x2 NV12/I420/BGRA/RGB fixtures; reject wrong byte counts, odd YUV dimensions, MJPEG, and empty frames. Verify Q/Escape/R keys, fixed-size BGR output, input preservation, landmark/box drawing, action text, and metrics text.
- [x] **Step 2: Build the test and verify RED.** Expect missing frame/UI source failures.
- [x] **Step 3: Implement strict conversions and overlay composition.** Use OpenCV Lite `cvtColor`, clone all returned matrices so no callback/mailbox view escapes, clamp drawn coordinates, and use named colors/layout constants.
- [x] **Step 4: Run focused tests GREEN and commit.** Link only the test to OpenCV Lite core/imgproc and TinyTest; no HighGUI or model runtime is needed.

### Task 4: CPU/TensorRT orchestration executable

**Files:**
- Create: `hand_interaction/examples/hand_interaction_demo.cpp`
- Modify: `hand_interaction/CMakeLists.txt`
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: executable `hand_interaction_demo` when `KFCORE_BUILD_HAND_INTERACTION_EXAMPLES=ON`.
- Consumes: Tasks 1-3, `KFCore::hand_models_cpu` and/or `KFCore::vision_models_tensorrt`, `KFCore::hand_interaction`, `Salts::Capture`, and `OpenCVLite::highgui`.

- [x] **Step 1: Add option dependency checks and executable target.** Require the official `Salts::Capture` imported target and fail configuration if the SDK was installed without capture exports.
- [x] **Step 2: Build and verify the executable initially fails.** Expect missing `main` orchestration until the next step.
- [x] **Step 3: Implement the event loop.** Handle list-only mode before model load; construct exactly one selected backend and HandPipeline, process the latest frame synchronously, feed a monotonic serial/time context to THIG, display overlay, reset both pipelines on R, and exit on Q/Escape/window close.
- [x] **Step 4: Preserve error and timing boundaries.** Catch exceptions only in `main`, print one actionable error, return failure, and display capture/model/THIG/end-to-end timings without per-frame INFO logging.
- [x] **Step 5: Build CPU and TensorRT variants.** CPU builds against the trusted ONNX directory; TensorRT configures and builds against the explicit local SDK while runtime engine smoke remains separate.
- [x] **Step 6: Commit.** Commit option, target, and executable after both compile paths are checked.

### Task 5: Presets, documentation, dependency repair, and verification

**Files:**
- Modify: `CMakeUserPresets.json`
- Modify: `hand_interaction/README.md`
- Modify: `docs/superpowers/plans/2026-08-28-thig-hand-interaction-demo.md`

**Interfaces:**
- Produces: `win-release-user` and `win-release-user` configure/build/test presets with OpenCV Lite and runtime DLL paths.
- Documents: complete list, CPU launch, TensorRT launch, controls, ownership, supported formats, and troubleshooting commands.

- [x] **Step 1: Repair the installed TurboParser capture export.** In the TurboParser `refactor/capture-serial-to-parser` worktree, run `cmake --preset win-capture-release-user`, build/test, then `cmake --build --preset install-win-capture-release-user`; verify a clean consumer sees `TARGET Salts::Capture`.
- [x] **Step 2: Add dedicated demo presets.** Keep existing vision presets unchanged; inherit their backend settings, enable only the example option, add `OPENCV_LITE_ROOT`, and include Turbo Capture/OpenCV/runtime directories in PATH.
- [x] **Step 3: Document complete commands and expected behavior.** Include `--list-cameras`, exact mode selection, CPU model directory, TensorRT engine flags, R/Q controls, and MJPEG rejection guidance.
- [x] **Step 4: Run focused and adjacent tests.** Configure/build the CPU demo preset; run demo CLI/capture/UI tests, hand interaction tests, vision pipeline tests, CPU real-model integration, and installed-consumer test.
- [x] **Step 5: Run hardware smoke verification.** Camera listing and an exact 640x480@30 NV12 CPU run completed 100 frames and exited normally via `--max-frames`; reset and quit key mappings are covered deterministically by TinyTest.
- [x] **Step 6: Verify TensorRT configuration/build.** TensorRT 11.2.1/CUDA 12.8 build and validation tests pass; no trusted hand engines were available, so GPU inference smoke is explicitly not claimed.
- [x] **Step 7: Final hygiene and commit.** Run `git diff --check`, `codegraph sync .`, inspect affected tests, confirm `.codegraph/` and build products are untracked/ignored, update plan checkboxes, and commit documentation/presets/verification evidence.

## Verification record (2026-08-28)

- TurboParser Capture migration: focused Capture tests 2/2 passed; installed package exports `Salts::Capture`.
- CPU dedicated preset: clean configure/build and 22/22 CTest tests passed, including real ONNX models and installed consumer.
- CPU camera smoke: Logitech BRIO mode 385, 640x480@30 NV12, model load 78.37 ms; captured 100, consumed 100, coalesced 0, rejected 0.
- TensorRT dedicated preset: TensorRT 11.2.1 and CUDA 12.8 configure/build passed; 28/28 tests passed; `--list-cameras` loaded the GPU executable and enumerated the BRIO.
- Opt-in boundary: configuring the CPU tree with the demo option disabled produced zero demo tests and no `hand_interaction_demo` target; restoring the preset re-enabled all three demo tests.
