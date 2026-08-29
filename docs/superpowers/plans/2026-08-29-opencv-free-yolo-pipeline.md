# OpenCV-free YOLO pipeline implementation plan

> Execute inline in this worktree. Use test-driven development for each behavior change and run the
> verification commands before claiming completion.

## Task 1: Define and validate packed native capture formats

Files:

- Modify `image_processor/include/kfcore/image_processor/types.hpp`
- Modify `image_processor/src/image_processor.cpp`
- Modify `image_processor/tests/test_image_processor.cpp`
- Modify `tensorrt_yolo/include/kfcore/yolo/types.hpp`
- Modify `tensorrt_yolo/src/detector_helpers.cpp`
- Modify `tensorrt_yolo/tests/test_tensorrt_detection_helpers.cpp`

Steps:

1. Add failing tests for exact NV12/I420 layout, odd dimensions, insufficient capacity and stride.
2. Run the focused CPU/helper tests and confirm the expected unsupported-format failures.
3. Add NV12/I420 enum values and central packed-layout validation/staging calculations.
4. Map YOLO native formats into ImageProcessor formats without changing detection dimensions.
5. Re-run focused tests.

## Task 2: Add fused CPU native preprocessing and mirror

Files:

- Modify `image_processor/include/kfcore/image_processor/types.hpp`
- Modify `image_processor/include/kfcore/image_processor/cpu.hpp`
- Modify `image_processor/src/cpu.cpp`
- Modify `image_processor/tests/test_cpu_image_processor.cpp`

Steps:

1. Add failing black/white/neutral-color, I420/NV12 equivalence, mirror and letterbox tests.
2. Implement bounded BT.601 limited-range YUV sampling and fuse it into `letterbox_nchw`.
3. Append `mirror_horizontal` to preprocessing options and apply it in source-coordinate mapping.
4. Extend owned BGR conversion for display use without adding a second inference conversion.
5. Run focused CPU tests and a release benchmark for legacy versus fused preprocessing.

## Task 3: Add fused CUDA native preprocessing

Files:

- Modify `image_processor/src/image_processor_cuda.cu`
- Modify `image_processor/tests/test_image_processor_cuda.cu`

Steps:

1. Add failing CUDA parity tests for NV12, I420 and mirror.
2. Extend host staging for packed native layouts.
3. Add device YUV sampling/conversion to the existing preprocessing kernel path.
4. Compare CUDA output with CPU output using explicit tolerance.
5. Run focused CUDA tests and benchmark upload plus kernel time.

## Task 4: Route CPU and TensorRT detectors directly from native frames

Files:

- Modify `tensorrt_yolo/examples/yolo_domain_onnx.cpp`
- Modify `tensorrt_yolo/examples/yolov8_domain_demo.cpp`
- Modify `tensorrt_yolo/tests/test_yolo_domain_onnx.cpp`
- Modify `tensorrt_yolo/tests/test_yolo_domain_onnx_integration.cpp`
- Modify `tensorrt_yolo/tests/test_tensorrt_detection_helpers.cpp`

Steps:

1. Add failing tests that send NV12/I420 views into CPU and TensorRT preparation paths.
2. Update source-span validation and native format mapping.
3. Build image views directly over the consumed capture frame and pass mirror as preprocessing
   configuration.
4. Keep file inputs as packed BGR views.
5. Run detector helper and ONNX tests, then an available model integration test.

## Task 5: Replace application codecs and frame conversion

Files:

- Replace `tensorrt_yolo/examples/yolo_domain_frame.hpp`
- Replace `tensorrt_yolo/examples/yolo_domain_frame.cpp`
- Modify `tensorrt_yolo/tests/test_yolo_domain_ui.cpp`

Steps:

1. Add failing tests for owning BGR decode/encode boundaries and native frame conversion.
2. Implement the stb-backed bounded codec adapter and BGR owner/view helpers.
3. Implement NV12/I420/RGB/BGRA-to-BGR conversion for display only.
4. Verify malformed files, unsupported extensions and capacity failures are explicit.
5. Run the frame/UI test target.

## Task 6: Replace overlay and interactive window

Files:

- Modify `tensorrt_yolo/examples/yolo_domain_ui.hpp`
- Modify `tensorrt_yolo/examples/yolo_domain_ui.cpp`
- Add `tensorrt_yolo/examples/yolo_domain_window.hpp`
- Add `tensorrt_yolo/examples/yolo_domain_window_win32.cpp`
- Add `tensorrt_yolo/examples/yolo_domain_window_stub.cpp`
- Modify `tensorrt_yolo/tests/test_yolo_domain_ui.cpp`

Steps:

1. Convert overlay tests to assert packed BGR mutations and unchanged tracking facts.
2. Implement clipped boxes, dark label backgrounds and high-contrast text using bounded software
   drawing.
3. Implement a Win32 presenter with event polling and BGR `StretchDIBits` presentation.
4. Implement a non-Windows fail-fast presenter while keeping headless paths dependency-free.
5. Run UI tests and manually verify key mapping and readable text on Windows.

## Task 7: Remove OpenCV from the application target and update docs

Files:

- Modify `tensorrt_yolo/CMakeLists.txt`
- Modify `tensorrt_yolo/examples/yolov8_domain_demo.cpp`
- Modify `tensorrt_yolo/README.md`
- Modify relevant preset comments only if configuration semantics changed

Steps:

1. Remove unconditional OpenCV discovery/linking from `KFCORE_BUILD_YOLO_APPLICATIONS`.
2. Keep `KFCORE_BUILD_YOLO_OPENCV` and `track_image_sequence` isolated and unchanged.
3. Add stb implementation sources exactly once and platform window sources conditionally.
4. Update usage and ownership/performance documentation.
5. Configure from a fresh preset and inspect the demo link dependencies to confirm OpenCV is absent.

## Task 8: Full verification and performance record

Commands (from the worktree in a Visual Studio developer environment with
`TENSORRT_ROOT=C:\projects\TensorRT-11.2.1.2`):

```powershell
cmake --fresh --preset win-yolov8-applications-release-user
cmake --build --preset win-yolov8-applications-release-user
ctest --preset win-yolov8-applications-release-user --output-on-failure
```

Then run the focused 640 x 480 NV12 benchmark and one CPU ONNX plus one TensorRT engine smoke test.
Record model, resolution, iterations, warmup, timing boundary, CPU/GPU and build type. Check
`git diff --check`, confirm `.codegraph`, build trees, model files and junctions are untracked or
ignored, and review the final diff before commit/push.
