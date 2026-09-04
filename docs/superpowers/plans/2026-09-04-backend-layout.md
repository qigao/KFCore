# Backend-Oriented Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group ONNX Runtime CPU and TensorRT CUDA implementations by backend without changing KFCore's public targets, headers, runtime dependencies, or inference behavior.

**Architecture:** This is a source-tree-only refactoring. Backend-neutral contracts remain separate from the two backend implementations; the root `CMakeLists.txt` owns traversal and affected child CMake files update only their private paths to moved sibling implementation sources. The existing `KFCore::*` aliases, installed headers, model assets, and `KFCoreConfig.cmake.in` remain unchanged.

**Tech Stack:** CMake 3.20+, C++17, CUDA, TensorRT, ONNX Runtime, Salts TinyTest.

**Spec:** `docs/superpowers/specs/2026-09-04-backend-layout-design.md`

## Global Constraints

- Keep existing target names and all `KFCore::*` aliases unchanged.
- Keep installed include paths and public C++ names unchanged.
- Do not alter ONNX Runtime, TensorRT, CUDA, target link dependencies, or package-config dependency behavior.
- Preserve `.onnx` assets for ONNX Runtime CPU and `.engine` assets for TensorRT CUDA.
- Use documented CMake user presets from a Visual Studio developer environment on Windows.

---

### Task 1: Establish the behavioral baseline

**Files:**
- Read: `CMakeUserPresets.json`
- Read: `runtime_onnx/tests/CMakeLists.txt`
- Read: `runtime_tensorrt/tests/CMakeLists.txt`
- Read: `face_models_cpu/tests/CMakeLists.txt`
- Read: `face_models_cuda/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: existing `KFCore::runtime_onnx`, `KFCore::runtime_tensorrt`, `KFCore::face_models_cpu`, and `KFCore::face_models_cuda` targets.
- Produces: a passing baseline for the target names and their CPU/TensorRT test registrations.

- [x] **Step 1: Configure the existing tree**

Run: `cmake --fresh --preset win-release-user` from a Visual Studio developer environment.

Expected: configure completes and writes the `build/Msvc-Release` tree.

- [x] **Step 2: Build the affected libraries and unit-test executables**

Run: `cmake --build --preset win-release-user --target kfcore_runtime_onnx kfcore_runtime_tensorrt kfcore_face_models_cpu kfcore_face_models_cuda test_runtime_onnx_api test_cpu_face_models_api test_face_model_contracts test_face_swap_adapter_api`.

Expected: each named target builds successfully.

- [x] **Step 3: Run the focused tests**

Run: `ctest --preset win-release-user -R "^(test_runtime_onnx_api|test_cpu_face_models_api|test_face_model_contracts|test_face_swap_adapter_api)$" --output-on-failure`.

Expected: all four tests pass; this run is the behavior baseline for the file moves.

### Task 2: Move backend-neutral and backend-specific source trees

**Files:**
- Move: `runtime_onnx/` to `backends/onnx_cpu/runtime/`
- Move: `yolo_onnx/` to `backends/onnx_cpu/yolo/`
- Move: `face_models_cpu/` to `backends/onnx_cpu/face_models/`
- Move: `hand_models_cpu/` to `backends/onnx_cpu/hand_models/`
- Move: `runtime_tensorrt/` to `backends/tensorrt_cuda/runtime/`
- Move: `yolo_tensorrt/` to `backends/tensorrt_cuda/yolo/`
- Move: `face_models_cuda/` to `backends/tensorrt_cuda/face_models/`
- Move: `hand_models_cuda/` to `backends/tensorrt_cuda/hand_models/`
- Move: `image_processor_core/` to `vision/core/image_processor/`
- Move: `yolo_core/` to `vision/core/yolo/`
- Move: `face_model_core/` to `vision/core/face_models/`
- Move: `hand_model_core/` to `vision/core/hand_models/`
- Move: `image_processor_cpu/` to `vision/image/cpu/`
- Move: `image_processor_cuda/` to `vision/image/cuda/`
- Move: `face_applications_cpu/` to `applications/onnx_cpu/`
- Move: `face_applications_cuda/` to `applications/tensorrt_cuda/`
- Modify: `backends/onnx_cpu/yolo/CMakeLists.txt`
- Modify: `backends/onnx_cpu/yolo/tests/CMakeLists.txt`
- Modify: `backends/onnx_cpu/face_models/CMakeLists.txt`
- Modify: `backends/onnx_cpu/hand_models/CMakeLists.txt`
- Modify: `backends/tensorrt_cuda/yolo/CMakeLists.txt`
- Modify: `backends/tensorrt_cuda/face_models/CMakeLists.txt`
- Modify: `backends/tensorrt_cuda/hand_models/CMakeLists.txt`
- Modify: `backends/tensorrt_cuda/yolo/tests/CMakeLists.txt`
- Modify: `backends/tensorrt_cuda/face_models/tests/CMakeLists.txt`
- Modify: `vision/image/cpu/CMakeLists.txt`
- Modify: `vision/image/cuda/CMakeLists.txt`
- Modify: `vision/core/yolo/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: source directories containing each existing target CMake definition.
- Produces: the target layout specified by the design without altering source
  files, public headers, target names, or target link dependencies. Child
  CMake files retain their definitions and change only private paths to moved
  sibling source directories.

- [x] **Step 1: Move the ONNX Runtime CPU directories with Git-aware renames**

Run: `git mv runtime_onnx backends/onnx_cpu/runtime`, `git mv yolo_onnx backends/onnx_cpu/yolo`, `git mv face_models_cpu backends/onnx_cpu/face_models`, and `git mv hand_models_cpu backends/onnx_cpu/hand_models` after creating the parent directory.

Expected: `git status --short` reports renames; every child `CMakeLists.txt` moves with its implementation and tests.

- [x] **Step 2: Move the TensorRT CUDA directories with Git-aware renames**

Run: `git mv runtime_tensorrt backends/tensorrt_cuda/runtime`, `git mv yolo_tensorrt backends/tensorrt_cuda/yolo`, `git mv face_models_cuda backends/tensorrt_cuda/face_models`, and `git mv hand_models_cuda backends/tensorrt_cuda/hand_models` after creating the parent directory.

Expected: TensorRT code remains physically isolated from ONNX Runtime code and no target name changes.

- [x] **Step 3: Move shared vision and application directories with Git-aware renames**

Run the `git mv` operations listed above for `vision/core`, `vision/image`, and `applications`.

Expected: all common contracts and image implementations remain distinct from backend-specific inference modules.

- [x] **Step 4: Update private CMake paths to the moved sibling implementations**

Replace each old `${CMAKE_SOURCE_DIR}` directory reference with its new
`backends/` or `vision/` path. Replace the moved Yolo test paths that traverse
to `yolo_core` or `trackers` with `${CMAKE_SOURCE_DIR}/vision/core/yolo/src`,
`${CMAKE_SOURCE_DIR}/trackers/include`, and `${CMAKE_SOURCE_DIR}/trackers/tests`.

Expected: every build-time private include path resolves to the same source
files as before the reorganization.

### Task 3: Reconnect the top-level build graph

**Files:**
- Modify: `CMakeLists.txt:64-82`
- Modify: the child CMake files listed in Task 2.

**Interfaces:**
- Consumes: each moved directory's target definition and updated private
  sibling source paths.
- Produces: the original complete target graph and exported aliases at their new source paths.

- [x] **Step 1: Replace the existing model and image `add_subdirectory()` calls**

Update the traversal block to add `vision/core/image_processor`, `vision/image/cpu`, `vision/image/cuda`, `vision/core/yolo`, `vision/core/face_models`, and `vision/core/hand_models`, then add the CPU and TensorRT backend paths plus the application paths.

Expected: every old source-tree directory reference is removed and each target still has one CMake definition.

- [x] **Step 2: Reconfigure from scratch**

Run: `cmake --fresh --preset win-release-user` from a Visual Studio developer environment.

Expected: CMake locates every moved child project, preserves all existing aliases, and emits no missing-directory error.

### Task 4: Verify backend separation and compatibility

**Files:**
- Verify: `CMakeLists.txt`
- Verify: moved CPU/TensorRT test CMake files

**Interfaces:**
- Consumes: the reconfigured target graph.
- Produces: build and test evidence that the source reorganization preserves behavior.

- [x] **Step 1: Build the affected target set**

Run: `cmake --build --preset win-release-user --target kfcore_runtime_onnx kfcore_runtime_tensorrt kfcore_face_models_cpu kfcore_face_models_cuda test_runtime_onnx_api test_cpu_face_models_api test_face_model_contracts test_face_swap_adapter_api`.

Expected: all targets build with unchanged names after source-tree relocation.

- [x] **Step 2: Run the focused CPU and TensorRT tests**

Run: `ctest --preset win-release-user -R "^(test_runtime_onnx_api|test_cpu_face_models_api|test_face_model_contracts|test_face_swap_adapter_api)$" --output-on-failure`.

Expected: all four tests pass.

- [x] **Step 3: Inspect the change set**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; implementation files appear as renames plus the one intentional top-level CMake traversal edit.

- [x] **Step 4: Commit**

Run: `git add CMakeLists.txt backends vision applications docs/superpowers/specs/2026-09-04-backend-layout-design.md docs/superpowers/plans/2026-09-04-backend-layout.md && git commit -m "refactor: group inference backends by runtime"`.

Expected: a single commit contains only the layout migration and its design record.
