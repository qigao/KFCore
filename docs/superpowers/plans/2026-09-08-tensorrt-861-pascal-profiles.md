# TensorRT 8.6.1 Pascal Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add isolated KFCore build and engine profiles for GTX 1060/1070 using TensorRT 8.6.1, CUDA 12.8, and SM 6.1.

**Architecture:** CMake presets remain the single source for SDK roots, CUDA architecture, install roots, and the compiled engine profile. TensorRT 8.6 compatibility is admitted through a narrow version gate while existing runtime tensor and alias validation remains unchanged. Engines are generated only on the matching Pascal deployment GPU.

**Tech Stack:** CMake 3.20 presets, Ninja, MSVC, CUDA 12.8, TensorRT 8.6.1, cuDNN 8.9, CTest/TinyTest.

**Spec:** `docs/superpowers/specs/2026-09-08-tensorrt-861-pascal-profiles-design.md`

## Global Constraints

- Never publish an engine built on the local RTX 4060 as an SM 6.1 engine.
- Keep `win-release-user` and its TensorRT 11.2.1 profile unchanged.
- Accept TensorRT 8.6.x, 10.x, and 11.x only; reject unverified versions.
- GTX 1060 and GTX 1070 use distinct build, install, and engine profile directories.
- Missing TensorRT, CUDA, cuDNN, model, binding, or shape requirements fail explicitly.
- No runtime backend fallback, public C++ API change, or model format migration is introduced.

---

### Task 1: Runtime and finder version contract

**Files:**
- Modify: `backends/tensorrt_cuda/runtime/tests/test_runtime_control.cpp`
- Modify: `backends/tensorrt_cuda/runtime/src/tensorrt_version.hpp`
- Modify: `cmake/FindTensorRT.cmake`

**Interfaces:**
- Produces: `constexpr bool tensorrt_version_supported(int major, int minor) noexcept`.
- Produces: configure-time acceptance of TensorRT 8.6.x without broadening to 8.0-8.5 or 9.x.

- [x] **Step 1: Add failing runtime version boundary tests**

Add literal cases requiring 8.6, 10.0, and 11.0 to be supported while 8.5, 9.0, and 12.0 are rejected. Require `validate_runtime_tensorrt_version(8, 5)` to throw `EngineContractMismatch` with a version-gate diagnostic.

- [x] **Step 2: Run the focused test and verify RED**

Run the existing `test_runtime_tensorrt_control` target/test from the normal TensorRT 11 build. Expected: failure because the runtime currently accepts 8.5 and 9.0.

- [x] **Step 3: Implement the minimum runtime version gate**

Implement `tensorrt_version_supported()` and call it before the existing alias-policy rejection. Preserve all existing supported-version and alias behavior.

- [x] **Step 4: Run the focused test and verify GREEN**

Rebuild and run `test_runtime_tensorrt_control`; require zero failures.

- [x] **Step 5: Narrowly extend FindTensorRT**

Change the configure-time gate to accept only TensorRT 8.6.x, 10.x, and 11.x, with an error message listing that exact range.

### Task 2: Isolated Pascal presets

**Files:**
- Modify: `CMakeUserPresets.json`

**Interfaces:**
- Produces: configure/build/test presets `win-gtx1060-release-user` and `win-gtx1070-release-user`.
- Produces: install build presets `install-win-gtx1060-release-user` and `install-win-gtx1070-release-user`.

- [x] **Step 1: Add a hidden TensorRT 8.6 Pascal dependency preset**

Inherit the existing Windows CUDA paths, override `TENSORRT_ROOT` with
`C:/projects/TensorRT-8.6.1`, define `CUDNN_ROOT` as
`$env{PKG_ROOT}/cudnn-8.9.7-cuda12`, and set `CMAKE_CUDA_ARCHITECTURES=61`.

- [x] **Step 2: Add GTX 1060 and GTX 1070 configure presets**

Use isolated binary/install directories and exact engine profiles
`gtx1060-sm61-trt8.6.1-default` and `gtx1070-sm61-trt8.6.1-default`. Runtime PATH includes CUDA,
TensorRT `lib`/`bin`, cuDNN `bin`, the active build `bin`, vcpkg, Salts, and inherited PATH.

- [x] **Step 3: Add build, test, and install entries**

Expose matching public entries and keep the `install-<profile>` naming contract.

- [x] **Step 4: Validate preset discovery**

Run `cmake --list-presets`, `cmake --build --list-presets`, and `ctest --list-presets`; require both Pascal profiles and install entries to appear.

### Task 3: TensorRT 8.6 configure and compile validation

**Files:**
- Modify: `backends/tensorrt_cuda/runtime/src/engine.cpp`
- Modify: `backends/tensorrt_cuda/runtime/src/executor.cpp`
- Modify: `backends/tensorrt_cuda/yolo/src/engine.cpp`

**Interfaces:**
- Consumes: `win-gtx1060-release-user` and real TensorRT/CUDA/cuDNN roots.
- Produces: compile evidence that KFCore TensorRT targets support the admitted SDK.

- [x] **Step 1: Configure the GTX 1060 profile**

Run `cmake --fresh --preset win-gtx1060-release-user` from `VsDevCmd.bat`. Require TensorRT 8.6.1 and CUDA architecture 61 in the generated cache.

- [x] **Step 2: Build the focused runtime and model targets**

Build `test_runtime_tensorrt_control`, `kfcore_runtime_tensorrt`, `kfcore_yolo_tensorrt`,
`kfcore_face_models_cuda`, and `kfcore_hand_models_cuda`. If TensorRT 8.6 lacks an API used by the current implementation, add the smallest compile-time version branch and a focused test where behavior is independently observable.

- [x] **Step 3: Run device-independent focused tests**

Run the runtime-control validation test. Do not run or claim Pascal inference on the RTX 4060.

- [x] **Step 4: Exercise TensorRT 8.6 with throwaway SM 8.9 plans**

Build clearly isolated `yolov8s` and Palm preflight plans, run the real YOLO and dynamic-output
integration paths, then delete the plans. This validates parser/runtime API behavior only and must
not be recorded or published as Pascal engine evidence.

### Task 4: Deployment documentation and handoff

**Files:**
- Modify: `README.md`
- Modify: `applications/tensorrt_cuda/README.md`
- Modify: `vision/core/yolo/README.md`
- Modify: `vision/core/hand_models/README.md`

**Interfaces:**
- Produces: exact Pascal build/install and per-model `trtexec` commands using the selected profile.

- [x] **Step 1: Document build profile selection and dependency roots**

Document the two presets, isolated package roots, cuDNN runtime dependency, and prohibition on generating SM 6.1 engines on a non-Pascal GPU.

- [x] **Step 2: Document face/YOLO engine generation**

Adapt the existing validated commands to an explicit `$profile`, TensorRT 8.6.1, and matching GTX host. Preserve FP32 I/O and existing fixed/dynamic shapes.

- [x] **Step 3: Document hand engine generation**

Preserve the existing Palm static input and hand/classifier dynamic batch profiles 1/2/8, followed by bounded load-engine smoke commands.

- [x] **Step 4: Record remaining target-host verification**

State that each GTX 1060/1070 engine must be generated and loaded on its matching host; do not record unexecuted results as facts.

### Task 5: Final verification

**Files:**
- Verify only.

**Interfaces:**
- Produces: reproducible configuration, build, test, diff, and impact evidence.

- [x] **Step 1: Re-run focused build/tests and preset discovery**

Run the exact Task 2/3 commands again and inspect their complete output.

- [x] **Step 2: Verify generated cache contracts**

Require TensorRT 8.6.1, CUDA architecture 61, the selected engine profile, and isolated install prefix in `CMakeCache.txt`.

- [x] **Step 3: Verify repository integrity and impact**

Run `git diff --check`, `git status --short`, `codegraph sync .`, and `codegraph affected` for modified CMake/runtime files. Confirm build trees, `.codegraph`, ONNX files, and engines are not staged.
