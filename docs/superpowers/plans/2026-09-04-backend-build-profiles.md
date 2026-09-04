# Backend Build Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make KFCore build CUDA/TensorRT by default, offer an isolated CPU/ONNX profile, and stop disabled backends from contributing dependencies, targets, examples, or installed DLLs.

**Architecture:** CMake options are the single backend-selection source. Top-level traversal and package dependency discovery follow those options; backend-switching demos receive private compile definitions derived from the same values. Existing public C++ APIs and target names remain intact when their backend is enabled.

**Tech Stack:** CMake 3.20 presets, Ninja, MSVC/CUDA, CTest, C++17, ONNX Runtime, TensorRT.

**Spec:** `docs/superpowers/specs/2026-09-04-backend-build-profiles-design.md`

## Global Constraints

- `KFCORE_ENABLE_CUDA` defaults to `ON`.
- `KFCORE_ENABLE_ONNX_CPU` defaults to `OFF`.
- Disabled backends are not discovered, built, exported, or required by installed consumers.
- Existing target names and public C++ APIs remain unchanged when enabled.
- Missing requested backends fail explicitly; no runtime fallback is added.
- CUDA-only and CPU-only presets use separate `build/cuda` and `build/cpu` roots and separate
  `kfcore/<configuration>/cuda` and `kfcore/<configuration>/cpu` install roots.

---

### Task 1: Backend options and conditional target graph

**Files:**
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`
- Modify: `sift/CMakeLists.txt`

**Interfaces:**
- Produces: CMake cache booleans `KFCORE_ENABLE_CUDA` and `KFCORE_ENABLE_ONNX_CPU`.
- Produces: a CUDA-only default target graph and an ONNX-only opt-in graph.

- [x] **Step 1: Verify the current graph ignores backend selection**

Run a configure with `-DKFCORE_ENABLE_ONNX_CPU=OFF`, then inspect `build.ninja` for
`kfcore_runtime_onnx.dll`. Expected before implementation: the rule is still present.

- [x] **Step 2: Add the two cache options**

Add exact option declarations to `CMakeOptions.cmake`:

```cmake
option(KFCORE_ENABLE_CUDA "Build CUDA and TensorRT backends" ON)
option(KFCORE_ENABLE_ONNX_CPU "Build ONNX Runtime CPU backends" OFF)
```

- [x] **Step 3: Gate languages, dependencies, subdirectories, and find modules**

Keep C++ enabled for the common vision modules. Enable CUDA and find CUDAToolkit/TensorRT only
inside `if(KFCORE_ENABLE_CUDA)`. Find ONNX Runtime only inside
`if(KFCORE_ENABLE_ONNX_CPU)`. Add backend subdirectories only within their matching block and
install only the matching find module.

- [x] **Step 4: Gate PopSift behind CUDA**

Keep `KFCore::sift` unconditional. Wrap the PopSift vendor target, adapter, license/revision install,
and PopSift-specific tests in `if(KFCORE_ENABLE_CUDA)`.

- [x] **Step 5: Reconfigure and inspect target absence**

Run `cmake --fresh --preset win-release-user`. Inspect `build.ninja` with `rg.exe` and require zero
matches for the five ONNX CPU DLL outputs: runtime, YOLO, face models, hand models, and face
applications.

### Task 2: Backend-aware examples and tests

**Files:**
- Modify: `vision/core/yolo/CMakeLists.txt`
- Modify: `vision/core/yolo/examples/yolov8_domain_demo.cpp`
- Modify: `hand_interaction/CMakeLists.txt`
- Modify: `hand_interaction/examples/hand_interaction_demo.cpp`
- Modify: `hand_interaction/examples/hand_interaction_demo_face.cpp`
- Modify: `hand_interaction/tests/CMakeLists.txt`
- Test: `vision/core/yolo/tests/test_yolo_domain_cli.cpp`
- Test: `hand_interaction/tests/test_hand_interaction_demo_cli.cpp`

**Interfaces:**
- Consumes: `KFCORE_ENABLE_CUDA` and `KFCORE_ENABLE_ONNX_CPU`.
- Produces: private integer compile definitions `KFCORE_DEMO_ENABLE_CUDA` and
  `KFCORE_DEMO_ENABLE_ONNX_CPU` for dual-backend examples.

- [x] **Step 1: Run existing CLI availability tests as the red/green behavioral contract**

Run `test_yolo_domain_cli` and `test_hand_interaction_demo_cli`; record that their availability
matrix already requires unique-backend auto-selection and unavailable-backend rejection. The new
failure to drive is compilation of those examples with one backend target absent.

- [x] **Step 2: Make YOLO example includes, fields, and links conditional**

Derive `compiled_backends()` from the two compile definitions. Compile only the selected detector
members and branches. Build its link list from targets that exist, preserving the existing CLI
errors for unavailable choices.

- [x] **Step 3: Make hand example includes, factories, and links conditional**

Derive `backend_availability()` from the same definitions. Guard CPU/TensorRT model construction,
TensorRT input staging, and face pipeline construction. Keep the CLI parser test backend-neutral
by removing model libraries it does not call.

- [x] **Step 4: Build and test the CUDA-only examples**

Build `yolov8_domain_demo`, `hand_interaction_demo`, `face_swap_image`,
`test_yolo_domain_cli`, and `test_hand_interaction_demo_cli`. Run the two CLI tests with CTest and
expect both to pass.

### Task 3: Conditional installed-package dependencies

**Files:**
- Modify: `cmake/KFCoreConfig.cmake.in`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: installed variables `KFCore_CUDA_ENABLED` and `KFCore_ONNX_CPU_ENABLED`.
- Produces: conditional CUDAToolkit/TensorRT or ONNX Runtime dependency discovery.

- [x] **Step 1: Inspect the generated CUDA-only config before implementation**

Open `build/Msvc-Release/KFCoreConfig.cmake`. Expected before implementation: unconditional
`find_dependency(ONNXRuntime REQUIRED MODULE)` remains.

- [x] **Step 2: Configure backend facts into the package config**

Set the two installed boolean variables from the build options. Wrap CUDA and ONNX dependency
discovery in their corresponding conditions while leaving Salts and trackers unchanged.

- [x] **Step 3: Verify generated config and exported targets**

Reconfigure the CUDA profile. Require the generated config to contain the false ONNX fact and
conditional dependency block, and require the generated export set to omit all ONNX targets.

### Task 4: Isolated CPU preset and documentation

**Files:**
- Modify: `CMakeUserPresets.json`
- Modify: `README.md`

**Interfaces:**
- Produces: configure/build/test presets named `win-cpu-release-user`.
- Produces: install build presets for supported user profiles.

- [x] **Step 1: Add a CPU path base without CUDA compiler configuration**

Move CUDA-only cache variables from `win-paths` into a new `win-cuda-paths` preset. Keep common
vcpkg/toolchain/architecture state in `win-paths`; make existing Windows presets inherit
`win-cuda-paths`. Move CUDA preset binary directories below `build/cuda/` and their install prefixes
below `$env{PKG_ROOT}/kfcore/<configuration>/cuda/`.

- [x] **Step 2: Add the isolated CPU release profile**

Create `win-cpu-release-user` with binary directory `build/cpu/Msvc-Release`, install prefix
`$env{PKG_ROOT}/kfcore/release/cpu`, CUDA OFF, ONNX CPU ON, and a PATH containing ONNX Runtime but
not CUDA or TensorRT. Add matching build and test entries.

- [x] **Step 3: Add install entry presets**

Add `install-<profile>` build presets whose target is `install`, including the new CPU profile, so
installation uses the documented preset path.

- [x] **Step 4: Document selection and deployment behavior**

Update README build options and examples to state that CUDA is default, CPU is opt-in, build and
install trees are split under `cuda`/`cpu`, CUDA packages do not require `onnxruntime.dll`, and
runtime switching requires a separate both-enabled `hybrid` build.

- [x] **Step 5: Validate preset discovery**

Run `cmake --list-presets`, `cmake --build --list-presets`, and `ctest --list-presets`. Require all
three CPU entries and install entries to appear.

### Task 5: Backend matrix verification

**Files:**
- Verify only; no production files added.

**Interfaces:**
- Consumes: all outputs of Tasks 1-4.
- Produces: reproducible build, test, install, and artifact-count evidence.

- [x] **Step 1: Verify CUDA default**

Fresh-configure `win-release-user`, build it, run its complete CTest suite, then install through
`install-win-release-user`. Confirm no ONNX targets or KFCore ONNX DLLs exist in the generated graph
or fresh install manifest.

- [x] **Step 2: Verify CPU-only**

Fresh-configure `win-cpu-release-user`, build it, run its complete CTest suite, then install through
`install-win-cpu-release-user`. Confirm no CUDA/TensorRT targets or KFCore CUDA DLLs exist in the
generated graph or fresh install manifest.

- [x] **Step 3: Verify repository integrity**

Run `git diff --check`, `git status --short`, and `codegraph affected` for the modified build and
example files. Confirm `.codegraph/`, build trees, junctions, DLLs, engines, and ONNX assets are not
staged.

- [x] **Step 4: Commit the implementation**

Stage only the spec, plan, CMake, preset, example, test-link, and README changes. Commit with:

```text
build: make cuda the default inference backend
```
