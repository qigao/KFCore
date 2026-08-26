# SIFT Extractor Module Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a KFCore-owned SIFT abstraction and an optional bounded PopSift adapter that consumes reusable ImageProcessor views.

**Architecture:** Extend ImageProcessor with explicit Host grayscale staging, then layer a dependency-inverted synchronous SIFT contract over it. Keep all PopSift headers, pointers, worker behavior, and configuration behind a Pimpl adapter and an opt-in CMake target.

**Tech Stack:** C++17, CUDA/CUDAToolkit, CMake presets/package exports, TurboUtils TinyTest, optional PopSift 0.10.x.

---

### Task 1: Add tested Host grayscale staging

**Files:**
- Modify: `image_processor/include/kfcore/image_processor/types.hpp`
- Modify: `image_processor/include/kfcore/image_processor/image_processor.hpp`
- Modify: `image_processor/src/image_processor.cpp`
- Test: `image_processor/tests/test_image_processor.cpp`
- Modify: `image_processor/README.md`

1. Add failing TinyTest cases for padded RGB/BGR, padded Gray8, invalid memory/stride/capacity, limits, and overflow.
2. Build `test_image_processor` and confirm the new tests fail to compile or run.
3. Add `Gray8`, a byte-count query, and bounded `stage_host_grayscale` implementation.
4. Rebuild and run CPU plus CUDA ImageProcessor tests.

### Task 2: Add the KFCore SIFT contract

**Files:**
- Create: `sift/include/kfcore/sift/error.hpp`
- Create: `sift/include/kfcore/sift/types.hpp`
- Create: `sift/include/kfcore/sift/sift_extractor.hpp`
- Create: `sift/src/sift_extractor.cpp`
- Create: `sift/tests/test_sift.cpp`
- Create: `sift/CMakeLists.txt`
- Create: `sift/README.md`

1. Add a failing public-contract test using a fake extractor.
2. Implement owned 128-float descriptors, FeatureSet, typed errors, and the synchronous strategy interface.
3. Build and run `test_sift`.

### Task 3: Add the optional PopSift adapter

**Files:**
- Create: `sift/include/kfcore/sift/popsift_extractor.hpp`
- Create: `sift/src/popsift_extractor.cpp`
- Create: `sift/tests/test_popsift_options.cpp`
- Modify: `sift/CMakeLists.txt`
- Modify: `sift/README.md`

1. Add adapter option/error tests independent of GPU execution.
2. Implement validated options, Pimpl, synchronous single-flight extraction, RAII for jobs/results, and orientation flattening.
3. Configure/build against the local PopSift source tree in an isolated binary directory; verify that a missing source root fails with an actionable error and record the runtime gap.

### Task 4: Integrate CMake package and presets

**Files:**
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`
- Modify: `cmake/KFCoreConfig.cmake.in`
- Modify: `CMakeUserPresets.json`
- Create: `cmake/tests/SiftInstalledConsumer/CMakeLists.txt`
- Create: `cmake/tests/SiftInstalledConsumer/main.cpp`
- Create: `cmake/tests/test_sift_installed_consumer.cmake`

1. Add opt-in `KFCORE_BUILD_SIFT` and `KFCORE_BUILD_SIFT_POPSIFT` dependency rules.
2. Export targets and conditional dependency metadata.
3. Add configure/build/test/install presets and installed-consumer test.
4. Run fresh configure, targeted builds/tests, install, and consumer test.

### Task 5: Review, verify, and publish

**Files:** all changed files.

1. Inspect diff for ownership, bounds, public dependency, and unrelated changes.
2. Request a code review and address findings with tests first.
3. Run the complete configured CTest suite twice if the known test-directory ordering issue appears.
4. Commit coherent changes, push `feature/sift-extractor`, and create a GitHub PR with evidence and remaining PopSift runtime risk.
