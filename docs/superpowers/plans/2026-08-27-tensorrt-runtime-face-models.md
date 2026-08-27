# Generic TensorRT Runtime and Face Model Adapters Implementation Plan

> Execute each task test-first and keep the existing YOLO implementation behavior unchanged.

## Task 1: Public runtime contract and pure validation

**Files:**

- Create `tensorrt_runtime/include/kfcore/tensorrt/{error,types,runtime}.hpp`
- Create `tensorrt_runtime/src/{error,tensor_validation}.{cpp,hpp}`
- Create `tensorrt_runtime/tests/test_tensor_validation.cpp`
- Create `tensorrt_runtime/CMakeLists.txt`
- Modify `CMakeOptions.cmake`, `CMakeLists.txt`

1. Add failing TinyTest cases for supported scalar sizes, checked shape bytes, profile bounds,
   duplicate/missing tensor views, and aggregate resource limits.
2. Build the test target and confirm the expected compile/test failures.
3. Implement the smallest pure validation layer and public value types needed by the tests.
4. Build and run `test_tensorrt_runtime_contract` until it passes.
5. Commit the runtime contract slice.

## Task 2: Engine loading, metadata, and executor

**Files:**

- Create `tensorrt_runtime/src/{engine,executor,engine_file,cuda_buffer,tensorrt_raii}.*`
- Create `tensorrt_runtime/tests/{test_engine_file,test_cuda_buffer}.cpp`
- Modify `tensorrt_runtime/CMakeLists.txt`

1. Add failing tests for exact engine reads and strong exception safety of device/pinned buffers.
2. Reuse the proven YOLO test patterns while translating them to the runtime error domain.
3. Implement bounded engine deserialization and metadata extraction for profile 0.
4. Implement synchronous host/device tensor execution, non-overlap guard, output-shape resolution,
   and cleanup synchronization.
5. Build and run all runtime unit tests.
6. Commit the executable runtime slice.

## Task 3: Strict face adapter contracts and typed results

**Files:**

- Create `face_models/include/kfcore/face_models/{error,types,tensorrt}.hpp`
- Create `face_models/src/{error,contracts,face68,arcface,age_gender}.*`
- Create `face_models/tests/{test_contracts,test_results}.cpp`
- Create `face_models/CMakeLists.txt`
- Modify `CMakeOptions.cmake`, `CMakeLists.txt`

1. Add failing pure tests for exact names/types/ranks/fixed dimensions/dynamic batch and output
   decoding sizes.
2. Implement construction-time contract validation and raw typed results.
3. Implement bounded reusable adapter output buffers and prepared-tensor inference entry points.
4. Build and run all face adapter unit tests.
5. Commit the face adapter slice.

## Task 4: Real-engine integration tests

**Files:**

- Create `tensorrt_runtime/tests/test_tensorrt_runtime_integration.cpp`
- Create `face_models/tests/test_face_models_integration.cpp`
- Modify `CMakeOptions.cmake`, `tensorrt_runtime/CMakeLists.txt`, `face_models/CMakeLists.txt`

1. Add cache paths for trusted ArcFace, age/gender, and Face68 engines.
2. Add configure-time path validation without downloading or rebuilding models implicitly.
3. Run zero-input smoke inference for available local engines and assert exact sizes and finite
   outputs. Exercise host and CUDA inputs in the generic runtime test.
4. Keep unavailable optional engines reported as skipped configuration, not passing fake tests.
5. Commit integration coverage.

## Task 5: Package export and documentation

**Files:**

- Modify `cmake/KFCoreConfig.cmake.in`, `cmake/FindTensorRT.cmake`
- Create `cmake/tests/TensorRtRuntimeInstalledConsumer/{CMakeLists.txt,main.cpp}`
- Create `cmake/tests/test_tensorrt_runtime_installed_consumer.cmake`
- Modify root/package CMake and relevant README model matrix

1. Add a failing installed-consumer CTest for `KFCore::tensorrt_runtime` and
   `KFCore::face_models`.
2. Export headers/targets/capability flags and dependency discovery.
3. Document the prepared-tensor boundary and list preprocessing still required by each model.
4. Run the installed-consumer test and the existing adjacent package tests.
5. Commit packaging and documentation.

## Task 6: Final verification and review

1. Configure from a fresh preset with TensorRT runtime, face models, YOLO, and integration tests.
2. Build all targets and run the smallest tests, then the complete configured CTest suite.
3. Inspect `git diff --check`, `git status`, and CodeGraph affected tests.
4. Request code review, address verified findings, rerun affected tests, and only then prepare the
   branch for push/PR.

