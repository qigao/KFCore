# Backend-Neutral Runtime Plugin ABI + YOLO Proof Implementation Plan

**Date:** 2026-09-14

**Design spec:** `docs/superpowers/specs/2026-09-14-backend-neutral-runtime-plugin-abi-design.md`

## Goal

Prove the backend-neutral runtime architecture end to end without rewriting KFCore: introduce a shared runtime contract and stable C plugin ABI, adapt the existing TensorRT and ONNX Runtime implementations behind that boundary, resolve artifacts through an explicit model package and execution policy, and migrate YOLO as the first typed domain model. The proof is complete only when ONNX Runtime and TensorRT can coexist in one process and the same YOLO semantic API can execute through either backend.

This plan intentionally stops after the YOLO proof. RTMW, Face, and Hand migration are follow-on plans after this acceptance gate is green.

## Architectural invariants

- Keep domain semantics in typed KFCore model code. Backend plugins execute tensors; they do not know YOLO, RTMW, Face, or Hand semantics.
- Preserve the existing TensorRT `Engine -> Executor` implementation as the first backend adapter rather than rewriting it.
- Treat a loaded model as immutable/shareable and an execution context as mutable/non-concurrent unless a backend explicitly documents stronger guarantees.
- Backend identity and device/provider identity are separate concepts: `tensorrt` vs `onnxruntime`, and `cuda:0` / `cpu` as selectable execution devices/providers.
- No implicit fallback. Fallback order is represented by an explicit `ExecutionPolicy`.
- No arbitrary plugin-directory scanning in v1. Plugins are loaded from explicit paths supplied by the application/test.
- No new JSON library. `model.json` parsing uses the existing SaltsUtils direct JSON/data-bind facilities already depended on by KFCore.
- C++ exceptions, STL containers, allocators, CUDA runtime objects, ONNX Runtime objects, and TensorRT objects never cross the plugin C ABI.
- Existing runtime-specific APIs may coexist during migration, but the YOLO proof must stop depending on them as the primary domain API before cleanup begins.

## Verification convention

Use dedicated build directories so architecture work does not depend on a developer's existing cache.

Before Task 10, configure a runtime-only test build with the current option names:

```bash
cmake -S . -B build/runtime-neutral -G Ninja \
  -DBUILD_TESTS=ON \
  -DKFCORE_ENABLE_CUDA=OFF \
  -DKFCORE_ENABLE_ONNX_CPU=OFF \
  -DKFCORE_ENABLE_ONNX_CUDA=OFF
```

The repository's existing Salts/SaltsUtils discovery environment must already be configured, as required by the current root `CMakeLists.txt`.

After Task 10, use the coexistence build:

```bash
cmake -S . -B build/backend-coexistence -G Ninja \
  -DBUILD_TESTS=ON \
  -DKFCORE_ENABLE_CUDA=ON \
  -DKFCORE_ENABLE_TENSORRT=ON \
  -DKFCORE_ENABLE_ONNXRUNTIME=ON
```

For every task, first run the focused test and observe the intended RED state before writing the production change. After GREEN, rerun the nearest existing runtime/YOLO tests touched by that task before committing.

---

## Task 1: Freeze the backend-independent YOLO semantic contract

**Purpose:** Prevent the architecture migration from silently changing YOLO behavior.

**Files:**

- Add: `vision/core/yolo/tests/yolo_semantic_contract.hpp`
- Modify: `backends/onnx_cpu/yolo/tests/test_yolo_onnx_integration.cpp`
- Modify: `backends/tensorrt_cuda/yolo/tests/test_tensorrt_integration.cpp`
- Modify: backend YOLO test `CMakeLists.txt` files only if the shared helper requires include-path exposure

**RED:**

1. Add a shared assertion helper that accepts `kfcore::yolo::DetectionFrame` and validates backend-independent invariants:
   - source dimensions are preserved;
   - boxes are finite and normalized to the source image coordinate frame used by existing KFCore YOLO APIs;
   - `x2 >= x1`, `y2 >= y1`;
   - confidence values are finite and within the currently supported score range;
   - class IDs and labels remain domain data rather than runtime metadata;
   - no backend/runtime identifier appears in `DetectionFrame`.
2. Call the helper from both existing ORT and TensorRT integration tests using their current real-model fixtures.
3. Deliberately assert one invariant before adapting both tests so the focused test demonstrates the helper is actually executed.

**Focused commands:**

```bash
cmake --build build/onnx-cpu --target kfcore_yolo_onnx_tests
ctest --test-dir build/onnx-cpu -R "yolo_onnx" --output-on-failure
cmake --build build/tensorrt-cuda --target kfcore_yolo_tensorrt_tests
ctest --test-dir build/tensorrt-cuda -R "tensorrt.*yolo|yolo.*tensorrt" --output-on-failure
```

Use the repository's existing configured backend build directories if their names differ; do not change implementation merely to satisfy a guessed target name.

**GREEN:** Make only the test-side adjustments needed for both current backends to satisfy the same semantic helper. Do not introduce the new runtime abstraction yet.

**Commit:**

```text
test: freeze backend-independent yolo semantics
```

---

## Task 2: Introduce the backend-neutral runtime value types

**Purpose:** Establish the common vocabulary currently duplicated by `kfcore::tensorrt` and `kfcore::runtime_onnx`.

**Files:**

- Add: `runtime/core/CMakeLists.txt`
- Add: `runtime/core/include/kfcore/runtime/types.hpp`
- Add: `runtime/core/include/kfcore/runtime/error.hpp`
- Add: `runtime/core/src/error.cpp`
- Add: `runtime/core/tests/CMakeLists.txt`
- Add: `runtime/core/tests/test_types.cpp`
- Modify: root `CMakeLists.txt`

**RED:** Add compile/runtime tests for these neutral types before defining them:

```cpp
namespace kfcore::runtime {

enum class DataType {
    Float32,
    Float16,
    Int8,
    Int32,
    Int64,
    UInt8,
    Bool,
    BFloat16,
};

enum class MemoryKind {
    Host,
    PinnedHost,
    Device,
};

struct DeviceRef {
    std::string backend_id;
    std::string device_id;
};

struct TensorShape {
    std::vector<std::int64_t> dims;
};

struct TensorDescriptor {
    std::string name;
    DataType type;
    TensorShape shape;
    bool is_input;
};

struct TensorView {
    std::string_view name;
    DataType type;
    TensorShape shape;
    const void* data;
    std::size_t byte_size;
    MemoryKind memory_kind;
    std::string_view device_id;
};

struct MutableTensorView {
    std::string_view name;
    DataType type;
    TensorShape shape;
    void* data;
    std::size_t byte_size;
    MemoryKind memory_kind;
    std::string_view device_id;
};

}
```

Test at minimum:

- all required `DataType` values exist;
- host tensors do not require a device ID;
- device tensors can carry a device ID without importing CUDA headers;
- the header compiles without ONNX Runtime or TensorRT includes.

**Build integration:**

- Ensure CXX is enabled for the new runtime core independently of CUDA.
- Export a target named `kfcore_runtime_core` with alias `KFCore::runtime_core`.
- Keep the target free of TensorRT, ONNX Runtime, and CUDA link dependencies.

**Focused commands:**

```bash
cmake -S . -B build/runtime-neutral -G Ninja \
  -DBUILD_TESTS=ON \
  -DKFCORE_ENABLE_CUDA=OFF \
  -DKFCORE_ENABLE_ONNX_CPU=OFF \
  -DKFCORE_ENABLE_ONNX_CUDA=OFF
cmake --build build/runtime-neutral --target kfcore_runtime_core_tests
ctest --test-dir build/runtime-neutral -R "runtime_core" --output-on-failure
```

**GREEN:** Implement only the neutral value types and runtime error category needed by the tests. Do not migrate backend types in this task.

**Commit:**

```text
feat: add backend-neutral runtime core types
```

---

## Task 3: Add an explicit-path cross-platform dynamic library loader

**Purpose:** Give the host a small, testable mechanism to load runtime backend modules without creating an auto-discovery framework.

**Files:**

- Add: `runtime/core/src/dynamic_library.hpp`
- Add: `runtime/core/src/dynamic_library.cpp`
- Add: `runtime/core/tests/fixtures/test_backend_module.cpp`
- Add: `runtime/core/tests/test_dynamic_library.cpp`
- Modify: `runtime/core/CMakeLists.txt`
- Modify: `runtime/core/tests/CMakeLists.txt`

**RED:** Tests must cover:

- loading a test module from an explicit filesystem path;
- resolving one known exported symbol;
- deterministic error for a missing module;
- deterministic error for a missing symbol;
- RAII unload after the loader object is destroyed;
- no directory scanning API exists.

**Implementation:**

- Windows: `LoadLibraryW` / `GetProcAddress` / `FreeLibrary`.
- POSIX: `dlopen` / `dlsym` / `dlclose`.
- Convert native loader failures into `kfcore::runtime` errors at the C++ host boundary.
- Keep the loader internal to `runtime/core`; do not expose platform handles in public headers.

**Focused commands:**

```bash
cmake --build build/runtime-neutral --target kfcore_runtime_core_tests
ctest --test-dir build/runtime-neutral -R "runtime_core.*dynamic|dynamic.*runtime_core" --output-on-failure
```

**GREEN:** The test fixture loads and unloads on each supported host platform without backend dependencies.

**Commit:**

```text
feat: add explicit runtime module loader
```

---

## Task 4: Define and compile-check the backend plugin C ABI v1

**Purpose:** Freeze a binary boundary that can survive C++ compiler/STL differences and future backend versions.

**Files:**

- Add: `runtime/abi/CMakeLists.txt`
- Add: `runtime/abi/include/kfcore/runtime/abi/backend_v1.h`
- Add: `runtime/abi/tests/CMakeLists.txt`
- Add: `runtime/abi/tests/test_backend_v1_c.c`
- Add: `runtime/abi/tests/test_backend_v1_cpp.cpp`
- Modify: root `CMakeLists.txt`

**RED:** Write C and C++ compile tests that include only `backend_v1.h` and instantiate the public ABI structures.

**ABI v1 must define:**

- `KFCORE_BACKEND_ABI_V1_MAJOR` and `KFCORE_BACKEND_ABI_V1_MINOR`;
- fixed-width C enums/integers for data type and memory kind;
- UTF-8 string views as pointer + byte length, never `std::string`;
- opaque handles for backend instance, executable model, and execution context;
- device/capability descriptors;
- tensor descriptor/view structures;
- explicit `struct_size` on extensible public structs/function tables;
- status code plus backend-owned diagnostic text with documented lifetime;
- create/destroy backend functions;
- enumerate devices/capabilities functions;
- load/destroy model functions;
- create/destroy execution-context functions;
- synchronous `run` function;
- one query entry point with C linkage, named `kfcore_backend_query_v1`.

The function table must be append-only for compatible minor revisions. A breaking layout or semantic change requires a new ABI major/query symbol.

**Prohibited in the header:**

- C++ namespaces/classes/templates;
- STL types;
- exceptions;
- CUDA/TensorRT/ORT headers or handles;
- ownership that requires host and plugin to share allocators.

**Focused commands:**

```bash
cmake --build build/runtime-neutral --target kfcore_runtime_abi_tests
ctest --test-dir build/runtime-neutral -R "runtime_abi" --output-on-failure
```

**GREEN:** Both C and C++ consumers compile and the ABI layout/version tests pass.

**Commit:**

```text
feat: define runtime backend plugin ABI v1
```

---

## Task 5: Build the host-side plugin/model/context wrappers and lifetime anchor

**Purpose:** Convert the C ABI into a safe C++ host API while ensuring child objects cannot outlive their loaded module.

**Files:**

- Add: `runtime/core/include/kfcore/runtime/plugin.hpp`
- Add: `runtime/core/include/kfcore/runtime/model.hpp`
- Add: `runtime/core/src/plugin.cpp`
- Add: `runtime/core/tests/fixtures/fake_backend_plugin.cpp`
- Add: `runtime/core/tests/test_plugin.cpp`
- Modify: `runtime/core/CMakeLists.txt`
- Modify: `runtime/core/tests/CMakeLists.txt`

**RED:** Add tests for:

- explicit `BackendPlugin::load(path)`;
- ABI-major rejection;
- backend ID retrieval;
- device enumeration;
- duplicate backend ID rejection in a host registry;
- model creation and context creation through the fake function table;
- `ExecutionContext::run` transport of neutral tensor views;
- module remains loaded after the `BackendPlugin` wrapper is released if a model/context still exists;
- module unloads after the last dependent object is destroyed;
- plugin status/diagnostic becomes a `kfcore::runtime` exception/error only after crossing back into host C++ code.

**Host API shape:**

```cpp
class BackendPlugin;
class ExecutableModel;
class ExecutionContext;

class BackendRegistry {
public:
    std::shared_ptr<BackendPlugin> load(const std::filesystem::path& explicit_path);
    std::shared_ptr<BackendPlugin> find(std::string_view backend_id) const;
};
```

`ExecutableModel` and `ExecutionContext` must hold a shared module/backend state anchor. `ExecutionContext` is move-only and documented as single-invocation-at-a-time in v1.

**Focused commands:**

```bash
cmake --build build/runtime-neutral --target kfcore_runtime_core_tests
ctest --test-dir build/runtime-neutral -R "runtime_core.*plugin|plugin.*runtime_core" --output-on-failure
```

**GREEN:** Fake backend proves load/query/model/context/run/destroy/unload semantics without any real inference runtime.

**Commit:**

```text
feat: add host runtime plugin wrappers
```

---

## Task 6: Adapt the existing TensorRT runtime as a plugin

**Purpose:** Prove that the ABI wraps the existing production runtime rather than forcing a TensorRT rewrite.

**Files:**

- Add: `backends/tensorrt_cuda/plugin/CMakeLists.txt`
- Add: `backends/tensorrt_cuda/plugin/src/plugin.cpp`
- Add: `backends/tensorrt_cuda/plugin/tests/CMakeLists.txt`
- Add: `backends/tensorrt_cuda/plugin/tests/test_tensorrt_plugin.cpp`
- Modify: root/backend CMake wiring as minimally required

**RED:** Plugin integration test must explicitly load the built module and verify:

- `backend_id == "tensorrt"`;
- CUDA devices are reported as stable device IDs such as `cuda:0`;
- an existing TensorRT engine fixture can be loaded through the ABI;
- creating two execution contexts from one model does not share mutable `Executor` state;
- synchronous inference succeeds through neutral tensor views;
- a requested unsupported memory/device combination fails rather than silently copying or falling back.

**Implementation mapping:**

- backend model handle owns/anchors `std::shared_ptr<const kfcore::tensorrt::Engine>`;
- backend execution-context handle owns one `kfcore::tensorrt::Executor`;
- convert ABI tensor descriptors/views to current `kfcore::tensorrt` types inside the plugin;
- advertise only memory kinds the adapter truly implements. Do not claim `PinnedHost` support merely because CUDA can allocate pinned memory.

Do not move or rename `backends/tensorrt_cuda/runtime` in this task.

**Focused commands:**

```bash
cmake --build build/tensorrt-cuda --target kfcore_backend_tensorrt_plugin_tests
ctest --test-dir build/tensorrt-cuda -R "tensorrt.*plugin|plugin.*tensorrt" --output-on-failure
```

**GREEN:** Existing direct TensorRT runtime tests and new plugin tests both pass.

**Commit:**

```text
feat: expose TensorRT through runtime plugin ABI
```

---

## Task 7: Adapt ONNX Runtime as one backend with explicit provider/device selection

**Purpose:** Remove the architectural assumption that ONNX CPU and ONNX CUDA are different KFCore backends while keeping current model wrappers working during migration.

**Files:**

- Add: `backends/onnx_cpu/plugin/CMakeLists.txt`
- Add: `backends/onnx_cpu/plugin/src/plugin.cpp`
- Add: `backends/onnx_cpu/plugin/tests/CMakeLists.txt`
- Add: `backends/onnx_cpu/plugin/tests/test_onnxruntime_plugin.cpp`
- Modify: `backends/onnx_cpu/runtime/include/kfcore/runtime_onnx/runtime.hpp`
- Modify: `backends/onnx_cpu/runtime/src/runtime.cpp`
- Modify: `backends/onnx_cpu/runtime/CMakeLists.txt`

**RED:** Tests first require:

- `backend_id == "onnxruntime"`;
- CPU provider exposed as device/provider ID `cpu`;
- CUDA provider exposed as `cuda:0` when the linked ORT build supports CUDA EP;
- requesting `cuda:0` does not silently create a CPU session if CUDA EP initialization fails;
- requesting `cpu` remains valid in a build that also has CUDA support;
- model/context/run works through the same ABI used by TensorRT;
- no TensorRT Execution Provider is enabled inside this plugin.

**Implementation:**

- Extend ORT session configuration from compile-selected provider behavior toward an explicit provider/device option used by the plugin.
- Keep the current `Session` implementation and current model wrappers operational while the plugin is introduced.
- Backend plugin translates neutral tensors to ORT values internally.
- v1 may continue host tensor I/O for ORT CUDA if that is the current proven path; it must advertise that limitation accurately instead of pretending zero-copy device I/O exists.

Do not rename `backends/onnx_cpu` yet. Directory cleanup follows the architecture proof to avoid mixing semantic migration with mass path churn.

**Focused commands:**

```bash
cmake --build build/onnx-cpu --target kfcore_backend_onnxruntime_plugin_tests
ctest --test-dir build/onnx-cpu -R "onnxruntime.*plugin|plugin.*onnxruntime" --output-on-failure
```

Run the CUDA-provider form on a CUDA-enabled ORT environment as an additional exact-head gate.

**GREEN:** CPU and available CUDA provider selection are explicit, while all pre-existing ORT runtime/model tests remain green.

**Commit:**

```text
feat: expose ONNX Runtime through runtime plugin ABI
```

---

## Task 8: Add `ModelPackage` parsing and deterministic artifact validation

**Purpose:** Represent one logical model with multiple runtime artifacts without teaching the domain layer about backend file formats.

**Files:**

- Add: `runtime/core/include/kfcore/runtime/model_package.hpp`
- Add: `runtime/core/src/model_package.cpp`
- Add: `runtime/core/tests/test_model_package.cpp`
- Add: `runtime/core/tests/fixtures/model_package/model.json`
- Add: `runtime/core/tests/fixtures/model_package/model.onnx` only if a tiny existing repository fixture cannot be reused
- Add: `runtime/core/tests/fixtures/model_package/model.plan` only as a small test artifact/metadata fixture if required; do not add a large generated engine to source control
- Modify: `runtime/core/CMakeLists.txt`

**Package contract:** `model.json` contains at minimum:

```json
{
  "schema_version": 1,
  "model": {
    "id": "yolo-person-test",
    "version": "1",
    "semantic_type": "yolo-detector",
    "semantic_flavor": "yolov8"
  },
  "artifacts": [
    {
      "id": "onnx-source",
      "backend": "onnxruntime",
      "path": "model.onnx",
      "sha256": "<64 lowercase hex characters>"
    },
    {
      "id": "trt-engine",
      "backend": "tensorrt",
      "path": "model.plan",
      "sha256": "<64 lowercase hex characters>",
      "device": "cuda:0",
      "provenance": {
        "source_artifact": "onnx-source",
        "compiler": "tensorrt",
        "compiler_version": "8.6",
        "profile": "default"
      }
    }
  ]
}
```

The test fixture must use real hashes for its committed fixture bytes; the example above documents schema shape only.

**RED:** Tests cover, in this exact validation order:

1. manifest/artifact existence;
2. SHA-256 integrity;
3. semantic metadata/schema validity;
4. backend/compiler/runtime compatibility metadata;
5. device compatibility metadata;
6. only then artifact load eligibility.

Also test duplicate artifact IDs, path escape outside package root, unknown schema major, malformed hash, missing provenance source, and stale TensorRT provenance.

**Implementation dependency:** Use the existing SaltsUtils direct JSON/data-bind parser already in KFCore's dependency set. Do not add `nlohmann-json` or a second JSON stack. Keep parser-specific data transfer structs private to `model_package.cpp`; the public API exposes KFCore runtime types, not SaltsUtils parser internals.

**Focused commands:**

```bash
cmake --build build/runtime-neutral --target kfcore_runtime_core_tests
ctest --test-dir build/runtime-neutral -R "model_package" --output-on-failure
```

**GREEN:** A package can be parsed and validated without loading any inference backend.

**Commit:**

```text
feat: add runtime model package contract
```

---

## Task 9: Add explicit `ExecutionPolicy` and artifact resolver

**Purpose:** Make backend/provider preference and fallback a deterministic host decision rather than hidden backend behavior.

**Files:**

- Add: `runtime/core/include/kfcore/runtime/resolver.hpp`
- Add: `runtime/core/src/resolver.cpp`
- Add: `runtime/core/tests/test_resolver.cpp`
- Modify: `runtime/core/CMakeLists.txt`

**Public concepts:**

```cpp
struct ExecutionCandidate {
    std::string backend_id;
    std::string device_id;
};

struct ExecutionPolicy {
    std::vector<ExecutionCandidate> ordered_candidates;
    bool allow_fallback;
};

struct ResolvedArtifact {
    std::shared_ptr<BackendPlugin> backend;
    ModelArtifact artifact;
    std::string device_id;
};
```

**RED:** Tests cover:

- strict TensorRT-only policy resolves only a compatible TensorRT artifact;
- strict policy fails if that artifact is stale/incompatible even when ONNX exists;
- explicit `[tensorrt/cuda:0, onnxruntime/cuda:0, onnxruntime/cpu]` policy follows that exact order;
- `allow_fallback=false` prevents candidate #2 after candidate #1 fails;
- missing backend, missing device, incompatible artifact, and hash/provenance errors retain actionable diagnostics;
- resolver result records the selected backend/device/artifact and does not encode fallback inside the backend plugin.

**Focused commands:**

```bash
cmake --build build/runtime-neutral --target kfcore_runtime_core_tests
ctest --test-dir build/runtime-neutral -R "resolver|execution_policy" --output-on-failure
```

**GREEN:** Resolver selects from fake registered backends/packages without real inference libraries.

**Commit:**

```text
feat: add explicit runtime artifact resolver
```

---

## Task 10: Split build capabilities and permit ONNX Runtime + TensorRT coexistence

**Purpose:** Stop using global build personality as a substitute for runtime backend/provider selection.

**Files:**

- Modify: `CMakeOptions.cmake`
- Modify: root `CMakeLists.txt`
- Modify: `backends/tensorrt_cuda/runtime/CMakeLists.txt` as needed
- Modify: `backends/onnx_cpu/runtime/CMakeLists.txt` as needed
- Modify: new plugin CMake files from Tasks 6-7
- Modify: preset files only if repository-wide CI/configuration needs named coexistence presets

**RED:** Add a configuration/build gate that requires all three capabilities simultaneously:

```bash
cmake -S . -B build/backend-coexistence -G Ninja \
  -DBUILD_TESTS=ON \
  -DKFCORE_ENABLE_CUDA=ON \
  -DKFCORE_ENABLE_TENSORRT=ON \
  -DKFCORE_ENABLE_ONNXRUNTIME=ON
```

This must initially fail under the old mutually exclusive option model.

**GREEN design:**

- `KFCORE_ENABLE_CUDA`: CUDA primitives/image operations only.
- `KFCORE_ENABLE_TENSORRT`: TensorRT runtime/plugin; implies CUDA capability but is no longer synonymous with the generic CUDA option.
- `KFCORE_ENABLE_ONNXRUNTIME`: ONNX Runtime runtime/plugin.
- Remove the build-level prohibition against ONNX Runtime CUDA and TensorRT coexisting.
- Provider choice is a runtime/session concern when the linked ORT distribution exposes that provider.
- `find_package(TensorRT)` only when TensorRT support is requested.
- `find_package(ONNXRuntime)` only when ONNX Runtime support is requested.
- `enable_language(CXX)` for runtime/vision C++ modules independently of CUDA.
- `enable_language(CUDA)` only when CUDA compilation is actually required.
- New runtime targets link dependencies target-locally; do not propagate TensorRT or ORT headers/libs through `KFCore::runtime_core`.

Do not preserve the old mutually exclusive `KFCORE_ENABLE_ONNX_CPU` / `KFCORE_ENABLE_ONNX_CUDA` architecture as aliases indefinitely. If a short transition is required for an existing CI preset, confine compatibility mapping to CMake and remove it in the cleanup task once all repository callers use the new capability switches.

**Focused commands:**

```bash
cmake -S . -B build/backend-coexistence -G Ninja \
  -DBUILD_TESTS=ON \
  -DKFCORE_ENABLE_CUDA=ON \
  -DKFCORE_ENABLE_TENSORRT=ON \
  -DKFCORE_ENABLE_ONNXRUNTIME=ON
cmake --build build/backend-coexistence --target \
  kfcore_runtime_core \
  kfcore_backend_tensorrt_plugin \
  kfcore_backend_onnxruntime_plugin
ctest --test-dir build/backend-coexistence -R "runtime|plugin" --output-on-failure
```

**Commit:**

```text
refactor: decouple runtime backend build capabilities
```

---

## Task 11: Introduce one typed YOLO detector over `ExecutableModel`

**Purpose:** Make YOLO the first proof that domain code can be backend-independent while keeping pre/postprocessing strongly typed.

**Files:**

- Add: `vision/core/yolo/include/kfcore/yolo/detector.hpp`
- Add: `vision/core/yolo/src/detector.cpp`
- Add: `vision/core/yolo/tests/test_detector.cpp`
- Modify: `vision/core/yolo/CMakeLists.txt`
- Modify/reuse: `vision/core/yolo/src/raw_yolo.cpp`
- Modify/reuse: `vision/core/yolo/src/compact_nms.cpp`
- Modify: ORT/TensorRT YOLO backend tests as migration adapters require

**RED:** Define the typed API first and test it with a fake `ExecutableModel`/context:

```cpp
namespace kfcore::yolo {

struct DetectorOptions {
    int input_width = 640;
    int input_height = 640;
    float confidence_threshold = 0.25F;
    float nms_threshold = 0.45F;
    int target_class = -1;
};

class Detector {
public:
    static std::unique_ptr<Detector> create(
        std::shared_ptr<runtime::ExecutableModel> model,
        DetectorOptions options = {});

    std::vector<Detection> detect(const image::ImageView& image);
    DetectionFrame detect_frame(const image::ImageView& image);
};

}
```

The exact input/output tensor contract is selected by `ModelPackage` semantic metadata (`semantic_type` / `semantic_flavor`), not by `backend_id`.

**Implementation boundaries:**

- image preprocess remains in KFCore vision/image processing code;
- YOLO tensor layout/decode/NMS remains in `vision/core/yolo`;
- plugin receives tensors and executes only inference;
- no `if (backend == "tensorrt") decode_a(); else ...` logic in the typed detector;
- if ONNX and TensorRT exports genuinely have different tensor contracts, represent that difference as semantic artifact flavor/contract metadata, not runtime brand.

**Focused commands:**

```bash
cmake --build build/backend-coexistence --target kfcore_yolo_core_tests
ctest --test-dir build/backend-coexistence -R "yolo.*detector|detector.*yolo" --output-on-failure
```

**GREEN:** Fake-runtime test proves typed YOLO pre/postprocessing is runtime-independent.

**Commit:**

```text
feat: add backend-neutral typed yolo detector
```

---

## Task 12: Route real ORT and TensorRT YOLO execution through the typed detector

**Purpose:** Remove runtime identity from the primary YOLO application path while preserving current semantics.

**Files:**

- Modify: `backends/onnx_cpu/yolo` implementation/tests as needed to become packaging/compatibility adapters rather than the primary domain implementation
- Modify: `backends/tensorrt_cuda/yolo` implementation/tests similarly
- Add: `vision/core/yolo/tests/test_detector_runtime_parity.cpp` if runtime fixtures can be referenced from one test target
- Modify: `vision/core/yolo/CMakeLists.txt`

**RED:** For the same image/model semantics, instantiate `kfcore::yolo::Detector` twice:

1. resolved ONNX Runtime model;
2. resolved TensorRT model.

Require both results to pass the Task 1 semantic contract and compare:

- detection count after agreed thresholding;
- class IDs;
- confidence within an explicit tolerance justified by FP32/FP16 engine precision;
- boxes within an explicit pixel/normalized-coordinate tolerance;
- source-frame dimensions exactly.

The comparison is semantic, not bitwise.

**Migration rule:** Existing `OnnxDetector` and TensorRT-specific `Detector` APIs may delegate to the new typed path during this task if needed to keep repository callers green. Do not maintain duplicate preprocessing/decode implementations once delegation is proven.

**Focused commands:**

```bash
cmake --build build/backend-coexistence --target kfcore_yolo_tests
ctest --test-dir build/backend-coexistence -R "yolo" --output-on-failure
```

**GREEN:** Both real runtimes use the same typed detector semantics; runtime-specific wrappers no longer own independent YOLO preprocessing/decode logic.

**Commit:**

```text
refactor: route yolo through neutral runtime model
```

---

## Task 13: Add the same-process multi-backend architecture acceptance gate

**Purpose:** Prove the architectural requirement that ORT and TensorRT coexist, resolve, run, and unload correctly in one process.

**Files:**

- Add: `runtime/integration/CMakeLists.txt`
- Add: `runtime/integration/test_multi_backend_yolo.cpp`
- Add: `runtime/integration/fixtures/yolo/model.json`
- Modify: root `CMakeLists.txt`
- Modify CI workflow/preset only if required to provide both real runtime dependencies in one job

**Acceptance test sequence:**

1. Explicitly load the TensorRT plugin module and ONNX Runtime plugin module into one process.
2. Verify unique backend IDs `tensorrt` and `onnxruntime`.
3. Enumerate available devices/providers.
4. Parse one logical YOLO `ModelPackage` with ONNX and TensorRT artifacts.
5. Resolve and run with strict `onnxruntime/cpu` policy when CPU EP is present.
6. Resolve and run with strict `onnxruntime/cuda:0` policy when CUDA EP is present.
7. Resolve and run with strict `tensorrt/cuda:0` policy.
8. Compare each available result with the shared YOLO semantic contract and backend parity tolerances.
9. Verify explicit fallback policy ordering separately from strict policy.
10. Verify stale TensorRT provenance, hash mismatch, wrong device, missing plugin, and incompatible ABI each fail deterministically before inference.
11. Destroy registry/plugin wrappers before model/context wrappers in one test and confirm module lifetime anchoring prevents use-after-unload.
12. Destroy in normal dependency order in another test and confirm clean unload.

No backend/provider case may silently fall back to CPU.

**Focused commands:**

```bash
cmake -S . -B build/backend-coexistence -G Ninja \
  -DBUILD_TESTS=ON \
  -DKFCORE_ENABLE_CUDA=ON \
  -DKFCORE_ENABLE_TENSORRT=ON \
  -DKFCORE_ENABLE_ONNXRUNTIME=ON
cmake --build build/backend-coexistence --target kfcore_runtime_integration_tests
ctest --test-dir build/backend-coexistence -R "multi_backend_yolo" --output-on-failure
```

**Broader gate:**

```bash
ctest --test-dir build/backend-coexistence --output-on-failure
```

If the full repository suite contains unrelated hardware/model-data tests that are intentionally opt-in, run the established CI profile used by KFCore rather than weakening the new architecture acceptance test.

**GREEN definition:** This task is not green until the exact-head same-process acceptance test passes with every backend/provider actually available on the runner and unsupported providers are reported as skipped/unavailable explicitly rather than substituted.

**Commit:**

```text
test: prove same-process runtime backend coexistence
```

---

## Task 14: Remove proof-phase duplication and publish the new architecture boundary

**Purpose:** Finish the migration proof without expanding into RTMW/Face/Hand work.

**Files:**

- Modify: root `CMakeLists.txt` project description
- Modify: `README.md`
- Modify: `backends/onnx_cpu/README.md`
- Modify: relevant runtime/YOLO CMake files
- Remove/deprecate: obsolete YOLO runtime-specific implementation paths only after repository search proves no remaining internal caller depends on them
- Modify: preset/CI option names to the new capability model

**RED review gate:** Search the repository for:

- old build switches `KFCORE_ENABLE_ONNX_CPU` and `KFCORE_ENABLE_ONNX_CUDA`;
- duplicate YOLO preprocess/decode implementations;
- public YOLO APIs whose primary model type still encodes `Onnx` or `TensorRt`;
- `backends/onnx_cpu` documentation claiming CPU-only semantics;
- root project description claiming KFCore is only a Kalman-filter library.

Do not delete a compatibility wrapper until the search proves it has no required repository caller or the wrapper cleanly delegates to the new typed implementation.

**Documentation must state:**

- runtime core is backend-neutral;
- backend plugins are explicit-path C ABI modules;
- backend and device/provider are separate;
- TensorRT and ONNX Runtime may coexist;
- fallback is explicit policy only;
- YOLO is the first migrated typed model;
- RTMW/Face/Hand remain on their current adapters until separate migration plans land.

**Verification:**

```bash
cmake --build build/backend-coexistence
ctest --test-dir build/backend-coexistence -R "runtime|plugin|yolo" --output-on-failure
ctest --test-dir build/backend-coexistence --output-on-failure
```

Also run the repository's existing configured CPU-only/ORT and TensorRT-focused CI profiles so the architecture proof does not regress supported standalone configurations.

**Commit:**

```text
docs: publish backend-neutral runtime architecture
```

---

## Required review checkpoints

Do not merge the entire refactor as one opaque change. Review at these architecture checkpoints:

1. **Runtime contract checkpoint — Tasks 1-5:** neutral types, dynamic loading, C ABI, host wrappers, fake backend lifetime tests. No real backend migration yet.
2. **Backend adapter checkpoint — Tasks 6-10:** TensorRT and ONNX Runtime plugins plus explicit provider/device semantics and coexistence build.
3. **YOLO proof checkpoint — Tasks 11-13:** one typed YOLO implementation through both runtimes and same-process acceptance.
4. **Cleanup checkpoint — Task 14:** remove only duplication made obsolete by the accepted proof and update public documentation/options.

At each checkpoint, stop and review the diff before starting the next phase. Do not carry a failing exact-head gate forward as technical debt.

## Explicitly deferred follow-on work

The following work starts only after Task 13 is green and Task 14 cleanup is reviewed:

- RTMW 133-point typed model and its affine/SimCC coordinate-transform contract;
- zero-copy/device-resident ORT CUDA I/O beyond what the ORT plugin can truthfully advertise in v1;
- Face/Hand typed-model migration;
- physical rename of `backends/onnx_cpu` to an ONNX Runtime-centric directory if still warranted after provider coexistence;
- asynchronous execution, streams/events, schedulers, graph DSLs, plugin auto-discovery, or hot reload;
- DirectML/other providers.

These are deliberately not prerequisites for proving the backend-neutral architecture.

## Completion criteria

The implementation plan is complete only when all of the following are true on one exact head:

- `KFCore::runtime_core` builds without TensorRT, ONNX Runtime, or CUDA headers/libraries.
- The C ABI header compiles as C and C++.
- Plugin module lifetime is proven safe through fake-backend tests.
- TensorRT and ONNX Runtime expose the same backend ABI while retaining backend-specific internals.
- TensorRT and ONNX Runtime can both be linked/loaded by the same KFCore process.
- Provider/device selection is explicit and no silent CPU fallback exists.
- `ModelPackage` integrity/provenance checks happen before backend engine/session load.
- `ExecutionPolicy` owns fallback ordering.
- `kfcore::yolo::Detector` contains typed YOLO semantics and does not branch on runtime brand.
- One same-process acceptance test executes the same YOLO package through every available requested backend/provider and compares semantic results.
- Existing focused runtime and YOLO tests remain green.
- RTMW/Face/Hand have not been dragged into this proof-phase refactor.
