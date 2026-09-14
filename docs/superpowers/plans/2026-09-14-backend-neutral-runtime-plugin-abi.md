# Backend-Neutral Runtime Plugin ABI + YOLO Proof Implementation Plan

**Date:** 2026-09-14  
**Design:** `docs/superpowers/specs/2026-09-14-backend-neutral-runtime-plugin-abi-design.md`

## Goal

Prove the backend-neutral runtime architecture end to end without rewriting KFCore: introduce a shared runtime contract and stable C plugin ABI, adapt the existing TensorRT and ONNX Runtime implementations behind that boundary, resolve artifacts through an explicit model package and execution policy, and migrate YOLO as the first typed domain model.

The proof is complete only when ONNX Runtime and TensorRT can coexist in one process and the same YOLO semantic API executes through either runtime. RTMW, Face, and Hand are explicitly deferred to follow-on plans after this proof is green.

## Non-negotiable boundaries

- Backend plugins execute tensor inference; they do not own YOLO/RTMW/Face/Hand semantics.
- Preserve the existing TensorRT `Engine -> Executor` implementation and adapt it; do not rewrite it for the proof.
- Loaded models are immutable/shareable. Execution contexts own mutable invocation state and are single-concurrent-call objects in v1.
- Runtime identity and device/provider identity are separate: `tensorrt` / `onnxruntime` versus `cuda:0` / `cpu`.
- No silent fallback. Fallback order exists only in an explicit `ExecutionPolicy`.
- No plugin-directory scanning, hot reload, graph DSL, async scheduler, or provider auto-selection in v1.
- No new JSON dependency. `model.json` uses the existing SaltsUtils direct JSON/data-bind facilities already present in KFCore's dependency graph.
- C++ exceptions, STL objects, allocators, TensorRT/ORT/CUDA objects, and compiler-specific C++ ABI types never cross the plugin ABI.
- Preserve current runtime-specific APIs only long enough to prove migration. Do not create a permanent second implementation path.

## Build/test convention

For Tasks 2-5 and 8-9, use a backend-free build:

```bash
cmake -S . -B build/runtime-neutral -G Ninja \
  -DBUILD_TESTS=ON \
  -DKFCORE_ENABLE_CUDA=OFF \
  -DKFCORE_ENABLE_ONNX_CPU=OFF \
  -DKFCORE_ENABLE_ONNX_CUDA=OFF
```

This uses the repository's existing Salts/SaltsUtils discovery environment.

After Task 10, use the coexistence build:

```bash
cmake -S . -B build/backend-coexistence -G Ninja \
  -DBUILD_TESTS=ON \
  -DKFCORE_ENABLE_CUDA=ON \
  -DKFCORE_ENABLE_TENSORRT=ON \
  -DKFCORE_ENABLE_ONNXRUNTIME=ON
```

Every production change follows RED -> GREEN: add/focus the test first, confirm it fails for the intended reason, implement the smallest change, rerun the focused test, then rerun the nearest existing runtime/YOLO tests before committing.

---

## Task 1 — Freeze current YOLO semantics

**Purpose:** Prevent the architecture migration from changing the domain contract while runtime plumbing moves underneath it.

**Files**

- Add: `vision/core/yolo/tests/yolo_semantic_contract.hpp`
- Modify: `backends/onnx_cpu/yolo/tests/test_yolo_onnx_integration.cpp`
- Modify: `backends/tensorrt_cuda/yolo/tests/test_tensorrt_integration.cpp`
- Modify test CMake files only if required to expose the shared helper

**Current contract to preserve**

The public types in `vision/core/yolo/include/kfcore/yolo/types.hpp` are already backend-neutral:

- input: `kfcore::yolo::ImageView`;
- box: `BoxF { left, top, right, bottom }`;
- result: `Detection { box, score, class_id }`;
- frame: `DetectionFrame { image_width, image_height, detections }`.

Do not add runtime names, backend names, labels, or provider metadata to these domain result types.

**RED**

Create a shared assertion helper for `DetectionFrame` and call it from both existing integration tests. Assert:

- source `image_width` / `image_height` are preserved;
- every `BoxF` coordinate is finite;
- `right >= left` and `bottom >= top`;
- scores are finite and within the supported score interval;
- `class_id` remains model/domain output;
- boxes remain in the existing source-image coordinate convention.

Focused existing tests are exactly:

```text
test_yolo_onnx_integration
test_yolo_tensorrt_integration
```

Run those two tests in the repository's already configured ORT and TensorRT build profiles and confirm the helper is exercised.

**GREEN**

Make only test-side normalization needed for both current implementations to satisfy the same semantic contract. No new runtime abstraction in this task.

**Commit**

```text
test: freeze backend-independent yolo semantics
```

---

## Task 2 — Add backend-neutral runtime value types

**Purpose:** Establish the vocabulary currently duplicated between `kfcore::tensorrt` and `kfcore::runtime_onnx`.

**Files**

- Add: `runtime/core/CMakeLists.txt`
- Add: `runtime/core/include/kfcore/runtime/types.hpp`
- Add: `runtime/core/include/kfcore/runtime/error.hpp`
- Add: `runtime/core/src/error.cpp`
- Add: `runtime/core/tests/CMakeLists.txt`
- Add: `runtime/core/tests/test_types.cpp`
- Modify: root `CMakeLists.txt`

**RED**

Write tests that include only the new neutral header and require:

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

} // namespace kfcore::runtime
```

Also require that the header compiles without TensorRT, ONNX Runtime, or CUDA headers.

**GREEN**

- Create target `kfcore_runtime_core` and alias `KFCore::runtime_core`.
- Enable CXX independently of CUDA.
- Keep this target free of runtime-specific link dependencies.
- Do not migrate TensorRT/ORT public types yet.

**Verify**

```bash
cmake --build build/runtime-neutral --target kfcore_runtime_core_tests
ctest --test-dir build/runtime-neutral -R "runtime_core" --output-on-failure
```

**Commit**

```text
feat: add backend-neutral runtime core types
```

---

## Task 3 — Add explicit-path dynamic module loading

**Purpose:** Load backend modules safely without creating a discovery framework.

**Files**

- Add: `runtime/core/src/dynamic_library.hpp`
- Add: `runtime/core/src/dynamic_library.cpp`
- Add: `runtime/core/tests/fixtures/test_backend_module.cpp`
- Add: `runtime/core/tests/test_dynamic_library.cpp`
- Modify: `runtime/core/CMakeLists.txt`
- Modify: `runtime/core/tests/CMakeLists.txt`

**RED**

Tests must cover:

- load one explicitly named module path;
- resolve one exported fixture symbol;
- deterministic missing-module error;
- deterministic missing-symbol error;
- unload through RAII;
- no directory scan/search API.

**GREEN**

- Windows: `LoadLibraryW`, `GetProcAddress`, `FreeLibrary`.
- POSIX: `dlopen`, `dlsym`, `dlclose`.
- Convert native errors to `kfcore::runtime` host-side errors.
- Keep native handles private to `runtime/core/src`.

**Verify**

```bash
cmake --build build/runtime-neutral --target kfcore_runtime_core_tests
ctest --test-dir build/runtime-neutral -R "dynamic_library" --output-on-failure
```

**Commit**

```text
feat: add explicit runtime module loader
```

---

## Task 4 — Define the stable C backend ABI v1

**Purpose:** Establish a compiler/STL-independent binary boundary.

**Files**

- Add: `runtime/abi/CMakeLists.txt`
- Add: `runtime/abi/include/kfcore/runtime/abi/backend_v1.h`
- Add: `runtime/abi/tests/CMakeLists.txt`
- Add: `runtime/abi/tests/test_backend_v1_c.c`
- Add: `runtime/abi/tests/test_backend_v1_cpp.cpp`
- Modify: root `CMakeLists.txt`

**RED**

Create C and C++ compile tests that include only `backend_v1.h` and instantiate the public ABI structs.

**ABI v1 requirements**

- `KFCORE_BACKEND_ABI_V1_MAJOR` / `KFCORE_BACKEND_ABI_V1_MINOR`.
- Fixed-width integer representations for data type, memory kind, status, and capability flags.
- UTF-8 string views are pointer + byte length.
- Opaque backend/model/context handles.
- Device/capability descriptors.
- Tensor descriptor and tensor view structs.
- `struct_size` on extensible structs and the function table.
- Status code plus diagnostic UTF-8 text with documented lifetime.
- Backend create/destroy.
- Device/capability enumeration.
- Model load/destroy.
- Execution-context create/destroy.
- Synchronous `run`.
- One exported query entry point: `kfcore_backend_query_v1`.

The table is append-only for compatible minor versions. Breaking semantics/layout require a new major query symbol.

**Forbidden across ABI**

C++ classes, STL types, exceptions, CUDA handles, TensorRT handles, ORT handles, or memory requiring shared host/plugin allocators.

**Verify**

```bash
cmake --build build/runtime-neutral --target kfcore_runtime_abi_tests
ctest --test-dir build/runtime-neutral -R "runtime_abi" --output-on-failure
```

**Commit**

```text
feat: define runtime backend plugin ABI v1
```

---

## Task 5 — Add host wrappers, registry, and module lifetime anchoring

**Purpose:** Convert the C ABI into the C++ runtime API and prove object lifetimes are safe.

**Files**

- Add: `runtime/core/include/kfcore/runtime/plugin.hpp`
- Add: `runtime/core/include/kfcore/runtime/model.hpp`
- Add: `runtime/core/src/plugin.cpp`
- Add: `runtime/core/tests/fixtures/fake_backend_plugin.cpp`
- Add: `runtime/core/tests/test_plugin.cpp`
- Modify: runtime core CMake files

**RED**

Test:

- `BackendPlugin::load(explicit_path)`;
- incompatible ABI-major rejection;
- backend ID and device enumeration;
- duplicate backend ID rejection in `BackendRegistry`;
- fake model/context creation;
- neutral tensor execution through the fake ABI;
- child model/context survives destruction of the outer plugin wrapper because it retains module state;
- module unload occurs only after the final dependent object dies;
- plugin status text is converted to host C++ error state only after crossing the ABI.

**Public host shape**

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

`ExecutionContext` is move-only and single-concurrent-call in v1.

**Verify**

```bash
cmake --build build/runtime-neutral --target kfcore_runtime_core_tests
ctest --test-dir build/runtime-neutral -R "runtime.*plugin|plugin.*runtime" --output-on-failure
```

**Commit**

```text
feat: add host runtime plugin wrappers
```

### Review checkpoint A

Review Tasks 1-5 as one architecture checkpoint. At this point no real runtime has migrated. The neutral contract, ABI, explicit loader, and fake-backend lifetime model must be reviewable independently.

---

## Task 6 — Adapt TensorRT behind the plugin ABI

**Purpose:** Prove the new boundary wraps the existing production runtime instead of replacing it.

**Files**

- Add: `backends/tensorrt_cuda/plugin/CMakeLists.txt`
- Add: `backends/tensorrt_cuda/plugin/src/plugin.cpp`
- Add: `backends/tensorrt_cuda/plugin/tests/CMakeLists.txt`
- Add: `backends/tensorrt_cuda/plugin/tests/test_tensorrt_plugin.cpp`
- Modify: parent TensorRT/root CMake wiring minimally

**RED**

The plugin test explicitly loads the built module and requires:

- backend ID `tensorrt`;
- stable CUDA device IDs such as `cuda:0`;
- load an existing test engine through the ABI;
- two contexts from one model do not share mutable `Executor` state;
- synchronous inference works through neutral tensor views;
- unsupported memory/device combinations fail explicitly.

**GREEN mapping**

- Plugin model handle anchors `std::shared_ptr<const kfcore::tensorrt::Engine>`.
- Plugin context owns one `kfcore::tensorrt::Executor`.
- ABI tensor views are translated to current TensorRT runtime views inside the plugin.
- Advertise only memory kinds actually implemented.
- Do not move/rename `backends/tensorrt_cuda/runtime`.

**Verify**

Run the new TensorRT plugin test plus the existing TensorRT runtime tests. Both paths must remain green during migration.

**Commit**

```text
feat: expose TensorRT through runtime plugin ABI
```

---

## Task 7 — Adapt ONNX Runtime as one backend with explicit provider selection

**Purpose:** Stop treating ORT CPU and ORT CUDA as different KFCore backend identities.

**Files**

- Add: `backends/onnx_cpu/plugin/CMakeLists.txt`
- Add: `backends/onnx_cpu/plugin/src/plugin.cpp`
- Add: `backends/onnx_cpu/plugin/tests/CMakeLists.txt`
- Add: `backends/onnx_cpu/plugin/tests/test_onnxruntime_plugin.cpp`
- Modify: `backends/onnx_cpu/runtime/include/kfcore/runtime_onnx/runtime.hpp`
- Modify: `backends/onnx_cpu/runtime/src/runtime.cpp`
- Modify: `backends/onnx_cpu/runtime/CMakeLists.txt`

**RED**

Require:

- backend ID `onnxruntime`;
- provider/device ID `cpu`;
- `cuda:0` is advertised only if the linked ORT actually supports CUDA EP;
- requesting `cuda:0` never silently creates a CPU session;
- requesting `cpu` remains valid in a CUDA-capable ORT build;
- model/context/run works through the same ABI as TensorRT;
- TensorRT EP is not enabled inside this plugin.

**GREEN**

- Extend ORT session creation from compile-selected behavior toward explicit provider/device options used by the plugin.
- Keep current `Session` and existing model wrappers operational during migration.
- Host tensor I/O is acceptable for ORT CUDA v1 if that is the proven path; advertise it truthfully and do not claim device zero-copy.
- Do not rename `backends/onnx_cpu` in this task.

**Verify**

Run existing ORT runtime/model tests plus the new plugin test. Run the CUDA-provider test in the existing CUDA-capable ORT environment as an exact-head gate.

**Commit**

```text
feat: expose ONNX Runtime through runtime plugin ABI
```

---

## Task 8 — Add `ModelPackage` and artifact provenance validation

**Purpose:** Represent one logical model with multiple backend artifacts without leaking file-format/runtime details into typed model APIs.

**Files**

- Add: `runtime/core/include/kfcore/runtime/model_package.hpp`
- Add: `runtime/core/src/model_package.cpp`
- Add: `runtime/core/tests/test_model_package.cpp`
- Add: `runtime/core/tests/fixtures/model_package/model.json`
- Reuse tiny existing ONNX fixtures where practical; do not commit large generated engines solely for parser tests
- Modify: runtime core CMake files

**Manifest fields required by v1**

Model-level:

- `schema_version`;
- logical model `id` and `version`;
- `semantic_type` such as `yolo-detector`;
- `semantic_flavor` such as `yolov8`.

Artifact-level:

- artifact `id`;
- `backend`;
- relative package `path`;
- real 64-character lowercase SHA-256 digest of the committed fixture bytes;
- optional explicit device/provider constraint.

TensorRT artifact provenance additionally records:

- source artifact ID;
- compiler/runtime family and version information required by the compatibility check;
- optimization/profile identity;
- device compatibility metadata required by the spec.

Do not place dummy hash placeholders in the committed fixture. Generate the tiny fixture first, compute its real SHA-256, then write `model.json` with that digest.

**RED**

Test validation in this exact order:

1. manifest/artifact existence;
2. SHA-256 integrity;
3. schema and semantic metadata;
4. backend/compiler/runtime provenance compatibility;
5. device compatibility;
6. only then artifact load eligibility.

Also test duplicate artifact IDs, malformed digest, package-root path escape, unknown schema major, missing provenance source, and stale TensorRT provenance.

**GREEN**

- Use SaltsUtils direct JSON/data-bind parsing already available to KFCore.
- Keep SaltsUtils parser DTOs private to `model_package.cpp`.
- Public `ModelPackage` exposes KFCore runtime data only.
- Parsing/validation does not load any inference backend.

**Verify**

```bash
cmake --build build/runtime-neutral --target kfcore_runtime_core_tests
ctest --test-dir build/runtime-neutral -R "model_package" --output-on-failure
```

**Commit**

```text
feat: add runtime model package contract
```

---

## Task 9 — Add explicit `ExecutionPolicy` and resolver

**Purpose:** Make fallback a deterministic host decision.

**Files**

- Add: `runtime/core/include/kfcore/runtime/resolver.hpp`
- Add: `runtime/core/src/resolver.cpp`
- Add: `runtime/core/tests/test_resolver.cpp`
- Modify: runtime core CMake files

**Required host concepts**

```cpp
struct ExecutionCandidate {
    std::string backend_id;
    std::string device_id;
};

struct ExecutionPolicy {
    std::vector<ExecutionCandidate> ordered_candidates;
    bool allow_fallback = false;
};

struct ResolvedArtifact {
    std::shared_ptr<BackendPlugin> backend;
    ModelArtifact artifact;
    std::string device_id;
};
```

**RED**

Test:

- strict `tensorrt/cuda:0` resolves only a compatible TensorRT artifact;
- strict policy fails on stale/incompatible TRT even if ONNX is present;
- ordered `[tensorrt/cuda:0, onnxruntime/cuda:0, onnxruntime/cpu]` follows exactly that order;
- `allow_fallback=false` never tries candidate 2;
- missing backend/device, incompatible artifact, integrity failure, and provenance failure retain actionable diagnostics;
- fallback logic never lives inside a plugin.

**Verify**

```bash
cmake --build build/runtime-neutral --target kfcore_runtime_core_tests
ctest --test-dir build/runtime-neutral -R "resolver|execution_policy" --output-on-failure
```

**Commit**

```text
feat: add explicit runtime artifact resolver
```

---

## Task 10 — Split build capabilities and enable runtime coexistence

**Purpose:** Stop using global build personality as runtime/provider selection.

**Files**

- Modify: `CMakeOptions.cmake`
- Modify: root `CMakeLists.txt`
- Modify: TensorRT runtime/plugin CMake files as required
- Modify: ONNX Runtime runtime/plugin CMake files as required
- Modify presets/CI only where repository callers need the new option names

**RED**

Under the current option model, this desired configuration is impossible:

```bash
cmake -S . -B build/backend-coexistence -G Ninja \
  -DBUILD_TESTS=ON \
  -DKFCORE_ENABLE_CUDA=ON \
  -DKFCORE_ENABLE_TENSORRT=ON \
  -DKFCORE_ENABLE_ONNXRUNTIME=ON
```

Capture that as the configuration-level RED condition.

**GREEN option model**

- `KFCORE_ENABLE_CUDA`: CUDA primitives/image operations.
- `KFCORE_ENABLE_TENSORRT`: TensorRT runtime/plugin; requires CUDA capability but is no longer synonymous with generic CUDA.
- `KFCORE_ENABLE_ONNXRUNTIME`: ONNX Runtime runtime/plugin.
- Remove the TensorRT-versus-ORT-CUDA mutual exclusion.
- ORT provider selection is runtime/session configuration when the linked ORT distribution supports the provider.
- Discover/link TensorRT only from TensorRT targets.
- Discover/link ONNX Runtime only from ORT targets.
- `KFCore::runtime_core` remains independent of both.

If an existing preset temporarily requires `KFCORE_ENABLE_ONNX_CPU` / `KFCORE_ENABLE_ONNX_CUDA`, confine compatibility mapping to CMake during this task and remove repository use of those names in final cleanup. Do not preserve their mutual-exclusion semantics.

**Verify**

```bash
cmake -S . -B build/backend-coexistence -G Ninja \
  -DBUILD_TESTS=ON \
  -DKFCORE_ENABLE_CUDA=ON \
  -DKFCORE_ENABLE_TENSORRT=ON \
  -DKFCORE_ENABLE_ONNXRUNTIME=ON
cmake --build build/backend-coexistence
ctest --test-dir build/backend-coexistence -R "runtime|plugin" --output-on-failure
```

**Commit**

```text
refactor: decouple runtime backend build capabilities
```

### Review checkpoint B

Review Tasks 6-10 before touching YOLO domain implementation. Required evidence: both real plugins pass their focused tests and one build can contain both runtimes without provider fallback coupling.

---

## Task 11 — Add one typed YOLO detector over `ExecutableModel`

**Purpose:** Make YOLO the first domain proof of backend-neutral execution.

**Files**

- Add: `vision/core/yolo/include/kfcore/yolo/detector.hpp`
- Add: `vision/core/yolo/src/detector.cpp`
- Add: `vision/core/yolo/tests/test_detector.cpp`
- Modify: `vision/core/yolo/CMakeLists.txt`
- Reuse/modify: `vision/core/yolo/src/raw_yolo.cpp`
- Reuse/modify: `vision/core/yolo/src/compact_nms.cpp`

**RED**

Test a typed detector against a fake `ExecutableModel` first. Preserve the existing KFCore YOLO image/result contract:

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

    std::vector<Detection> detect(const ImageView& image);
    DetectionFrame detect_frame(const ImageView& image);
};

} // namespace kfcore::yolo
```

Do not introduce another image-view abstraction in this proof. Use the existing `kfcore::yolo::ImageView`; a future shared-image contract is separate work.

**Semantic dispatch rule**

Input/output decode behavior is selected from model/artifact semantic contract metadata, not from `backend_id`. There must be no `if (backend == "tensorrt")` branch in typed YOLO preprocessing or decode.

**GREEN boundaries**

- Image preprocess remains KFCore vision code.
- YOLO tensor layout/decode/NMS remains `vision/core/yolo`.
- Plugin executes tensor inference only.
- If two artifacts truly expose different tensor contracts, encode that difference as semantic flavor/contract metadata rather than runtime brand.

**Verify**

```bash
cmake --build build/backend-coexistence
ctest --test-dir build/backend-coexistence -R "test_.*yolo.*detector|test_detector" --output-on-failure
```

**Commit**

```text
feat: add backend-neutral typed yolo detector
```

---

## Task 12 — Route real ORT and TensorRT YOLO through the typed detector

**Purpose:** Eliminate duplicate runtime-owned YOLO semantics.

**Files**

- Modify: `backends/onnx_cpu/yolo` implementation/tests as compatibility adapters require
- Modify: `backends/tensorrt_cuda/yolo` implementation/tests as compatibility adapters require
- Add: `vision/core/yolo/tests/test_detector_runtime_parity.cpp` if both runtime fixtures can be referenced from one test target
- Modify: `vision/core/yolo/CMakeLists.txt`

**RED**

For the same input image/model semantics, instantiate `kfcore::yolo::Detector` through:

1. resolved ONNX Runtime model;
2. resolved TensorRT model.

Both results must pass Task 1's semantic contract. Compare:

- detection count under the same thresholds;
- `class_id`;
- score within an explicit tolerance justified by artifact precision;
- `BoxF.left/top/right/bottom` within an explicit pixel/coordinate tolerance;
- frame dimensions exactly.

Comparison is semantic, not bitwise.

**GREEN**

Existing `OnnxDetector` / TensorRT-specific detector entry points may temporarily delegate to the typed path so repository callers remain green, but preprocessing/decode logic must not remain duplicated after delegation is proven.

**Verify**

```bash
cmake --build build/backend-coexistence
ctest --test-dir build/backend-coexistence -R "yolo" --output-on-failure
```

**Commit**

```text
refactor: route yolo through neutral runtime model
```

---

## Task 13 — Add the same-process architecture acceptance gate

**Purpose:** Prove coexistence, resolution, execution, failure semantics, and unloading in one process.

**Files**

- Add: `runtime/integration/CMakeLists.txt`
- Add: `runtime/integration/test_multi_backend_yolo.cpp`
- Add: `runtime/integration/fixtures/yolo/model.json`
- Modify: root `CMakeLists.txt`
- Modify CI/preset only if the existing CI needs one job with both runtime dependencies

**Acceptance sequence**

1. Explicitly load ONNX Runtime and TensorRT plugin modules in the same process.
2. Assert unique backend IDs `onnxruntime` and `tensorrt`.
3. Enumerate devices/providers.
4. Parse one logical YOLO `ModelPackage` containing ONNX and TensorRT artifacts with real integrity/provenance metadata.
5. Run strict `onnxruntime/cpu` when CPU EP is available.
6. Run strict `onnxruntime/cuda:0` when CUDA EP is available.
7. Run strict `tensorrt/cuda:0` when TensorRT/CUDA is available.
8. Validate each result with the shared YOLO semantic contract and parity tolerances.
9. Separately prove explicit fallback order.
10. Prove stale TensorRT provenance, SHA mismatch, wrong device, missing plugin, and incompatible ABI fail deterministically before inference.
11. Destroy registry/plugin wrappers before model/context wrappers in one test and prove the lifetime anchor prevents unload/use-after-free.
12. Destroy in normal dependency order in another test and prove clean unload.

An unavailable provider is reported as unavailable/skip according to the test environment. It is never replaced silently by CPU.

**Verify**

```bash
cmake -S . -B build/backend-coexistence -G Ninja \
  -DBUILD_TESTS=ON \
  -DKFCORE_ENABLE_CUDA=ON \
  -DKFCORE_ENABLE_TENSORRT=ON \
  -DKFCORE_ENABLE_ONNXRUNTIME=ON
cmake --build build/backend-coexistence
ctest --test-dir build/backend-coexistence -R "multi_backend_yolo" --output-on-failure
ctest --test-dir build/backend-coexistence --output-on-failure
```

Do not weaken the architecture test because unrelated hardware/model-data tests are opt-in; use the repository's established CI profile for the broader exact-head gate.

**Commit**

```text
test: prove same-process runtime backend coexistence
```

### Review checkpoint C

Do not proceed to cleanup until Task 13 is green on one exact head with every requested provider available on that runner behaving explicitly.

---

## Task 14 — Cleanup proof-phase duplication and publish the boundary

**Purpose:** Finish the architecture proof without expanding into unrelated model migrations.

**Files**

- Modify: root `CMakeLists.txt` project description
- Modify: `README.md`
- Modify: `backends/onnx_cpu/README.md`
- Modify: runtime/YOLO CMake files
- Modify preset/CI option names to the new capability model
- Remove runtime-specific YOLO implementation paths only after repository search proves they are unused or only delegating compatibility wrappers

**Review/search gate**

Search the repository for:

- `KFCORE_ENABLE_ONNX_CPU` and `KFCORE_ENABLE_ONNX_CUDA`;
- duplicate YOLO preprocess/decode implementations;
- primary public YOLO types whose name still encodes `Onnx` or `TensorRt`;
- `backends/onnx_cpu` documentation that incorrectly claims CPU-only runtime behavior;
- root project metadata that still describes KFCore as only a Kalman-filter library.

Do not delete an adapter before callers have migrated or it delegates cleanly to the new path.

**Documentation must state**

- runtime core is backend-neutral;
- plugins use an explicit-path C ABI;
- backend and device/provider are separate;
- TensorRT and ONNX Runtime can coexist;
- fallback is explicit policy only;
- YOLO is the first migrated typed model;
- RTMW/Face/Hand remain on current adapters until their own migration plans.

**Verify**

```bash
cmake --build build/backend-coexistence
ctest --test-dir build/backend-coexistence -R "runtime|plugin|yolo" --output-on-failure
ctest --test-dir build/backend-coexistence --output-on-failure
```

Also run the repository's existing CPU/ORT-focused and TensorRT-focused CI profiles to preserve standalone configurations.

**Commit**

```text
docs: publish backend-neutral runtime architecture
```

---

## Completion criteria

One exact head must demonstrate all of the following:

- `KFCore::runtime_core` has no TensorRT, ONNX Runtime, or CUDA dependency.
- `backend_v1.h` compiles as both C and C++.
- Fake-plugin tests prove module/model/context lifetime safety.
- TensorRT and ONNX Runtime implement the same plugin ABI while retaining runtime-specific internals.
- One KFCore build/process can load both runtime plugins.
- ORT provider and TensorRT device selection are explicit; no silent CPU fallback exists.
- `ModelPackage` verifies integrity and provenance before backend load.
- `ExecutionPolicy` owns fallback ordering.
- `kfcore::yolo::Detector` owns typed YOLO semantics and never branches on runtime brand.
- The same logical YOLO package executes through every explicitly requested available backend/provider in the same-process acceptance test.
- Existing focused runtime and YOLO tests remain green.
- RTMW, Face, and Hand have not been pulled into this proof-phase refactor.

## Deferred follow-on plans

Only after the acceptance gate is green:

- RTMW 133-point typed model plus affine/SimCC coordinate transforms;
- device-resident ORT CUDA I/O beyond the capabilities truthfully exposed in v1;
- Face/Hand typed-model migration;
- physical rename of `backends/onnx_cpu` if still valuable after provider coexistence;
- async execution, streams/events, graph scheduling, plugin auto-discovery, hot reload;
- DirectML or other providers.
