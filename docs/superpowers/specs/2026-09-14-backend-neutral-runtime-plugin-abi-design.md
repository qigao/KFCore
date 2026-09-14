# KFCore Backend-Neutral Runtime and Plugin ABI Design

Date: 2026-09-14
Status: Proposed architecture

## 1. Purpose

KFCore shall treat model semantics and inference execution as separate concerns. A logical model such as YOLO, RTMW, ArcFace, or a face landmark model remains the same model regardless of whether it is executed through ONNX Runtime CPU, ONNX Runtime CUDA, or TensorRT using a prebuilt engine.

The runtime architecture therefore makes execution backends dynamically loadable and selectable per model instance. Multiple backends may coexist in one process. A model may explicitly permit or prohibit particular execution routes; KFCore must not silently substitute CPU for CUDA or otherwise change the requested route.

This design replaces build-wide backend personality with runtime composition while preserving strongly typed model APIs.

## 2. Goals

- Keep public model APIs independent of ONNX Runtime, CUDA EP, TensorRT, and backend-specific classes.
- Allow ONNX Runtime and TensorRT backends to be loaded simultaneously in one process.
- Allow each model instance to choose its own backend/device execution policy.
- Support models that are CPU-only, CUDA-only, or support selected backend/provider combinations.
- Represent one logical model with multiple executable artifacts, including ONNX source artifacts and prebuilt TensorRT engines.
- Load execution layers dynamically through a stable C ABI.
- Keep preprocessing/postprocessing model-specific but backend-independent.
- Make backend/device fallback explicit and user-authorized.
- Preserve a path to host, pinned-host, and device-resident tensors.

## 3. Non-goals

- No generic YAML/JSON preprocessing language.
- No automatic TensorRT engine building in the inference runtime.
- No implicit CPU fallback.
- No runtime scanning of arbitrary system DLL search paths.
- No requirement for asynchronous scheduling in ABI V1.
- No exposure of STL, C++ exceptions, RTTI, or allocator ownership across plugin ABI boundaries.

## 4. Architecture principles

### 4.1 Model is not backend

The public API exposes semantic models:

- `YoloDetector`
- `Rtmw`
- `ArcFace`
- face/hand landmark models

It must not require public runtime-specific model classes such as `TensorRtRtmw`, `OnnxCudaRtmw`, or `CpuRtmw`.

Backend selection occurs when a model package is resolved into an executable model.

### 4.2 Backend is not device

Backend and device/provider are separate concepts.

Examples:

- backend `onnxruntime`, provider/device `cpu`
- backend `onnxruntime`, provider/device `cuda:0`
- backend `tensorrt`, device `cuda:0`

CUDA support is therefore not synonymous with TensorRT support.

### 4.3 Multiple backends may coexist

A process may load, for example:

- `kfcore_backend_onnxruntime.dll`
- `kfcore_backend_tensorrt.dll`

Different models may bind to different backends. The same logical model may also be instantiated more than once through different backends for parity testing or benchmarking.

### 4.4 Explicit execution policy

A model load requires an explicit execution policy, including when the caller chooses automatic selection deliberately.

Examples:

```cpp
ExecutionPolicy::exact("tensorrt", "cuda:0")

ExecutionPolicy::ordered({
    {"tensorrt", "cuda:0"},
    {"onnxruntime", "cuda:0"},
})

ExecutionPolicy::best_available()
```

`best_available()` is an explicit authorization to select automatically. It is not the implicit default.

## 5. Core object model

KFCore distinguishes four concepts that must not be collapsed into one `Model` type:

1. **ModelPackage** — logical model identity, semantic metadata, artifacts, and compatibility metadata.
2. **ModelArtifact** — a concrete ONNX model, TensorRT engine, or another future executable representation.
3. **ExecutableModel** — one artifact successfully bound to one backend/device route.
4. **ExecutionContext** — mutable execution state created from an `ExecutableModel`.

The lifecycle is:

```text
ModelPackage
    |
    | ModelResolver
    v
ExecutionRoute
    |
    v
ExecutableModel        immutable/shareable
    |
    +-- ExecutionContext #1
    +-- ExecutionContext #2
    +-- ExecutionContext #3
```

`ExecutableModel` is expected to be immutable and thread-safe. `ExecutionContext` is mutable and not concurrently reentrant unless a future capability explicitly states otherwise.

## 6. Model Package V1

Model Package V1 is the stable identity layer connecting one logical model to multiple artifacts.

Example layout:

```text
models/rtmw-l-384x288/
    model.json
    rtmw.onnx
    artifacts/
        trt11-sm89-fp16.engine
        trt86-sm61-fp16.engine
```

The package records:

- schema version
- logical model ID and version
- semantic model type/variant
- source artifact identity and hash
- artifact paths and hashes
- artifact format
- artifact flavor where tensor contracts differ but semantic output is equivalent
- backend/device/runtime compatibility requirements
- provenance linking derived artifacts to their source artifact

### 6.1 Source and derived artifacts

For the initial ONNX/TensorRT workflow:

- ONNX is the source artifact.
- TensorRT engine files are derived artifacts.

Each TensorRT engine must record enough provenance to reject stale or incompatible engines, including at least:

- source artifact digest
- TensorRT compatibility information
- target device/compute capability constraints
- precision
- optimization profile information
- artifact digest

The inference runtime does not build a missing TensorRT engine. If no compatible prebuilt artifact exists, model resolution fails with a typed error.

### 6.2 Execution constraints

Execution constraints describe allowed routes, not only a `cpu=true/cuda=false` binary flag.

A CUDA-only model may permit:

```text
onnxruntime + cuda
tensorrt + cuda
```

while omitting ONNX Runtime CPU entirely.

The valid route is the intersection of:

```text
model/package constraints
        AND
artifact constraints
        AND
loaded backend capabilities
        AND
actual device capabilities
        AND
user execution policy
```

### 6.3 Model Package is not a preprocessing DSL

The package may identify semantic type and dimensions, but does not encode arbitrary preprocessing/postprocessing programs. Model-specific C++ code owns semantic preprocessing/postprocessing.

## 7. Model resolver

`ModelResolver` belongs to KFCore Core, not to a backend plugin.

Resolution flow:

```text
read + validate model package
        |
enumerate package artifacts
        |
enumerate loaded backends/devices
        |
probe candidate artifacts
        |
apply model constraints
        |
apply user ExecutionPolicy
        |
select exactly one ExecutionRoute
        |
backend loads selected artifact
```

A backend plugin does not scan model packages and does not decide fallback order.

The backend provides a side-effect-minimized `probe_artifact` capability so the resolver can reject incompatible candidates before expensive model initialization.

## 8. Dynamic backend loading

### 8.1 Platform abstraction

KFCore provides an internal `DynamicLibrary` abstraction.

Windows implementation:

- `LoadLibraryExW`
- `GetProcAddress`
- `FreeLibrary`

POSIX implementation:

- `dlopen`
- `dlsym`
- `dlclose`

The Windows loader uses canonical absolute paths and `LoadLibraryExW`, not unrestricted `LoadLibrary` search behavior.

Recommended flags include:

```text
LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
```

This lets plugin-local dependencies resolve while avoiding arbitrary current-directory/system search behavior controlled by an application accident.

### 8.2 Discovery policy

KFCore loads plugins from:

1. explicitly supplied absolute/canonical plugin paths; or
2. a caller-selected controlled plugin directory such as `<app>/plugins`.

KFCore does not recursively scan PATH, the current working directory, or arbitrary system directories.

A DLL filename does not establish backend identity. Identity comes from the plugin query result.

Duplicate backend IDs are rejected in ABI V1 rather than silently replacing an already loaded implementation.

### 8.3 Unload policy

Runtime backend unloading is not a normal V1 feature. A loaded plugin remains loaded until its owning KFCore `Runtime` is destroyed/process termination occurs.

No module may be unloaded while any backend, executable model, execution context, device handle, or function-table reference originating from that module remains alive.

## 9. Plugin ABI V1

### 9.1 One exported entry point

Each backend plugin exports one stable C symbol conceptually equivalent to:

```c
kf_status kfcore_backend_query(
    uint32_t requested_abi,
    const kf_host_api_v1* host,
    kf_backend_api_v1* out_api);
```

The query returns backend identity, ABI compatibility, capabilities, and a function table.

The DLL does not export a C++ class hierarchy.

### 9.2 Opaque handles

ABI objects use opaque handles, conceptually:

```c
typedef struct kf_backend_t* kf_backend_handle;
typedef struct kf_device_t*  kf_device_handle;
typedef struct kf_model_t*   kf_model_handle;
typedef struct kf_context_t* kf_context_handle;
```

The function table covers at least:

- create/destroy backend
- enumerate devices
- inspect backend/device capabilities
- probe artifact support
- load/destroy executable model
- create/destroy execution context
- inspect tensor contracts
- synchronous execute
- diagnostic/error formatting

### 9.3 C ABI only

The plugin ABI must not expose:

- `std::string`
- `std::vector`
- `std::filesystem::path`
- smart pointers
- C++ exceptions
- RTTI-dependent object types
- STL iterators or containers

ABI data uses fixed-width integers, explicit-size structures, pointer/length views, opaque handles, and function pointers.

### 9.4 Allocation ownership

Memory ownership never crosses the boundary ambiguously.

General rule: the side that allocates owns and frees the allocation.

Prefer caller-provided buffers and two-phase count/fill APIs for enumeration and diagnostics. Do not require Core to free plugin CRT allocations or vice versa.

## 10. Tensor ABI

The backend-neutral tensor ABI supports a common scalar-type set, shape, byte size, memory location, and associated device.

Required V1 memory locations:

- host
- pinned host
- device

Required scalar categories include the scalar types already needed by KFCore runtimes, including FP32, FP16, BF16, INT8, UINT8, INT32, INT64, and BOOL.

A tensor view records:

- name or stable binding identity
- scalar type
- rank and shape
- data pointer
- byte count
- memory kind
- device identity when applicable

The ABI distinguishes memory location from memory interoperability. Two backends both supporting CUDA device memory does not automatically authorize zero-copy exchange.

Capabilities must explicitly describe support for features such as:

- device-memory I/O
- pinned-memory I/O
- compatible CUDA memory interop
- future external stream/event interoperability

If interoperability is not proven, KFCore must use an explicit copy path or reject a zero-copy request.

## 11. ABI versioning

ABI compatibility uses major/minor semantics and append-only structures.

Rules:

- major mismatch is incompatible
- compatible minor extensions append fields or function-table entries
- existing field order and meaning never change inside one major ABI
- externally exchanged structures carry `struct_size`
- unavailable appended functions/capabilities are treated as unsupported, not assumed

This allows a newer KFCore Core to continue loading an older ABI-compatible plugin when it only requires capabilities that plugin implements.

## 12. Error model

No C++ exception crosses the DLL boundary.

Plugin calls return a fixed-width status code with categories such as:

- invalid argument
- unsupported operation
- incompatible artifact
- device unavailable
- dependency unavailable
- out of memory
- runtime failure
- internal failure

Diagnostic text is copied into caller-provided storage or otherwise retrieved without cross-module deallocation.

The C++ KFCore wrapper may convert statuses into KFCore exceptions or `expected`-style results according to the public API policy.

## 13. Preprocessing and postprocessing

Preprocessing/postprocessing belongs to **model semantics**, but the concrete compute implementation does not belong to the inference backend.

Examples:

### YOLO

Model semantics own:

- model input transform requirements
- output decoding
- compact-NMS/EfficientNMS flavor adaptation
- conversion into `DetectionFrame`

### RTMW

Model semantics own:

- person bbox expansion
- aspect-ratio correction
- affine crop transform
- model input normalization
- SimCC decoding
- inverse affine coordinate mapping
- 133-point semantic result construction

Inference plugins own only tensor-to-network-to-tensor execution.

CPU/CUDA image processing implementations may accelerate model preprocessing independently of whether inference runs through ONNX Runtime or TensorRT.

## 14. Semantic contract and artifact flavor

Artifacts belonging to the same logical model are not required to expose byte-for-byte identical tensor contracts if they intentionally represent different executable flavors.

For example a YOLO logical detector may have:

- ONNX compact-NMS artifact
- TensorRT EfficientNMS artifact

Both must map to the same semantic contract:

```text
Image -> DetectionFrame
```

The model package therefore distinguishes logical/semantic model identity from artifact tensor flavor. Typed model code selects the appropriate decoder by artifact flavor, not by checking whether the backend is ONNX Runtime or TensorRT.

## 15. Pipeline boundary

A typed model owns one model's semantic input-to-output transformation.

A pipeline composes models.

Example whole-body pipeline:

```text
Image
  |
  v
YOLO
  |
person boxes
  |
  v
RTMW
  |
133-point poses
```

YOLO decoding remains YOLO logic. RTMW affine/SimCC logic remains RTMW logic. Choosing which detected persons to batch into RTMW and coordinating multiple models belongs to the pipeline.

## 16. Synchronous ABI V1

ABI V1 guarantees synchronous execution semantics: when `execute` returns success, declared outputs are ready for consumption according to their memory contract.

The ABI is designed so future append-only capabilities may add asynchronous execution, events, external CUDA streams, or scheduler integration without changing the V1 synchronous contract.

A heterogeneous asynchronous scheduler is explicitly outside the first migration.

## 17. Build architecture

Current build-wide choices shall be decomposed conceptually into independent capabilities:

- CUDA/image compute support
- ONNX Runtime backend plugin
- TensorRT backend plugin
- typed model/domain targets

Enabling one execution backend must not require disabling another execution backend in the same application deployment.

Backend SDK dependencies remain target-local to their plugin implementations.

The application links against KFCore Core and typed model APIs rather than linking directly to every execution SDK it may load at runtime.

## 18. Migration strategy

Migration is incremental; existing behavior is first protected by parity tests.

Recommended order:

1. Freeze current YOLO ORT CPU/ORT CUDA/TensorRT behavior with semantic parity tests.
2. Introduce backend-neutral primitive types required by the C++ Core and C ABI.
3. Introduce `DynamicLibrary` and Plugin ABI V1 without changing public YOLO semantics.
4. Implement ONNX Runtime and TensorRT runtime plugins behind the ABI.
5. Introduce `ModelPackage`, artifact provenance validation, `ExecutionPolicy`, `ExecutionRoute`, and `ModelResolver`.
6. Migrate YOLO as the first complete backend-neutral proof: one public typed YOLO API, multiple runtime plugins.
7. Remove build-wide ORT/TensorRT mutual exclusion after coexistence tests prove isolation.
8. Add RTMW using the same model/runtime boundary and first-class coordinate transforms.
9. Migrate face/hand models gradually; do not block the runtime architecture on migrating every existing model at once.
10. Retire runtime-specific public model APIs after migration and replacement tests are complete.

## 19. Acceptance criteria

The architecture is considered proven when all of the following hold:

- One process can load ONNX Runtime and TensorRT plugins simultaneously.
- Two different models can execute concurrently or sequentially through different backends without build-time backend exclusivity.
- The same logical YOLO model can be instantiated through two supported execution routes while returning the same semantic result type.
- A CUDA-only model cannot silently execute on CPU.
- An exact TensorRT policy fails deterministically when no compatible engine exists.
- An ordered policy falls back only in the caller-specified order.
- Artifact provenance rejects an engine that does not correspond to the declared source model/version.
- Plugin ABI tests verify no STL/C++ exception/allocator ownership crosses the boundary.
- A backend DLL can be replaced by another ABI-compatible implementation without recompiling typed model code.
- Backend unloading cannot occur while handles/function pointers originating from the plugin remain live.
- Existing YOLO semantic regression tests remain green through the migration.

## 20. Resulting dependency direction

```text
Application
    |
    v
Pipelines
    |
    v
Typed Models (YOLO / RTMW / Face / Hand)
    |                     \
    |                      \ model-semantic image/compute
    v                       v
ModelPackage + Resolver   CPU/CUDA image processing
    |
    v
ExecutableModel / ExecutionContext
    |
    v
Stable Plugin C ABI
    |
    +-- ONNX Runtime backend DLL
    +-- TensorRT backend DLL
    +-- future execution backends
```

The central invariant is:

> Logical model identity and model semantics are stable; execution backend, device/provider, and executable artifact are runtime-selected implementation details.