# Generic TensorRT Runtime and Face Model Adapters

## Context

KFCore currently exposes TensorRT only through `KFCore::tensorrt_yolo`. Its engine loader,
execution context, CUDA buffers, and errors are intentionally coupled to the YOLO detector
contract. The local `yolo-models` directory also contains ArcFace, age/gender, and 68-landmark
ONNX models that are valid ONNX graphs but do not implement a YOLO output contract.

The `faceswap` project demonstrates the model-specific preprocessing and output meanings, but it
uses ONNX Runtime, creates CPU tensors, and does not provide a reusable GPU tensor execution
boundary. No source code is copied from that project; only the observed model contracts are used.

## Decision

Add two opt-in, independently exported modules:

- `KFCore::tensorrt_runtime`: a model-neutral TensorRT engine and synchronous executor.
- `KFCore::face_models`: strict typed adapters for the local Face68, ArcFace, and age/gender
  engines. These adapters consume already prepared NCHW FP32 tensors in host or CUDA memory.

The existing `KFCore::tensorrt_yolo` API and implementation remain unchanged in this change. A
later migration may make YOLO consume the generic runtime after equivalent performance and
integration tests exist.

## Runtime API and state ownership

`kfcore::tensorrt::Engine` owns immutable TensorRT runtime/engine state and extracted tensor
metadata. It is shareable across threads. Each `Engine::create_executor()` call creates one
`Executor` that owns exactly one mutable execution context, CUDA stream, device staging buffers,
and pinned-host staging buffers. Calls on one executor must not overlap; separate executors may run
concurrently.

The public data unit is a named `TensorView`:

- `name`, `data_type`, and `shape` describe the logical tensor.
- `data`, `byte_size`, and `memory_kind` describe borrowed storage.
- Input storage is const; output storage is mutable.
- No pointer is retained after `run()` returns.

`Executor::run(inputs, outputs)` is synchronous. CUDA-device views are bound directly. Host inputs
are copied through executor-owned pinned/device staging. Host outputs are produced in device
staging and copied to the caller before the stream is synchronized and `run()` returns. On a CUDA
or TensorRT failure after work was submitted, the executor attempts to synchronize before
propagating the original error so borrowed storage is no longer in use.

The engine is the only source of tensor names, I/O modes, types, physical formats, and profile
bounds. Caller shapes select profile 0 and must lie inside its bounds. Output shapes are resolved
from the execution context after input shapes are set. Missing, duplicate, extra, mismatched, or
unresolved tensors fail fast.

## Supported contracts and limits

The first runtime version accepts linear device tensors with TensorRT scalar types FP32, FP16,
INT8, INT32, BOOL, UINT8, BF16, and INT64. Packed/vectorized formats, shape-inference I/O tensors,
output allocators, aliased I/O, and data-dependent output dimensions are rejected as contract
mismatches rather than silently approximated.

`EngineOptions` provides positive hard limits for serialized engine bytes, tensor count, aggregate
input bytes, and aggregate output bytes. All shape products and byte sums use checked arithmetic.
Buffers grow only within those limits and are reused by later calls. There is no queue and no
fallback backend; a concurrent call on the same executor is rejected.

## Face model adapters

Each adapter owns a shared generic engine and one executor. Construction validates one exact
contract and records names in options rather than guessing alternatives:

- Face68: FP32 `[N,3,256,256]` input; FP32 `[N,68,3]` landmark output. Optional heatmap outputs are
  still required if present in the serialized engine but are discarded into bounded adapter-owned
  host storage. The result contains 68 `(x, y, score)` values in the model's 256-pixel coordinate
  system.
- ArcFace: FP32 `[N,3,112,112]` input; FP32 `[N,512]` output. The result is 512 raw embedding
  values per item; normalization is not implicit.
- Age/gender: FP32 `[N,3,224,224]` input; FP32 `[N,2]` output. The adapter returns the two raw logits
  because the local graph's semantic order must be established by model provenance or a golden
  test before exposing named age/gender fields.

Inputs may have a dynamic batch only; spatial and channel dimensions are fixed by the adapter.
Adapter output buffers have a configured maximum batch and are allocated during construction, so
steady-state inference does not allocate output storage.

Image preprocessing is deliberately outside these adapters. Face68 needs a bounding-box affine
crop, ArcFace needs five-point similarity alignment, and age/gender needs an ROI resize with
ImageNet normalization. The existing ImageProcessor only implements letterbox, so pretending it
implements these transforms would produce valid-looking but wrong results. New ROI/affine
ImageProcessor operations will be designed and tested separately, after this tensor boundary is
stable.

## Errors and compatibility

The new modules use `TensorRtError` and `FaceModelError` domains with stage-rich messages. Model
contract violations are separated from invalid caller views and runtime failures. Exceptions are
translated only at the adapter boundary where a runtime error gains model context.

This is an additive API and package change. Existing options, targets, serialized formats, YOLO
behavior, and deployment remain unchanged. `KFCORE_BUILD_FACE_MODELS=ON` requires
`KFCORE_BUILD_TENSORRT_RUNTIME=ON`. Both default to `OFF`.

Installed packages expose capability flags and call `find_dependency(CUDAToolkit)` and
`find_package(TensorRT MODULE)` whenever the runtime was built. Disabling both new options fully
rolls back the feature without data migration.

## Verification

- Unit tests cover shape/byte arithmetic, exact view matching, contract validation, result decoding,
  and error classification without requiring a GPU.
- CUDA/TensorRT integration tests are opt-in and use trusted engine paths supplied through cache
  variables. They run at least ArcFace and age/gender with bounded zero input, validate finite
  outputs and exact result sizes, and exercise both host and CUDA input views.
- Installed-consumer CTest verifies target export and transitive dependency discovery.
- Existing YOLO, ImageProcessor, SIFT, and package tests remain green.
