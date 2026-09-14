# KFCore Model Package V1

KFCore treats a logical model separately from the execution artifact used to run it. A model package is a directory containing `model.json` plus one or more artifacts such as ONNX files and prebuilt TensorRT engines.

## Layout

```text
models/rtmw-l-384x288/
  model.json
  rtmw.onnx
  artifacts/
    rtmw-trt11-sm89-fp16.engine
```

## Manifest

```json
{
  "schema": "kfcore.model/1",
  "id": "rtmw-l-384x288",
  "version": "1.0",
  "model_type": "pose.rtmw",
  "variant": "coco-wholebody-133",
  "artifacts": [
    {
      "id": "onnx",
      "format": "onnx",
      "path": "rtmw.onnx",
      "flavor": "simcc",
      "sha256": "<64 lowercase hex characters>",
      "backend": "onnxruntime",
      "device": "any"
    },
    {
      "id": "trt11-sm89-fp16",
      "format": "tensorrt-engine",
      "path": "artifacts/rtmw-trt11-sm89-fp16.engine",
      "flavor": "simcc",
      "sha256": "<64 lowercase hex characters>",
      "backend": "tensorrt",
      "device": "cuda",
      "source_artifact": "onnx",
      "source_sha256": "<the exact sha256 value of the onnx artifact>",
      "runtime_major": 11,
      "compute_capability": "8.9",
      "precision": "fp16",
      "profile": "default"
    }
  ]
}
```

`pose.rtmw` is the canonical Model Package V1 type accepted by the typed `kfcore::pose::Rtmw` API. The backend-neutral YOLO API accepts the canonical `yolo-detection` type (and currently also recognizes `yolo` during migration).

`artifacts` is represented by a TBE `group<Artifact>` internally but is ordinary JSON array syntax in `model.json`.

## TensorRT rules

A `tensorrt-engine` artifact is a derived deployment artifact, not a model identity. Model Package V1 therefore requires all of the following:

- `backend` is `tensorrt`.
- `device` is `cuda` or an exact `cuda:N` constraint.
- `source_artifact` names an ONNX artifact in the same package.
- `source_sha256` exactly matches the source artifact digest.
- `runtime_major` records the TensorRT major version used by the engine.
- `compute_capability` uses `major.minor` form, for example `8.9`.
- `precision` is explicit, for example `fp32`, `fp16`, or `int8`.

The resolver compares `runtime_major` against the loaded TensorRT plugin and compares `compute_capability` against the selected CUDA device before attempting engine deserialization.

## Execution policy

There is no implicit fallback. Callers provide an ordered policy explicitly:

```cpp
using kfcore::runtime::ExecutionPolicy;
using kfcore::runtime::ExecutionPreference;

const auto policy = ExecutionPolicy::ordered({
    ExecutionPreference{"tensorrt", "cuda:0"},
    ExecutionPreference{"onnxruntime", "cuda:0"},
    ExecutionPreference{"onnxruntime", "cpu"},
});
```

A model that must not run on CPU simply omits a CPU-compatible artifact/route or uses a policy without one.

## Runtime loading

```cpp
kfcore::runtime::Runtime runtime;
(void)runtime.load_backend("plugins/kfcore_backend_tensorrt.dll");
(void)runtime.load_backend("plugins/kfcore_backend_onnxruntime.dll");

const auto package = kfcore::runtime::ModelPackage::load("models/rtmw-l-384x288");
auto pose = kfcore::pose::Rtmw::load(runtime, package, policy);
```

The same logical model API is used regardless of which execution backend is selected.

## Inspection and integrity validation

KFCore builds the `kfmodel` utility by default:

```text
kfmodel inspect  <package-directory>
kfmodel validate <package-directory>
```

`validate` checks manifest rules, package-relative path confinement, artifact SHA-256 values, and derived-artifact source provenance.

## Typed model semantics

The manifest is not a preprocessing DSL. Model-specific behavior stays in typed C++ model code:

- YOLO owns letterbox and detection decode semantics.
- RTMW owns bbox padding/aspect correction, affine preprocessing, normalization, SimCC decoding, and coordinate restoration.
- backend DLLs only execute tensors.

This keeps ONNX Runtime and TensorRT interchangeable without moving model semantics into execution plugins.
