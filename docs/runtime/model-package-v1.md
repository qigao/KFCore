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

`artifacts` is represented by a TBE `group<Artifact>` internally but is ordinary JSON array syntax in `model.json`.

## Canonical model types

Model Package V1 uses canonical model identities only. There are no compatibility aliases for old backend-specific model names.

| `model_type` | Typed API |
| --- | --- |
| `yolo-detection` | `kfcore::yolo::YoloDetector` |
| `pose.rtmw` | `kfcore::pose::Rtmw` |
| `face.detector` | `kfcore::face_models::FaceDetector` |
| `face.landmarker` | `kfcore::face_models::FaceLandmarker` |
| `face.face68` | `kfcore::face_models::Face68` |
| `face.arcface` | `kfcore::face_models::ArcFace` |
| `face.age-gender` | `kfcore::face_models::AgeGender` |
| `face.inswapper` | `kfcore::face_models::InSwapper` |
| `face.gfpgan` | `kfcore::face_models::GfpGan` |
| `hand.palm-detector` | Palm stage of `kfcore::hand_models::HandBackend` |
| `hand.landmarker` | Landmark stage of `kfcore::hand_models::HandBackend` |
| `hand.gesture-classifier` | Gesture stage of `kfcore::hand_models::HandBackend` |

The three Hand packages remain separate logical models. Each can use its own `ExecutionPolicy`; the pipeline does not implicitly force Palm, landmark, and gesture classification onto the same backend or device.

## YOLO artifact flavors

`yolo-detection` artifacts must declare `flavor` explicitly. KFCore does not infer decoder semantics from the selected backend, output count, tensor names, or tensor shapes.

Supported V1 flavors are:

- `raw-yolo` — one raw detection head with semantic shape `[1,4+C,A]`; KFCore performs score filtering and NMS.
- `compact-nms` — one NMS-complete tensor with semantic shape `[1,N,6]`, where each row is box coordinates, score, and class id.
- `efficient-nms` — four NMS-complete tensors for detection count, boxes, scores, and labels.

A static `compact-nms` artifact can expose `[1,N,6]`. A data-dependent compact artifact can expose `[1,-1,6]`; KFCore then uses plugin ABI v1.2 bounded dynamic Host output and validates the actual `[1,N,6]` shape after execution. `N=0` is a valid empty detection result. The configured `YoloDetectorOptions::max_detections` remains the hard caller-side bound for dynamic N.

Different artifacts for the same logical YOLO model may use different flavors as long as every flavor maps to the same typed semantic API: `Image -> DetectionFrame`.

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

## Dynamic outputs

Plugin ABI v1.2 adds bounded dynamic Host outputs for tensors whose actual shape is known only after execution. The caller supplies a maximum byte capacity; the backend returns the actual shape and actual byte count after a synchronous run.

This is used for data-dependent outputs such as the Palm detector `[N,8]` result and dynamic compact-NMS `[1,N,6]` detection results. Zero-sized result axes are valid when the typed model semantics allow them. It is not an implicit unbounded allocation API and it does not claim dynamic CUDA-memory interoperability.

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
- Face models own face-specific preprocessing/decoding contracts.
- Hand owns Palm decode, per-hand landmark geometry, gesture feature construction, and stage composition.
- backend DLLs only execute tensors.

This keeps ONNX Runtime and TensorRT interchangeable without moving model semantics into execution plugins.
