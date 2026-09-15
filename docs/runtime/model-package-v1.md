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
      "runtime_version": "11.2.1.2",
      "platform": "windows-x86_64",
      "hardware_compatibility": "exact-device",
      "device_name": "NVIDIA GeForce RTX 4090",
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
| `hand.palm-detector` | Palm stage of `kfcore::hand_models::HandDetector` |
| `hand.landmarker` | Landmark stage of `kfcore::hand_models::HandDetector` |
| `hand.gesture-classifier` | Static-pose stage of `kfcore::hand_models::HandDetector` |
| `gesture.temporal-gru` | `kfcore::hand_gesture::TemporalGestureRecognizer` |

The three Hand packages remain separate logical models. Each can use its own `ExecutionPolicy`; `HandDetector` does not implicitly force Palm, landmark, and static-pose classification onto the same backend or device.

The temporal gesture recognizer is a fourth, independent logical model. V1 deliberately requires an explicit ONNX Runtime CPU route for it, while Hand detection may independently run on TensorRT CUDA, ORT CUDA, or ORT CPU.

## YOLO artifact flavors

`yolo-detection` artifacts must declare `flavor` explicitly. KFCore does not infer decoder semantics from the selected backend, output count, tensor names, or tensor shapes.

Supported V1 flavors are:

- `raw-yolo` — one raw detection head with semantic shape `[1,4+C,A]`; KFCore performs score filtering and NMS.
- `compact-nms` — one NMS-complete tensor with semantic shape `[1,N,6]`, where each row is box coordinates, score, and class id.
- `efficient-nms` — four NMS-complete tensors for detection count, boxes, scores, and labels.

A static `compact-nms` artifact can expose `[1,N,6]`. A data-dependent compact artifact can expose `[1,-1,6]`; KFCore then uses plugin ABI v1.2 bounded dynamic Host output and validates the actual `[1,N,6]` shape after execution. `N=0` is a valid empty detection result. The configured `YoloDetectorOptions::max_detections` remains the hard caller-side bound for dynamic N.

Different artifacts for the same logical YOLO model may use different flavors as long as every flavor maps to the same typed semantic API: `Image -> DetectionFrame`.

## Temporal gesture artifact flavor

`gesture.temporal-gru` V1 uses the explicit artifact flavor:

```text
causal-gru-v1
```

The required V1 artifact is an ONNX model routed explicitly to ONNX Runtime CPU:

```json
{
  "id": "onnx-cpu",
  "format": "onnx",
  "path": "temporal_gesture.onnx",
  "flavor": "causal-gru-v1",
  "sha256": "<64 lowercase hex characters>",
  "backend": "onnxruntime",
  "device": "cpu"
}
```

The typed runtime validates the fixed FP32 tensor contract at load time:

```text
inputs
  features          [1,78]
  hidden_in         [2,1,64]

outputs
  gesture_logits    [1,8]
  phase_logits      [1,4]
  hidden_out        [2,1,64]
```

All tensors are fixed-shape Host tensors in V1. No plugin ABI extension and no dynamic-output capability are required. A future TensorRT derived artifact may be added under the ordinary exact-runtime TensorRT rules, but TensorRT is not required for Temporal Gesture V1.

## TensorRT rules

A `tensorrt-engine` artifact is a derived deployment artifact, not a model identity. Model Package V1 deliberately supports deterministic exact-runtime plans only. It does not enable TensorRT version-compatible plans or embedded/external lean-runtime host code.

Every TensorRT artifact therefore requires all of the following:

- `backend` is `tensorrt`.
- `device` is `cuda` or an exact `cuda:N` route constraint.
- `source_artifact` names an ONNX artifact in the same package.
- `source_sha256` exactly matches the source artifact digest.
- `runtime_version` records the exact TensorRT `major.minor.patch.build` used to build the engine.
- `platform` is one of `windows-x86_64`, `windows-aarch64`, `linux-x86_64`, or `linux-aarch64`.
- `compute_capability` uses `major.minor` form, for example `8.9`.
- `hardware_compatibility` is either `exact-device` or `same-compute-capability`.
- `device_name` is required for `exact-device` and must match the CUDA device name reported by the runtime.
- `precision` is explicit, for example `fp32`, `fp16`, or `int8`.

`exact-device` is the conservative default for engines built without an explicit TensorRT hardware-compatibility level. `same-compute-capability` may only be declared for an engine intentionally built with TensorRT same-compute-capability hardware compatibility, and KFCore only accepts that mode for `runtime_version` 10.9 or newer. TensorRT 8.6/Pascal packages therefore use `exact-device` in Model Package V1. The manifest is a deployment assertion, not a mechanism that changes an already-built engine.

The resolver requires exact `runtime_version`, exact `platform`, and the declared hardware compatibility before it calls the backend probe or attempts engine deserialization. An ABI older than v1.3 cannot report the TensorRT build component and therefore cannot prove compatibility for a V1 TensorRT package.

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

Temporal Gesture V1 is stricter: its typed loader requires exactly one preference, `onnxruntime/cpu`.

## Runtime loading

```cpp
kfcore::runtime::Runtime runtime;
(void)runtime.load_backend("plugins/kfcore_backend_tensorrt.dll");
(void)runtime.load_backend("plugins/kfcore_backend_onnxruntime.dll");

const auto package = kfcore::runtime::ModelPackage::load("models/rtmw-l-384x288");
auto pose = kfcore::pose::Rtmw::load(runtime, package, policy);
```

The same logical model API is used regardless of which execution backend is selected, except where a typed V1 contract deliberately constrains the route such as `gesture.temporal-gru`.

## Dynamic outputs

Plugin ABI v1.2 adds bounded dynamic Host outputs for tensors whose actual shape is known only after execution. The caller supplies a maximum byte capacity; the backend returns the actual shape and actual byte count after a synchronous run.

This is used for data-dependent outputs such as the Palm detector `[N,8]`, dynamic compact-NMS `[1,N,6]`, and dynamic face-detection `[1,N,6]` results. Zero-sized result axes are valid when the typed model semantics allow them. It is not an implicit unbounded allocation API and it does not claim dynamic CUDA-memory interoperability.

Plugin ABI v1.3 adds the execution-runtime build component used for exact TensorRT engine compatibility. It does not change the execution function table.

## Inspection and integrity validation

KFCore builds the `kfmodel` utility by default:

```text
kfmodel inspect  <package-directory>
kfmodel validate <package-directory>
```

`validate` checks manifest rules, package-relative path confinement, artifact SHA-256 values, derived-artifact source provenance, and the required TensorRT deployment metadata. Runtime selection additionally rejects `same-compute-capability` TensorRT artifacts whose declared runtime is older than 10.9.

Typed tensor semantics such as the Temporal Gesture fixed GRU names/shapes are validated by the typed model loader rather than by the generic package parser.

## Typed model semantics

The manifest is not a preprocessing DSL. Model-specific behavior stays in typed C++ model code:

- YOLO owns letterbox and detection decode semantics.
- RTMW owns bbox padding/aspect correction, affine preprocessing, normalization, SimCC decoding, and coordinate restoration.
- Face models own face-specific preprocessing/decoding contracts.
- Hand owns Palm decode, per-hand landmark geometry, static-pose feature construction, and stage composition.
- Temporal Gesture owns the fixed 78-value canonical hand feature encoding, recurrent hidden-state semantics, softmax/event decoding, and per-track state transaction rules.
- backend DLLs only execute tensors.

This keeps execution backends interchangeable without moving model semantics into plugins.
