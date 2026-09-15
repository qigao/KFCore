# Face models

`vision/core/face_models` is backend-neutral. Model execution is selected with
`runtime::ExecutionPolicy`; there are no face-specific CPU/TensorRT backend
classes and no `FaceMeshPipeline` abstraction.

## Face detection and FaceMesh

`FaceDetector` and `FaceLandmarker` are concrete typed runtime models:

```cpp
#include <kfcore/face_models/runtime.hpp>

using namespace kfcore;

auto detector = face_models::FaceDetector::load(
    runtime, detector_package, detector_policy);
auto landmarker = face_models::FaceLandmarker::load(
    runtime, landmarker_package, landmarker_policy);
```

For the common detector -> landmark composition, `FaceMesh` owns the two concrete
models directly:

```cpp
auto face_mesh = face_models::FaceMesh::load(
    runtime,
    detector_package, detector_policy,
    landmarker_package, landmarker_policy);

face_models::FaceMeshFrame frame = face_mesh->infer(image);
```

The detector and landmarker policies are independent and explicit. `FaceMesh`
contains only model-semantic composition, thresholding, timing, and contract
validation; it does not introduce a virtual backend layer.

Canonical Model Package types are `face.detector` and `face.landmarker`.
Dynamic detector output `[1,-1,6]` uses plugin ABI v1.2 and preserves the actual
detection count, including zero detections.

## Other typed models

The runtime target also exposes:

- `Face68`
- `ArcFace`
- `AgeGender`
- `InSwapper`
- `GfpGan`

These prepared-tensor models use the same Model Package/runtime resolver and do
not know whether execution is supplied by TensorRT or ONNX Runtime.

## Build targets

- `KFCore::face_model_core` — shared face types, decode/geometry and embedding helpers.
- `KFCore::face_model_runtime` — concrete runtime-loaded typed models and `FaceMesh`.

Both targets are static archives. TensorRT and ONNX Runtime remain runtime-loaded
execution plugins rather than face-specific libraries.
