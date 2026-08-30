# THIG demo FaceMesh overlay

## Decision

Extend the existing THIG camera demo with an optional face path composed from
YOLOv12-face detection and the existing MediaPipe 468-landmark model. Hand and
face inference consume the same immutable captured BGR frame, and the UI draws
both results into the single existing HighGUI window.

```text
Turbo Capture -> owning BGR frame --+-> HandPipeline -> THIG
                                    |
                                    +-> FaceDetector -> FaceLandmarker
                                    |
                                    +-> one composed overlay -> one imshow
```

## Boundaries and state ownership

`vision_models` owns backend-neutral face detection and FaceMesh result types,
the detector/landmarker interfaces, and the stateless composition pipeline.
The CPU adapter owns its ONNX Runtime detector session. Existing CPU and
TensorRT face landmark adapters implement the common landmark interface.

The TensorRT demo owns a thin adapter from `KFCore::yolo_tensorrt` detections to
the backend-neutral face contract. This keeps TensorRT-YOLO types out of the
vision-model core and avoids making every TensorRT vision-model consumer depend
on the YOLO runtime.

The demo owns the optional FaceMesh pipeline and passes borrowed frame views for
the duration of one synchronous call. Results own all boxes, points, and timing
values. No face is an ordinary empty result; invalid models, malformed tensors,
resource-limit violations, and runtime failures remain exceptions.

## Configuration and compatibility

FaceMesh remains disabled unless both `--face-detector` and `--facemesh` are
provided. Supplying only one path is rejected. `--face-score` controls the
YOLOv12-face threshold and `--facemesh-score` controls landmark visibility;
both accept finite values in `[0,1]`.

CPU paths name ONNX files. TensorRT paths name engines built for the active
TensorRT/GPU environment. Existing hand-only invocations and public THIG
behavior remain unchanged. Reset still resets hand tracking and THIG state;
the FaceMesh path is stateless.

## Rendering and performance

The overlay draws the selected face box and all 468 image-space landmarks when
the landmark confidence meets the configured threshold. It does not embed an
unverified third-party connectivity table; landmark points are sufficient to
show the model output and retain provenance clarity.

Metrics add face detection, FaceMesh preprocessing, FaceMesh inference, and
FaceMesh total wall time. The first implementation shares the host BGR frame,
but the existing independent TensorRT detector and landmarker adapters retain
their own CUDA staging buffers. A later zero-copy frame context requires a
separate public lifetime contract and is not hidden behind this UI change.

## Verification and rollback

CLI tests cover disabled, complete, partial, duplicate, malformed-threshold,
and model-directory behavior. Core tests cover detector-to-landmarker flow,
empty detections, confidence filtering, and timing propagation. UI tests cover
face-box and landmark drawing without changing the source image. CPU and
TensorRT camera smoke tests load real local models and process bounded frames.

Rollback is additive: omit the two face model arguments or revert the FaceMesh
pipeline and demo adapter changes. Existing hand inference, THIG state, capture,
and rendering remain available.
