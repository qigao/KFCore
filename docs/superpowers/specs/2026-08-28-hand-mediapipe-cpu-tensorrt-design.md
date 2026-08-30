# Hand and MediaPipe Landmark CPU/TensorRT Design

## Context

KFCore already owns image preprocessing, a model-neutral TensorRT executor, and ByteTrack/Kalman.
The local model set adds three hand models and one MediaPipe face-landmark model:

- palm detection: `input [1,3,192,192] -> detections [N,8]`
- hand landmark: `input [N,3,224,224] -> xyz [N,63], score [N,1], handedness [N,1]`
- keypoint classifier: `input [N,42] -> class_ids [N]`
- MediaPipe face landmark: `image [1,3,192,192] -> scores [1], landmarks [1,468,3]`

TensorRT 11.2.1 parses all four ONNX files. Palm detection retains ONNX NMS in the engine and therefore
has a data-dependent first output dimension. The current executor rejects such outputs before enqueue.

## Scope

This change adds:

1. bounded data-dependent TensorRT host outputs;
2. shared hand/face-landmark types, geometry, decoding, and ByteTrack orchestration;
3. an OpenCV-free ONNX Runtime CPU backend;
4. a TensorRT CUDA backend using KFCore ImageProcessor;
5. opt-in real-model CPU and GPU integration tests and documented engine conversion.

MediaPipe face detection is not added. Existing YOLOv12-face remains the face detector; the 468-point
landmarker consumes a caller-provided face box. Existing face-swap APIs and model choices do not change.

## Architecture

The subsystem uses a bridge between model-independent orchestration and backend implementations:

- `KFCore::hand_model_core` owns public value types, geometry, classification decoding, and one
  `HandPipeline` instance's ByteTrack state.
- `KFCore::hand_models_cpu` owns ONNX Runtime sessions and CPU preprocessing.
- `KFCore::vision_models_tensorrt` owns TensorRT adapters and one CUDA ImageProcessor per backend instance.

`HandInferenceBackend` is constructor-injected into `HandPipeline`. This keeps tracking independent of
ONNX Runtime and TensorRT and permits deterministic pipeline tests without mocking third-party runtimes.
There is no global backend, session, tracker, or mutable model cache. Multiple instances are independent.

## Public behavior

- CPU entry points accept host `ImageView` values only.
- TensorRT entry points accept host or CUDA `ImageView` values.
- Calls are synchronous and non-reentrant. Concurrent use of one mutable instance fails explicitly.
- Palm confidence, landmark confidence, maximum hands, tensor/model byte limits, device id, and ByteTrack
  thresholds are configurable and validated at construction.
- Keypoint classifier ids map to `Open`, `Closed`, and `Pointer`; any other id is reported as `Unknown`
  rather than indexing outside the label set.
- MediaPipe face landmarks are mapped into source-image coordinates using the same expanded, vertically
  shifted square face ROI for both CPU and GPU.
- Hand tracking uses the palm box as the observation and preserves the backend result through
  `bytetrack_update_ex().detection_index`.

## Data and lifetime protocol

| Item | Contract |
|---|---|
| Data unit | Packed image view, FP32 NCHW tensor, bounded host tensor output, or owning result vector. |
| Fact source | The caller owns input image bytes; each backend instance owns its sessions, executor staging, and CUDA processor buffers; returned result objects own their scalar data. |
| Ownership | Inputs are borrowed only for the synchronous call. TensorRT dynamic outputs are returned as owning host byte vectors. ByteTrack is uniquely owned by one `HandPipeline`. |
| Lifetime | A CUDA `stage()` view remains valid until the next stage call or backend destruction. `process_affine()` views remain valid until the next affine call; inference completes before reuse. No borrowed view escapes a public call. |
| Topology | One caller at a time per instance. Different instances may run concurrently and own independent state. |
| Ordering | One frame is processed palm -> landmark -> classifier -> tracker. Hands retain palm confidence order before tracker association. |
| Capacity | `max_palm_candidates` bounds the Palm engine's data-dependent output (2016 for the trusted model); `max_hands` separately bounds selected detections and ROI inference. Engine input/output limits and per-request dynamic output limits use checked multiplication/addition. |
| Backpressure | No queue exists. Concurrent calls fail with `ConcurrentExecution`; oversized input/output fails with `ResourceLimitExceeded`. |
| Failure | No partial result is returned. TensorRT addresses are cleared after success or failure. ByteTrack is updated only after all model inference for the frame succeeds. |
| Shutdown | Destruction requires no active call; RAII releases sessions, executors, CUDA buffers, and tracker state. |
| Observability | Profiled calls report preprocessing, each inference stage, tracking, and total duration; no hot-path INFO logging is added. |

TensorRT dynamic output requests state a name, data type, and maximum bytes. The executor allocates bounded
device/pinned staging, enqueues synchronously, resolves the actual post-execution shape, validates actual
bytes against both request and engine limits, downloads only actual bytes, and returns owning host storage.

## Compatibility and migration

All new options default OFF. Existing targets, public headers, presets, and behavior remain available.
The only existing public type extended is TensorRT `Executor`, through an additive `run_dynamic()` method.
Consumers opt in by enabling either CPU or TensorRT vision-model targets and linking the corresponding
exported target. Engine files remain deployment assets and are never committed.

Rollback consists of disabling the new CMake options or reverting the additive modules. Existing face and
YOLO modules have no data migration and remain independently usable.

## Validation

- Unit tests cover dynamic-output request validation, geometry, malformed outputs, resource limits,
  tracker association, reset, and independent pipeline instances.
- CPU integration runs all four ONNX models with deterministic bounded inputs.
- TensorRT integration runs the four generated engines on CUDA and compares output shapes and finite values
  with the CPU contracts; Palm specifically exercises the data-dependent output path.
- Existing TensorRT runtime, image processor, trackers, face models, and face application tests are rerun.
