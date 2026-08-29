# YOLOv8 Domain Applications Design

## Context

KFCore already has a TensorRT YOLO detector for EfficientNMS and compact `[B,N,6]`
outputs, a ByteTrack/Kalman session, CPU/CUDA ImageProcessor implementations, OpenCV Lite
drawing support, and Turbo Capture integration in the hand demo. The local model directory adds
three compact-NMS YOLOv8 applications:

| Application | Model input | Output | Classes |
|---|---:|---:|---|
| drone | `1x3x640x640` FP32 | `1x300x6` FP32 | `class-0`, `class-1` |
| football | `1x3x960x960` FP32 | `1x300x6` FP32 | `ball`, `goalkeeper`, `player`, `referee` |
| parking | `1x3x640x640` FP32 | `1x300x6` FP32 | `space-empty`, `space-occupied` |

The drone ONNX metadata contains only numeric names, so the application must not invent semantic
labels for its two classes.

## Scope

Add one opt-in `yolov8_domain_demo` that:

- selects `drone`, `football`, or `parking` through an explicit application profile;
- runs ONNX Runtime on CPU or the existing TensorRT detector on GPU without backend fallback;
- processes either a sorted image directory or a Turbo Capture camera stream;
- defaults camera selection to an exact 640x480, 30 fps, NV12 native mode;
- feeds every detection frame into KFCore `ByteTrackSession`;
- displays readable class names, confidence, track IDs, current counts, parking occupancy, FPS,
  capture counters, model load time, and per-frame stage timings;
- supports headless bounded camera runs and annotated image-directory output for repeatable tests
  and benchmarks.

The change does not commit model/engine binaries, decode MJPEG, add a public capture API, expose a
new installed ONNX detector API, infer drone class semantics, or alter existing detector/tracker
behavior.

## Architecture

The application remains inside `tensorrt_yolo` because its domain data is exactly the existing
YOLO `DetectionFrame`/`TrackFrame` contract:

```text
image directory ---------------------> packed host BGR -------------------+
Turbo Capture NV12 -> latest mailbox -> OpenCV Lite NV12-to-BGR ----------+
                                                                          |
                                  +---------------------------------------+
                                  v
                     CPU ONNX adapter | TensorRT detector
                                  v
                         DetectionFrame (fact source)
                                  v
                         score/class validation
                                  v
                         ByteTrackSession
                                  v
                 domain summary + overlay + output/display
```

Application source files are separated by responsibility:

- `yolo_domain_profile`: class contracts, labels, filtering, and summaries;
- `yolo_domain_cli`: mutually exclusive source/backend configuration and fail-fast validation;
- `yolo_domain_capture`: camera enumeration, exact mode selection, lifecycle, and bounded mailbox;
- `yolo_domain_frame`: validated Turbo pixel conversion to owning BGR storage;
- `yolo_domain_onnx`: strict single-input/single-output ONNX Runtime CPU adapter;
- `yolo_domain_ui`: deterministic readable overlays and key decoding;
- `yolov8_domain_demo`: backend construction and event-loop orchestration only.

The existing `detector_helpers` implementation is moved into one internal static target so the
CPU and TensorRT adapters share identical source-image validation, letterbox geometry, and compact
NMS decoding. It is not installed or exposed as a public target.

## State and ownership

| State | Owner | Mutation | Failure state |
|---|---|---|---|
| ONNX session/input/output | one CPU adapter instance | synchronous `detect()` | exception; instance remains owned, no fallback |
| TensorRT context/stream/buffers | existing detector instance | synchronous `detect()` | existing `YoloError` semantics |
| tracking/Kalman state | one `ByteTrackSession` | once per accepted frame | frame fails; no partial output is published |
| capture callback bytes | Turbo Capture until callback returns | device thread | copied or rejected before return |
| latest captured frame | bounded mailbox | producer copies, consumer swaps | malformed/oversized frames counted and rejected |
| domain summary | derived from current tracked frame | rebuilt every frame | never independently persisted |

The packed BGR frame is the single host image shared by display and inference. CPU inference reads
it directly. TensorRT uploads and preprocesses it through the existing CUDA ImageProcessor. There
is no hidden CPU/GPU backend switch.

## CPU model contract

The ONNX adapter accepts only:

- one FP32 input named `images`, shape `[1,3,H,W]`, with positive static `H/W`;
- one FP32 output named `output0`, shape `[1,N,6]`, with positive bounded static `N`;
- host BGR8/RGB8 input with valid dimensions and stride.

It uses `CpuImageProcessor::letterbox_nchw`, RGB unit-range normalization, and border value 114,
then decodes the in-graph NMS rows through the same compact-NMS helper used by TensorRT. Model,
source, input tensor, and output tensor byte counts have configurable hard limits. Calls on one
instance are non-reentrant; applications create independent instances for concurrent work.

## Capture and source policy

Normal camera startup selects an exact 640x480, 30 fps, NV12 native mode unless `--mode` selects an
explicit supported native mode. It fails with a useful message if the exact default is absent.
`--list-cameras` reports devices and modes without loading a model. I420, RGB24, and BGRA remain
valid only through explicit mode selection. MJPEG is listed but rejected because adding an implicit
decoder would change resource and latency semantics.

The callback never blocks on inference. It copies into one pre-reserved bounded slot; an unread
frame is explicitly coalesced and counted. The consumer swaps storage with a second pre-reserved
buffer. Shutdown stops capture before destroying the handle.

Image-directory mode validates input/output directories, bounds the enumerated frame count, sorts
paths deterministically, and never overwrites the source directory.

## Observability and performance

The overlay and final console report separate:

- model load time;
- capture wait and NV12/BGR conversion time;
- detector time, explicitly defined as preprocess + backend inference + compact-NMS decode;
- ByteTrack time;
- overlay/output time;
- total frame time, FPS, processed/coalesced/rejected frame counts.

Rolling statistics are bounded and report current/mean/P50/P95 values without per-frame INFO
logging. CPU/GPU comparisons use the same 640x480 NV12 source and the same application model;
football still performs its contract-required 960x960 model preprocessing.

## Build and compatibility

`KFCORE_BUILD_YOLO_APPLICATIONS` enables the demo and requires YOLO tracking, OpenCV Lite, and
TurboParser Capture. `KFCORE_BUILD_YOLO_APPLICATIONS_CPU` adds ONNX Runtime CPU support. TensorRT
support is present only when `KFCORE_BUILD_TENSORRT_YOLO` is enabled. Configuration fails if no
backend is available.

A dedicated Windows preset enables both backends and takes model paths from cache variables. No
machine-specific path is compiled into source. Existing targets, exported APIs, package metadata,
presets, and default builds remain unchanged. Rollback is disabling the application option or
reverting the example-only sources and build wiring.

## Alternatives considered

1. Put domain behavior into `TensorRtDetector`: rejected because labels, parking occupancy, camera
   state, and UI are application concerns and would couple the inference library to one deployment.
2. Add a new installed `yolo_applications` library: rejected for now because the requested behavior
   has only one consumer and would prematurely commit a public ABI.
3. Reuse `vision_models`' private ONNX session: rejected because it throws vision-model-specific
   errors and would create a reverse dependency from YOLO into hand/face model code.
4. Use OpenCV DNN for CPU: rejected because ONNX Runtime is already the repository's explicit CPU
   inference dependency and has strict tensor-contract validation.

## Validation

- TinyTest covers profiles, unknown classes, score filtering, summaries, CLI rules, exact camera
  mode selection, byte bounds/coalescing, pixel conversion, UI text/key behavior, and ONNX adapter
  validation.
- Real-model CPU integration loads and executes all three local ONNX files and validates finite,
  in-bounds detections with legal class IDs.
- Existing TensorRT contract/integration tests continue to cover the shared compact-NMS decoder;
  generated local engines provide application smoke tests for all three models.
- The complete application preset builds and runs its focused tests, then the full CTest suite.
