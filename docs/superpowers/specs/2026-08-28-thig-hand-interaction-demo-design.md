# THIG Hand Interaction Demo Design

## Context

KFCore already provides CPU and TensorRT hand inference, per-pipeline ByteTrack/Kalman state,
and backend-neutral THIG hand interaction. It does not yet provide an executable that connects
those pieces to a live camera. OpenCV Lite intentionally has no `videoio` component, while the
installed TurboParser SDK provides native camera enumeration and capture through
`TurboParser::Capture`.

## Scope

Add one opt-in desktop demo that:

- captures a USB camera through Turbo Capture or lists available cameras and native modes;
- converts supported uncompressed Turbo Capture frames to packed BGR for display and inference;
- runs either the compiled CPU/ONNX Runtime or GPU/TensorRT hand backend;
- feeds the tracked hand frame into `HandInteractionPipeline`;
- overlays palm boxes, 21 landmarks, primitive relations, semantic actions, stage timings, and
  capture counters in an OpenCV Lite HighGUI window;
- remains testable without a camera, display, ONNX Runtime session, or TensorRT engine.

The demo does not add a capture API to KFCore, record video, decode MJPEG, search for models,
generate TensorRT engines, or change any installed KFCore public library interface.

## Architecture

The executable boundary owns all device and UI state:

```text
Turbo Capture callback
  -> LatestFrameMailbox (bounded owned frame, latest-frame-wins)
  -> Turbo frame-to-BGR adapter (OpenCV Lite imgproc)
  -> HandPipeline (CPU or TensorRT backend)
  -> HandInteractionPipeline
  -> deterministic overlay composer
  -> OpenCV Lite HighGUI
```

`demo_capture` owns device enumeration, exact native mode selection, capture lifecycle, and the
mailbox. `demo_cli` owns argument validation and backend/model path selection. `demo_frame` owns
pixel-format validation and BGR conversion. `demo_ui` owns deterministic drawing and key mapping.
`hand_interaction_demo.cpp` is the only orchestration/event-loop translation unit.

The executable is built only when `KFCORE_BUILD_HAND_INTERACTION_EXAMPLES=ON`. That option requires
`KFCORE_BUILD_HAND_INTERACTION=ON` and at least one of `KFCORE_BUILD_VISION_MODELS_CPU` or
`KFCORE_BUILD_VISION_MODELS_TENSORRT`. OpenCV Lite and Turbo Capture remain example-only
dependencies.

## Capture and memory protocol

| Item | Contract |
|---|---|
| Data unit | One fixed-mode video frame plus format, dimensions, timestamp, and serial. |
| Fact source | Turbo Capture owns callback memory; after callback copy, the mailbox-owned vector is authoritative. |
| Ownership | Callback input is borrowed until return. `publish()` copies into one pre-reserved vector. `take_latest()` swaps that vector with a caller-owned pre-reserved vector. |
| Lifetime | No Turbo pointer escapes the callback. The consumer frame remains valid until the next successful `take_latest()` on that caller-owned object. |
| Topology | Exactly one Turbo Capture producer callback and one UI/inference consumer thread (SPSC). |
| Ordering | Consumer observes monotonically increasing serials, but intermediate unread frames may be explicitly coalesced. |
| Capacity | Exactly two pixel vectors, each capped by configurable `max_frame_bytes`; default 64 MiB. Byte counts use checked multiplication. |
| Backpressure | Producer never waits for inference. If an unread frame exists, it is replaced and `coalesced_frames` increments. Oversized/malformed frames are rejected and counted. |
| Failure | Callback is `noexcept` and reports rejection through counters. Device/control-plane failures throw before the event loop or close the mailbox through the state callback. |
| Shutdown | UI requests exit; main thread calls `turbo_capture_stop()` so callbacks drain, then destroys capture/device handles. The callback never calls stop/destroy. |
| Observability | Captured, consumed, coalesced, rejected, capture timestamp, model stage timing, THIG timing, and end-to-end frame timing are displayed. |

A mutex and condition variable protect the one published slot. This is preferred over a lock-free
ring because capture is a single low-frequency producer, the consumer holds the lock only for a
vector swap, and the required semantics are coalescing rather than FIFO delivery.

## Mode and pixel policy

`--list-cameras` prints devices and every native mode without loading models. Normal capture uses
`--camera <index>` and either `--mode <mode-id>` or an exact
`--width/--height/--fps` request. If neither is supplied, the deterministic default request is
1280x720 at 30 fps. Among exact matches, formats are preferred in this order: NV12, I420, BGRA,
RGB24. MJPEG is listed but rejected for inference because this demo does not silently add a JPEG
decode path. If no supported exact mode exists, startup fails and tells the user to run
`--list-cameras`.

NV12, I420, BGRA, and RGB24 inputs are validated against their exact packed byte counts before
OpenCV conversion. Resulting BGR storage is owning and is the single host image shared by display
and the selected inference backend. On TensorRT, the backend's existing ImageProcessor uploads and
preprocesses this host view on the GPU.

## CLI and backend behavior

- `--backend cpu --model-dir <dir>` derives the three trusted hand ONNX paths already used by the
  integration test.
- `--backend tensorrt --palm <engine> --hand <engine> --classifier <engine>` requires all three
  explicit engine paths.
- If only one backend was compiled, omitting `--backend` selects it. If both were compiled,
  `--backend` is required.
- Every model path must be an existing readable regular file. There is no backend fallback.
- `Q`/Escape quits and `R` resets both tracking and THIG temporal state.

## Compatibility and rollback

All additions are opt-in and example-local. Existing library targets, package exports, public APIs,
presets, CPU/GPU inference semantics, and OpenCV-free CPU libraries remain unchanged. Rollback is
disabling `KFCORE_BUILD_HAND_INTERACTION_EXAMPLES` or reverting the new example sources and option.

The TurboParser SDK must export `TurboParser::Capture`; having only `turbo_capture.dll/.lib/.h` is
considered an invalid installation and configuration fails. Reinstalling TurboParser with its
`install-win-capture-release-user` preset restores the official export target.

## Validation

- TinyTest covers CLI validation, deterministic mode selection, exact byte validation/conversion,
  latest-frame coalescing, shutdown wakeup, key decoding, and non-mutating overlay composition.
- CPU preset builds the demo and runs all unit tests plus the existing real-model vision test.
- TensorRT preset configures and builds the same demo when trusted engines are supplied; existing
  GPU integration remains the model-path proof.
- A manual Windows smoke test lists cameras, opens one exact uncompressed mode, displays live
  landmarks/actions, resets with `R`, and exits with `Q` while counters remain bounded.
