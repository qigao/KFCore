# OpenCV-free YOLO application data path

## Background and evidence

The YOLO domain demo currently receives a borrowed Turbo Capture frame, copies it into a
single-slot latest-frame mailbox, converts the complete frame to packed BGR with OpenCV Lite,
optionally mirrors that BGR image, and only then preprocesses it into a model tensor.

Turbo Capture's callback contract makes the source bytes invalid when the callback returns. The
Windows backend locks an `IMFMediaBuffer`, invokes the callback synchronously, then unlocks and
releases it. Linux invokes the callback before `VIDIOC_QBUF`; macOS invokes it while a
`CVPixelBuffer` is locked. A pointer retained beyond the callback would therefore be invalid on all
three backends.

At 640 x 480, one packed NV12 frame contains `640 * 480 * 3 / 2 = 460800` bytes. At 30 FPS the
mailbox ownership copy writes 13,824,000 bytes/s, approximately 13.18 MiB/s. This copy is bounded
and preserves latest-frame behavior. In contrast, the current camera path also materializes a
921,600-byte BGR image and subsequently reads it again during resize, color conversion and tensor
normalization. The first optimization target is therefore fused native-format preprocessing, not
unsafe shared mutation or a backend-wide retained-sample protocol.

## Decision

Keep one bounded ownership copy at the capture callback boundary and remove OpenCV from the YOLO
application's codec, preprocessing, overlay and window paths.

Add packed NV12 and I420 as explicit input formats to the existing borrowed image view. Their
layout contract is:

- width and height are positive and even;
- `row_stride` is the Y-plane stride and is even;
- NV12 contains `height` Y rows followed by `height / 2` interleaved UV rows, all using
  `row_stride`;
- I420 contains `height` Y rows using `row_stride`, then U and V planes containing `height / 2`
  rows each using `row_stride / 2`;
- `byte_size` covers the complete required span;
- the pointer remains borrowed until the synchronous CPU operation returns or until queued CUDA
  work and its stream synchronization complete.

Fuse source sampling, optional horizontal mirror, YUV-to-RGB conversion, bilinear letterbox,
normalization and NCHW write. The CPU backend performs one bounded destination traversal. The CUDA
backend stages packed host input once to pinned memory/device memory and performs one kernel over
the destination tensor. Headless camera mode never materializes BGR. Display mode performs a
separate YUV-to-BGR conversion after inference because the display surface needs pixels; this cost
is reported under rendering/conversion metrics rather than inference.

The mailbox remains SPSC in intent but mutex-protected in implementation: one backend callback
producer and one application-loop consumer. Capacity is exactly one frame. Publishing while full
coalesces the previous unread frame. Shutdown wakes the consumer. No pointer escapes the mailbox,
and mutation is limited to the producer-owned unpublished buffer or consumer-owned frame.

## OpenCV-free application adapters

The application uses a small owned packed-BGR type at its outer boundary. Image files are decoded
and encoded through the already-declared `stb` dependency. Decode and encode have explicit byte and
dimension limits and fail fast on unsupported extensions or malformed data. File input remains an
owned image because codec output has no stable external lease.

Overlay drawing operates on the owned BGR buffer and does not mutate tracking facts. It draws a
dark header, high-contrast text and bounded boxes. On Windows, a Win32 window presents the BGR
buffer and translates Q/Escape and R key events. Headless mode has no window dependency. Other
platforms fail fast when interactive display is requested until a native presenter is supplied;
headless file and camera processing remains available.

The existing optional `KFCore::yolo_opencv` adapter and its unrelated example remain available.
`KFCORE_BUILD_YOLO_APPLICATIONS` no longer implies or links OpenCV Lite.

## State and error ownership

- Turbo Capture owns the backend sample only during the callback.
- `LatestFrameMailbox` owns the copied latest frame until it swaps ownership with the consumer.
- The application loop owns the consumed frame, detector, tracker and timing window.
- Image views are read-only borrows; processors retain no source pointer after their documented
  completion boundary.
- Codec, format, capacity, CUDA and inference failures are explicit exceptions at the application
  boundary. The capture callback remains `noexcept`; rejected frames increment a counter.
- No automatic OpenCV fallback is provided.

## Alternatives considered

### Mutable shared frame

Rejected. Allowing capture to overwrite storage while inference reads it is a data race and can
mix two frames in one tensor. Locking the same buffer for the full inference duration degenerates
to synchronous callback processing without expressing ownership.

### Retained backend sample lease

Deferred. It can eliminate the 13.18 MiB/s mailbox copy but requires a public acquire/release state
machine and backend-specific retention: COM sample retention on Windows, delayed V4L2 requeue on
Linux, and retained/locked pixel buffers on Apple platforms. It also consumes scarce driver
buffers while inference stalls. This change is justified only if profiling shows the bounded copy
is at least 20% of end-to-end time or memory bandwidth on a target device.

### Synchronous inference inside the capture callback

Rejected as the default. It is zero-copy but blocks the capture backend, prevents latest-frame
coalescing, couples driver timing to model latency and complicates error propagation and UI event
handling. It may later be exposed as an explicit low-latency mode with measured constraints.

### Change every public image view to a multi-plane variant

Rejected. CodeGraph reports 60 affected symbols across YOLO, face, SIFT and vision-model modules.
Packed NV12/I420 covers the current Turbo Capture contract with a smaller migration surface. A
separate multi-plane view remains the correct future extension for native textures or independently
strided planes.

## Compatibility, migration and rollback

Adding input pixel formats and an appended mirror option preserves existing aggregate initializers
and packed BGR/RGB behavior. YOLO application sources and tests migrate from `cv::Mat` to the owned
BGR type. The application target loses its OpenCV link dependency; the optional OpenCV adapter is
unchanged.

Rollback is a normal revert of this feature branch. No persistent data or model format changes are
introduced. Output filenames and supported JPEG/PNG/BMP image workflows remain stable; exact
encoded bytes are not promised because the codec implementation changes.

## Verification

- Unit tests for NV12/I420 size, stride, capacity, neutral colors, channel order, mirror and
  letterbox parity on CPU.
- CUDA tests comparing NV12/I420 output against the CPU implementation within a documented numeric
  tolerance.
- YOLO helper tests proving native formats are accepted and dimensions remain the decoding fact
  source.
- Codec and overlay tests without OpenCV.
- Capture mailbox contract tests for bounded copy, coalescing, close and rejected formats.
- Focused builds/tests followed by the complete `win-release-user` preset.
- A benchmark at 640 x 480 comparing legacy full-frame conversion plus preprocessing with fused
  native preprocessing, separately reporting mailbox copy, CPU tensor preparation and CUDA tensor
  preparation.

## Measured result (2026-08-29)

Release build on an AMD Ryzen 9 7940HX and NVIDIA GeForce RTX 4060 Laptop GPU, using CUDA 12.8:

| 640 x 480 NV12 operation | Mean latency | Samples |
| --- | ---: | ---: |
| bounded 460,800-byte mailbox copy | 0.0088 ms | 1000 |
| legacy CPU NV12 -> BGR -> letterbox tensor | 18.35 ms | 20 |
| fused CPU NV12 -> letterbox tensor | 12.10 ms | 20 |
| fused CUDA upload + NV12 -> letterbox tensor | 0.141 ms | 100 |

The fused CPU route reduced tensor-preparation latency by 34.1% in this run. The mailbox copy was
0.07% of the fused CPU preparation time, so replacing the capture callback contract with retained
backend samples is not justified by this measurement.

The application also completed two ten-frame headless camera smoke runs on the Logitech BRIO's
native `640x480@30 NV12` mode: ONNX Runtime with `yolov8n-drone.onnx`, and TensorRT 11.2 with the
matching RTX 4060 engine. Both routes reported `convert=0.00 ms`, confirming that headless capture
did not materialize an intermediate BGR image. These short runs validate execution only; they are
not stable throughput measurements because camera startup is included.
