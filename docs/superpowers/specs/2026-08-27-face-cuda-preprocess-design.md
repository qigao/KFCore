# Face CUDA preprocessing design

## Scope and measured unit

The face application keeps detection, model outputs, embedding projection, and final composition
semantics unchanged. This change replaces Face68, ArcFace, InSwapper, and GFPGAN OpenCV affine
plus CPU HWC-to-NCHW normalization with an ImageProcessor-owned CUDA stage. Optional AgeGender
keeps its CPU ROI resize so its OpenCV edge-replication semantics remain exact. The performance
unit is one frame upload plus its affine operations; engine loading and TensorRT execution are
measured separately.

## Ownership and lifetime protocol

- The caller may stage one BGR8/RGB8 host or CUDA image once, then reuse the returned packed CUDA
  image for detection and multiple synchronous preprocessing calls.
- A CUDA image must belong to the processor's configured device and remains borrowed until the
  call returns. A host image is copied row-by-row into reusable pinned storage before upload.
- The processor exclusively owns its CUDA stream, pinned staging buffer, device staging buffer,
  and device FP16/FP32 NCHW output buffer.
- A staged image view remains valid until the next stage call or destruction and is not invalidated
  by affine processing. This permits one H2D upload per frame and device.
- The returned `TensorView` borrows the output buffer. It remains valid until the next process
  call on that processor or processor destruction. TensorRT inference consumes the view
  synchronously before reuse.
- The production stream is synchronized before returning the view. This is required because the
  TensorRT executor owns a distinct stream and exposes no cross-stream event handoff API.
- The facade remains non-reentrant and single-producer/single-consumer. There is no queue,
  backpressure policy, or cross-thread mutable state.

## Capacity and failure state

- Source and tensor byte counts use checked arithmetic and are rejected above configured hard
  limits. Buffers grow only within those limits and are reused on subsequent calls.
- Validation and allocation failure leave the previous allocation valid. Once CUDA work is
  submitted, an error path synchronizes the processor stream before propagating failure.
- A successful call publishes exactly one borrowed device view. A failed call publishes no new
  view; callers must not continue with an earlier view after attempting another call.
- Destruction occurs after synchronous calls are quiescent. CUDA cleanup runs on the configured
  device; if device selection itself is unavailable, device allocations cannot be safely freed.

## Compatibility and validation

The public face application input remains `cv::Mat`, and results remain host-owned. CPU geometry
helpers remain available for callers and as the numerical reference. Tests compare CUDA affine,
channel order, border behavior, and normalization against the existing OpenCV/CPU path within an
explicit interpolation tolerance. The real-model face-swap integration test verifies detector,
Face68, ArcFace, InSwapper, optional GFPGAN, and composition together.

## Measured result

On 2026-08-27, an RTX 4060 Laptop GPU and a synthetic 1920x1080 BGR frame produced the following
100-sample Release averages:

| operation | OpenCV affine + CPU normalize | CUDA on staged frame |
| --- | ---: | ---: |
| Face68 256x256 | 577.040 us | 22.651 us |
| ArcFace 112x112 | 82.884 us | 11.825 us |
| InSwapper 128x128 | 149.479 us | 23.464 us |
| GFPGAN 512x512 | 1544.929 us | 25.838 us |

The one-time 1080p host-to-device staging average was 710.229 us. One staged frame plus all four
CUDA operations therefore measured about 0.794 ms, versus about 2.354 ms for the four CPU
reference operations. This result is hardware- and image-dependent and is a reproducible local
benchmark, not a universal latency guarantee.
