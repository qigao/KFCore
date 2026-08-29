# Packed YUV Pixel Formats Design

## Scope

Add three decoded, single-frame pixel layouts to KFCore: `Nv21`, `Yuy2`, and `Uyvy`.
This work does not add image codecs, video codecs, containers, timestamps, or multi-plane ABI.

## Existing architecture

`kfcore::image::ImageView` is the frame contract. `ImageProcessor` validates and converts it,
TensorRT-YOLO adapts it to model preprocessing, and Face/Vision applications forward the same
borrowed frame. The input remains immutable and valid only for the synchronous call. Owned staging
buffers remain bounded by the existing `max_source_bytes` limits.

## Layout contract

- `Nv21`: contiguous 8-bit 4:2:0, Y plane followed by interleaved VU, even width and height,
  even `row_stride >= width`, chroma stride equal to Y stride.
- `Yuy2`: packed 8-bit 4:2:2 byte groups `Y0 U Y1 V`, even width,
  `row_stride >= width * 2`.
- `Uyvy`: packed 8-bit 4:2:2 byte groups `U Y0 V Y1`, even width,
  `row_stride >= width * 2`.
- Conversion continues to use the existing BT.601 limited-range integer transform so prior NV12
  and I420 output remains byte-for-byte stable.

## Boundaries and ownership

The caller owns the source bytes. CPU processing borrows Host memory for the duration of the call.
CUDA processing accepts Host or CUDA Device memory, packs padded Host input into the existing
bounded pinned/device staging buffers, and retains no caller pointer after the synchronous return.
No new allocation policy, queue, fallback, or cross-thread state is introduced.

## Compatibility and migration

The new `PixelFormat` enumerators are appended, so existing aggregate `ImageView` initializers and
field layout remain unchanged. Exhaustive adapters in TensorRT-YOLO, Face Applications and Vision
Models must be updated together. Consumers must rebuild against the updated headers; existing
formats retain their prior validation and conversion behavior.

## Failure semantics

Odd widths for packed 4:2:2, odd NV21 dimensions, undersized stride/capacity, unsupported memory
kind, and configured byte-limit violations fail at the existing validation boundary. There is no
implicit conversion fallback.

## Verification

Literal fixtures verify CPU byte output, padded Host staging, CUDA/CPU tensor parity, TensorRT-YOLO
planning, and public application acceptance. Relevant installed-consumer and TensorRT face suites
must remain green. Rollback consists of removing the appended enumerators and their switch branches;
no persisted data migration is involved.
