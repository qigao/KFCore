# Face Applications Native Image Design

## Context

`TensorRtFaceSwapApplication` currently exposes `cv::Mat` and therefore forces every
consumer to depend on OpenCV, even though its detector and model preprocessing already run
through `KFCore::image_processor`. The camera path already produces borrowed BGR/RGB/NV12/I420
`ImageView` objects. Converting NV12 to `cv::Mat` before the application would add a full-frame
host conversion and discard the shared GPU staging path.

## Decision

- The TensorRT application accepts borrowed `kfcore::image::ImageView` inputs and returns an
  owned `kfcore::image::BgrImage`.
- Supported inputs are BGR8, RGB8, NV12 and I420 in Host or CUDA device memory. Invalid layout,
  capacity, device and resource-limit conditions fail at the image processor boundary.
- One synchronous application call owns no input memory. Each configured CUDA device stages a
  host input at most once per analyzed frame; all models on that device reuse the staged view.
- Face68, ArcFace, Age/Gender, InSwapper and GFPGAN receive model-specific CUDA NCHW tensors
  produced directly from the staged image. Models do not receive raw NV12.
- The first paste-back may sample an NV12/I420 CUDA base directly and always produces packed
  CUDA BGR8. Only the final BGR result is downloaded.
- TensorRT geometry and masks use project-owned point, affine and float-mask structures. OpenCV
  remains an examples/integration-test adapter for decoding, display and encoding only.
- The CPU application adds `ImageView` overloads. It accepts Host BGR8/RGB8/NV12/I420 and makes
  one bounded conversion to its existing owned BGR pipeline; CUDA input fails explicitly.

## Ownership and state

The caller owns each input buffer for the complete synchronous call. `ImageView` is immutable and
borrowed. `CudaImageProcessor` owns reusable per-device staging, tensor, mask and composition
buffers. `BgrImage` owns the returned host pixels. Application instances remain non-reentrant;
separate instances are required for concurrent tasks.

## Compatibility

Removing the TensorRT `cv::Mat` overloads is an intentional source-level API break authorized for
this development line. Demo and integration code wrap decoded `cv::Mat` storage in `ImageView` and
wrap returned `BgrImage` storage in `cv::Mat` only while displaying or encoding. CPU `BgrImage`
overloads remain available.

## Verification

- CUDA unit tests prove NV12 and I420 can be used as affine-composition bases and yield literal
  BGR pixels.
- Geometry/mask tests prove the OpenCV-free replacements retain transform and mask behavior.
- API/installed-consumer tests compile against `ImageView` input and owned `BgrImage` output.
- Opt-in TensorRT and CPU integration tests run real models with YUV input when assets are
  available.
- The relevant preset build and CTest suite must pass before completion is claimed.
