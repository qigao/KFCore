# Backend-Oriented Layout Design

## Goal

Reorganize the KFCore source tree so that ONNX Runtime CPU and TensorRT CUDA
implementations are grouped by backend, while preserving existing targets,
public headers, model asset formats, and package requirements.

## Scope

- Move ONNX Runtime implementation directories under `backends/onnx_cpu/`.
- Move TensorRT implementation directories under `backends/tensorrt_cuda/`.
- Move backend-neutral model and image contracts under `vision/core/` and image
  execution helpers under `vision/image/`.
- Move face-swap application implementations under `applications/`.
- Update source-tree CMake traversal paths and child-only private include paths
  that name a moved sibling implementation directory.

## Compatibility constraints

- Keep every existing `KFCore::*` alias and exported target name unchanged.
- Keep installed include paths and all public C++ names unchanged.
- Keep `.onnx` assets on the ONNX Runtime CPU path and `.engine` assets on the
  TensorRT CUDA path.
- Do not change `KFCoreConfig.cmake.in` dependency behavior in this migration.
- Keep all target link dependencies unchanged; only their source-directory
  spelling may change.
- Do not add a fallback runtime, provider selection, or new dependency.

## Target layout

```text
backends/
  onnx_cpu/{runtime,yolo,face_models,hand_models}/
  tensorrt_cuda/{runtime,yolo,face_models,hand_models}/
vision/
  core/{image_processor,yolo,face_models,hand_models}/
  image/{cpu,cuda}/
applications/{onnx_cpu,tensorrt_cuda}/
```

## Verification

The existing API and backend tests remain the behavioral contract.  Configure
with the documented Windows development preset, build the affected targets,
and run the CPU and TensorRT unit-test filters.  No new user-visible behavior
is introduced, so the migration must not change the existing test inputs or
expected outputs.
