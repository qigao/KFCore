# KFCore

KFCore is a backend-neutral C/C++ inference and vision runtime.

## Runtime architecture

Logical models are independent from execution backends. Applications use typed model APIs such as YOLO and RTMW, while execution is selected per model through runtime-loaded backend plugins.

KFCore uses a **static SDK + dynamic backend plugin** layout. Normal KFCore libraries are installed as static archives; the only KFCore DLL/shared-library boundaries are execution plugins:

- `kfcore_backend_tensorrt` — TensorRT execution on CUDA devices.
- `kfcore_backend_onnxruntime` — ONNX Runtime execution with CPU and optional CUDA providers.

TensorRT's KFCore runtime implementation is statically linked into the TensorRT backend plugin. Core/image/YOLO/pose/face/hand/tracking/SIFT/pipeline libraries do not require separate KFCore DLLs.

CPU/CUDA are execution devices/providers, not separate model APIs. TensorRT engines are derived artifacts selected from a Model Package using exact TensorRT runtime, platform, and declared hardware-compatibility metadata.

See:

- `docs/runtime/model-package-v1.md`
- `docs/runtime/manual-build.md`
- `docs/superpowers/specs/2026-09-14-backend-neutral-runtime-plugin-abi-design.md`

## Runtime loading

KFCore execution plugins install to `${CMAKE_INSTALL_PREFIX}/${KFCORE_INSTALL_PLUGINDIR}`. The default `KFCORE_INSTALL_PLUGINDIR` is `plugins`.

```cpp
kfcore::runtime::Runtime runtime;
(void)runtime.load_backend("<prefix>/plugins/kfcore_backend_tensorrt.dll");
(void)runtime.load_backend("<prefix>/plugins/kfcore_backend_onnxruntime.dll");
```

Or load from one explicitly controlled plugin directory:

```cpp
(void)runtime.load_backends_from("<prefix>/plugins");
```

KFCore does not scan system DLL paths for execution plugins.

## Typed models

```cpp
const auto yolo_package =
    kfcore::runtime::ModelPackage::load("models/yolo-person");
const auto rtmw_package =
    kfcore::runtime::ModelPackage::load("models/rtmw-l-384x288");

const auto yolo_policy = kfcore::runtime::ExecutionPolicy::ordered({
    {"tensorrt", "cuda:0"},
    {"onnxruntime", "cpu"},
});

const auto pose_policy = kfcore::runtime::ExecutionPolicy::ordered({
    {"tensorrt", "cuda:0"},
    {"onnxruntime", "cpu"},
});

auto detector = kfcore::yolo::YoloDetector::load(
    runtime, yolo_package, yolo_policy);
auto pose = kfcore::pose::Rtmw::load(
    runtime, rtmw_package, pose_policy);

auto pipeline = kfcore::pipelines::WholeBodyPipeline::create(
    std::move(detector), std::move(pose));
```

YOLO packages use only the canonical `model_type="yolo-detection"`. Every YOLO artifact explicitly declares one of the V1 semantic flavors `raw-yolo`, `compact-nms`, or `efficient-nms`; decoder flavor is never inferred from the selected backend. Dynamic compact-NMS `[1,-1,6]` output uses plugin ABI v1.2 and preserves the actual detection count, including the valid `N=0` case.

## Build switches

The execution-capability switches are:

```text
KFCORE_ENABLE_CUDA
KFCORE_ENABLE_TENSORRT
KFCORE_ENABLE_ONNXRUNTIME
KFCORE_ONNXRUNTIME_ENABLE_CUDA
KFCORE_ONNXRUNTIME_ALLOW_CPU_NODES
```

Plugin deployment uses:

```text
KFCORE_INSTALL_PLUGINDIR=plugins
```

Portable execution-capability presets are tracked in `presets/RuntimeCapabilities.json` and included by `CMakePresets.json`. Keep machine-specific paths in an ignored `CMakeUserPresets.json`; `CMakeUserPresets.json.example` is a starting point for local overrides.

A CUDA-enabled installed static SDK requires CMake consumers to have `CUDAToolkit` available for CUDA-backed static targets. TensorRT and ONNX Runtime remain runtime plugin dependencies rather than normal KFCore link dependencies.

## Model packages and runtime probe

Use `kfmodel` to inspect the installed execution backend metadata needed by a Model Package:

```text
kfmodel backend <backend-plugin>
```

For TensorRT this prints the exact execution runtime version plus CUDA device IDs, names, and compute capabilities.

Inspect and validate model packages with:

```text
kfmodel inspect <model-package-directory>
kfmodel validate <model-package-directory>
```

Use `probe` after build/install to exercise the actual runtime-loaded backend path through model resolution and execution-context creation:

```text
kfmodel probe <model-package-directory> <backend-plugin> <device-id>
```

For example:

```text
kfmodel backend <prefix>\plugins\kfcore_backend_tensorrt.dll
kfmodel probe C:\models\rtmw-l-384x288 <prefix>\plugins\kfcore_backend_tensorrt.dll cuda:0
```

A successful probe prints the selected route and tensor contract, then ends with `runtime probe ready: <model-id>`.

TensorRT artifacts require ONNX source provenance, exact TensorRT `major.minor.patch.build`, target platform, hardware-compatibility metadata, CUDA compute capability, and explicit precision metadata. `same-compute-capability` is accepted only for TensorRT 10.9 or newer; TensorRT 8.6/Pascal packages use `exact-device`.
