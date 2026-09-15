# Manual build of the backend-neutral runtime

KFCore builds the backend-neutral runtime architecture with a static SDK and runtime-loaded execution plugins. Model, vision, preprocessing, tracking, runtime-core, and pipeline targets are static archives; only backend plugin boundaries are shared libraries.

Static libraries install under `${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}`. Runtime plugins install to `${CMAKE_INSTALL_PREFIX}/${KFCORE_INSTALL_PLUGINDIR}`. `KFCORE_INSTALL_PLUGINDIR` defaults to `plugins`.

## Static SDK layout

The normal KFCore targets are static, including:

- `KFCore::kfcore`
- `KFCore::trackers`
- `KFCore::runtime_core`
- `KFCore::image_processor_core`
- `KFCore::image_processor_cpu`
- `KFCore::image_processor_cuda` when CUDA compute support is enabled
- `KFCore::sift` and `KFCore::sift_popsift`
- `KFCore::yolo_core` and `KFCore::yolo_runtime`
- `KFCore::pose_core`
- `KFCore::face_model_core` and `KFCore::face_model_runtime`
- `KFCore::hand_model_core` and `KFCore::hand_model_runtime`
- `KFCore::whole_body_pipeline`

Internal archives such as miniblas and PopSift are also installed/exported only to close static-link dependencies; they are named as internal CMake targets and are not public API surfaces.

A CUDA-enabled static SDK needs `CUDAToolkit` available when a downstream CMake project consumes CUDA-backed KFCore targets. TensorRT and ONNX Runtime themselves remain runtime plugin dependencies and are not part of the normal KFCore consumer link interface.

## TensorRT runtime plugin

```powershell
cmake -S . -B build -G Ninja `
  -DKFCORE_ENABLE_CUDA=ON `
  -DKFCORE_ENABLE_TENSORRT=ON `
  -DKFCORE_ENABLE_ONNXRUNTIME=OFF `
  -DBUILD_TESTS=OFF `
  -DBUILD_EXAMPLES=OFF
cmake --build build
cmake --install build
```

The important KFCore/TensorRT deployment DLLs on Windows are:

```text
plugins/kfcore_backend_tensorrt.dll
plugins/nvinfer_<major>.dll
plugins/nvinfer_plugin_<major>.dll
```

There is no separate `kfcore_runtime_tensorrt.dll`; the KFCore TensorRT runtime implementation is statically linked into `kfcore_backend_tensorrt.dll`.

TensorRT 8.6 packages that use unversioned DLL names are supported as well. During a Windows build KFCore copies the validated `nvinfer` and `nvinfer_plugin` runtime DLLs from the configured `TENSORRT_ROOT` beside `kfcore_backend_tensorrt.dll`; `cmake --install` installs the same files into the controlled plugin directory. KFCore CUDA code links `CUDA::cudart_static`, so KFCore DLLs themselves do not rely on `cudart` being found through `PATH`.

KFCore V1 uses the standard/full TensorRT runtime for ordinary prebuilt engines. Lean/dispatch runtimes are intentionally not deployed because they correspond to TensorRT version-compatible engines, which are outside the Model Package V1 trust model. A TensorRT engine that uses optional external CUDA, cuBLAS, cuDNN, or custom-plugin dependencies must still deploy those dependencies according to that engine's build contract; KFCore does not weaken its DLL search policy to discover them from the process current directory or an arbitrary system path.

On Linux the backend plugin remains the only KFCore shared execution boundary. TensorRT/CUDA shared-library deployment remains the platform package manager/runtime-linker responsibility.

## ONNX Runtime CPU plugin

```powershell
cmake -S . -B build-ort -G Ninja `
  -DKFCORE_ENABLE_CUDA=OFF `
  -DKFCORE_ENABLE_TENSORRT=OFF `
  -DKFCORE_ENABLE_ONNXRUNTIME=ON `
  -DKFCORE_ONNXRUNTIME_ENABLE_CUDA=OFF `
  -DBUILD_TESTS=OFF `
  -DBUILD_EXAMPLES=OFF
cmake --build build-ort
cmake --install build-ort
```

This installs `plugins/kfcore_backend_onnxruntime.dll`, which exposes the `cpu` device.

At runtime set `ONNXRUNTIME_ROOT` to the absolute ONNX Runtime SDK/runtime root used by the build. KFCore resolves the configured ONNX Runtime library beneath that root to a canonical regular file. On Windows the main runtime DLL is loaded with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`, so provider dependencies must be deployed in the SDK/runtime layout below that root rather than relying on the process current directory.

## ONNX Runtime CPU + CUDA providers

```powershell
cmake -S . -B build-ort-cuda -G Ninja `
  -DKFCORE_ENABLE_CUDA=OFF `
  -DKFCORE_ENABLE_TENSORRT=OFF `
  -DKFCORE_ENABLE_ONNXRUNTIME=ON `
  -DKFCORE_ONNXRUNTIME_ENABLE_CUDA=ON `
  -DBUILD_TESTS=OFF `
  -DBUILD_EXAMPLES=OFF
cmake --build build-ort-cuda
cmake --install build-ort-cuda
```

The single ONNX Runtime plugin always exposes `cpu`; when the CUDA provider is built and a usable CUDA device is present it additionally exposes `cuda:N` devices. `KFCORE_ENABLE_CUDA` controls KFCore CUDA image/compute components and is independent of the ORT CUDA provider. KFCore links its direct CUDA runtime use statically; ONNX Runtime CUDA-provider dependencies remain part of the selected ONNX Runtime distribution and must be deployed according to that distribution's contract.

## TensorRT + ONNX Runtime in one process

```powershell
cmake -S . -B build-all -G Ninja `
  -DKFCORE_ENABLE_CUDA=ON `
  -DKFCORE_ENABLE_TENSORRT=ON `
  -DKFCORE_ENABLE_ONNXRUNTIME=ON `
  -DKFCORE_ONNXRUNTIME_ENABLE_CUDA=OFF `
  -DBUILD_TESTS=OFF `
  -DBUILD_EXAMPLES=OFF
cmake --build build-all
cmake --install build-all
```

This configuration provides TensorRT on CUDA and ONNX Runtime on CPU in the same process. Enable `KFCORE_ONNXRUNTIME_ENABLE_CUDA=ON` as well when the installed ORT SDK includes CUDA EP and the application explicitly wants ORT CUDA devices.

For explicit loading from an installed prefix:

```cpp
kfcore::runtime::Runtime runtime;
(void)runtime.load_backend("<prefix>/plugins/kfcore_backend_tensorrt.dll");
(void)runtime.load_backend("<prefix>/plugins/kfcore_backend_onnxruntime.dll");
```

For controlled directory discovery:

```cpp
(void)runtime.load_backends_from("<prefix>/plugins");
```

Directory discovery is non-recursive, sorted, and only accepts KFCore backend library names. It never scans `PATH`, `System32`, the current working directory, or other system locations.

To use another controlled install location, configure it explicitly:

```powershell
-DKFCORE_INSTALL_PLUGINDIR=lib/kfcore/plugins
```

## Backend metadata

Before authoring or checking a TensorRT Model Package, inspect the installed plugin directly:

```powershell
<prefix>\bin\kfmodel.exe backend `
  <prefix>\plugins\kfcore_backend_tensorrt.dll
```

The command prints the backend ID/name, current platform, exact TensorRT `major.minor.patch.build`, and the available CUDA device IDs, names, and compute capabilities. Those values are the runtime-side facts used by the resolver; `hardware_compatibility` remains an assertion about how the engine itself was built.

For ONNX Runtime the same command lists the provider-backed devices exposed by that plugin. Set `ONNXRUNTIME_ROOT` first because loading the plugin initializes its explicitly configured runtime library.

## Model package check

Before running an application:

```powershell
build\bin\kfmodel.exe inspect  C:\models\rtmw-l-384x288
build\bin\kfmodel.exe validate C:\models\rtmw-l-384x288
```

For a TensorRT artifact, validation requires ONNX source provenance, exact TensorRT `major.minor.patch.build`, target platform, hardware compatibility mode, CUDA compute capability, and precision metadata. `exact-device` packages also require the CUDA device name. `same-compute-capability` is accepted only for TensorRT 10.9 or newer; TensorRT 8.6/Pascal packages use `exact-device`. TensorRT version-compatible plans are outside the V1 trust model.

## Installed runtime probe

`kfmodel probe` verifies the runtime-loaded execution path without requiring model-specific input data. It loads one explicit backend plugin, constructs an exact execution policy from the backend's reported ID and the requested device, resolves the package, loads/deserializes the selected artifact, creates an execution context, and prints the discovered tensor contract.

For an installed TensorRT package on Windows:

```powershell
<prefix>\bin\kfmodel.exe probe `
  C:\models\rtmw-l-384x288 `
  <prefix>\plugins\kfcore_backend_tensorrt.dll `
  cuda:0
```

For an installed ONNX Runtime CPU package:

```powershell
$env:ONNXRUNTIME_ROOT = "C:\path\to\onnxruntime"
<prefix>\bin\kfmodel.exe probe `
  C:\models\rtmw-l-384x288 `
  <prefix>\plugins\kfcore_backend_onnxruntime.dll `
  cpu
```

A successful probe ends with:

```text
runtime probe ready: <model-id>
```

The probe deliberately uses an explicit plugin path rather than loading every plugin in a directory. This keeps a TensorRT smoke independent from an unconfigured ONNX Runtime installation, and vice versa, while still exercising the same secure dynamic loader and resolver used by applications.
