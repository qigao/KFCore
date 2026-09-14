# Manual build of the backend-neutral runtime

The default build now targets the backend-neutral architecture. Runtime-specific model wrappers are excluded unless explicitly requested.

## TensorRT runtime plugin

```powershell
cmake -S . -B build -G Ninja `
  -DKFCORE_ENABLE_CUDA=ON `
  -DKFCORE_ENABLE_TENSORRT=ON `
  -DKFCORE_ENABLE_ONNXRUNTIME=OFF `
  -DKFCORE_ENABLE_LEGACY_MODEL_BACKENDS=OFF `
  -DBUILD_TESTS=OFF `
  -DBUILD_EXAMPLES=OFF
cmake --build build
```

The important deployment DLLs on Windows are:

```text
kfcore_backend_tensorrt.dll
kfcore_runtime_tensorrt.dll
```

They are installed together in the KFCore plugin/runtime destination. On Linux the TensorRT backend plugin has an `$ORIGIN` install rpath so the internal KFCore TensorRT runtime library can live beside it.

## ONNX Runtime CPU plugin

```powershell
cmake -S . -B build-ort -G Ninja `
  -DKFCORE_ENABLE_CUDA=OFF `
  -DKFCORE_ENABLE_TENSORRT=OFF `
  -DKFCORE_ENABLE_ONNXRUNTIME=ON `
  -DKFCORE_ONNXRUNTIME_ENABLE_CUDA=OFF `
  -DKFCORE_ENABLE_LEGACY_MODEL_BACKENDS=OFF `
  -DBUILD_TESTS=OFF `
  -DBUILD_EXAMPLES=OFF
cmake --build build-ort
```

This produces `kfcore_backend_onnxruntime.dll`, which exposes the `cpu` device.

At runtime set `ONNXRUNTIME_ROOT` to the absolute ONNX Runtime SDK/runtime root used by the build. KFCore resolves the configured ONNX Runtime library beneath that root to a canonical regular file. On Windows the main runtime DLL is loaded with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`, so the ONNX Runtime DLL and its provider dependencies should be deployed in the SDK/runtime layout expected below that root rather than relying on the process current directory.

## ONNX Runtime CPU + CUDA providers

```powershell
cmake -S . -B build-ort-cuda -G Ninja `
  -DKFCORE_ENABLE_CUDA=ON `
  -DKFCORE_ENABLE_TENSORRT=OFF `
  -DKFCORE_ENABLE_ONNXRUNTIME=ON `
  -DKFCORE_ONNXRUNTIME_ENABLE_CUDA=ON `
  -DKFCORE_ENABLE_LEGACY_MODEL_BACKENDS=OFF `
  -DBUILD_TESTS=OFF `
  -DBUILD_EXAMPLES=OFF
cmake --build build-ort-cuda
```

The single ONNX Runtime plugin always exposes `cpu`; when the CUDA provider is built and a usable CUDA device is present it additionally exposes `cuda:N` devices. A machine without a usable CUDA device/driver does not disable the CPU provider.

## TensorRT + ONNX Runtime in one process

Build both execution plugins in one configuration:

```powershell
cmake -S . -B build-all -G Ninja `
  -DKFCORE_ENABLE_CUDA=ON `
  -DKFCORE_ENABLE_TENSORRT=ON `
  -DKFCORE_ENABLE_ONNXRUNTIME=ON `
  -DKFCORE_ONNXRUNTIME_ENABLE_CUDA=ON `
  -DKFCORE_ENABLE_LEGACY_MODEL_BACKENDS=OFF `
  -DBUILD_TESTS=OFF `
  -DBUILD_EXAMPLES=OFF
cmake --build build-all
```

Then load both DLLs through `kfcore::runtime::Runtime` and select execution per model instance with `ExecutionPolicy`.

For explicit loading:

```cpp
kfcore::runtime::Runtime runtime;
(void)runtime.load_backend("plugins/kfcore_backend_tensorrt.dll");
(void)runtime.load_backend("plugins/kfcore_backend_onnxruntime.dll");
```

For controlled directory discovery, point KFCore at one application-owned directory:

```cpp
(void)runtime.load_backends_from("plugins");
```

Directory discovery is non-recursive, sorted, and only accepts KFCore backend library names. It never scans `PATH`, `System32`, the current working directory, or other system locations.

## Legacy wrappers

The old runtime-specific YOLO/face/hand wrappers and old backend-specific applications are no longer part of the default architecture. They can temporarily be built with:

```text
-DKFCORE_ENABLE_LEGACY_MODEL_BACKENDS=ON
```

This option is intended only for migration of remaining consumers; new code should use runtime plugins plus typed model APIs.

## Model package check

Before running an application:

```powershell
build\bin\kfmodel.exe inspect  C:\models\rtmw-l-384x288
build\bin\kfmodel.exe validate C:\models\rtmw-l-384x288
```

For a TensorRT artifact, validation requires ONNX source provenance, TensorRT major version, exact CUDA compute capability, and precision metadata.
