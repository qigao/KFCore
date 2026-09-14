# Manual build of the backend-neutral runtime

KFCore builds only the backend-neutral runtime architecture. Execution SDKs are runtime-loaded backend plugins; typed model APIs do not link to ONNX Runtime or TensorRT directly.

Runtime plugins install to `${CMAKE_INSTALL_PREFIX}/${KFCORE_INSTALL_PLUGINDIR}`. `KFCORE_INSTALL_PLUGINDIR` defaults to `plugins`, so a normal install places all execution plugins and plugin-internal runtime libraries together under `<prefix>/plugins`.

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

The important deployment DLLs on Windows are:

```text
plugins/kfcore_backend_tensorrt.dll
plugins/kfcore_runtime_tensorrt.dll
```

They are installed together in the controlled KFCore plugin directory. On Linux the TensorRT backend plugin has an `$ORIGIN` install rpath so the internal KFCore TensorRT runtime library can live beside it.

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

The single ONNX Runtime plugin always exposes `cpu`; when the CUDA provider is built and a usable CUDA device is present it additionally exposes `cuda:N` devices. `KFCORE_ENABLE_CUDA` controls KFCore CUDA image/compute components and is independent of the ORT CUDA provider.

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

## Model package check

Before running an application:

```powershell
build\bin\kfmodel.exe inspect  C:\models\rtmw-l-384x288
build\bin\kfmodel.exe validate C:\models\rtmw-l-384x288
```

For a TensorRT artifact, validation requires ONNX source provenance, exact TensorRT `major.minor.patch.build`, target platform, hardware compatibility mode, CUDA compute capability, and precision metadata. `exact-device` packages also require the CUDA device name. TensorRT version-compatible plans are outside the V1 trust model.
