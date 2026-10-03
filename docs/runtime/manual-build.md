# Building and consuming KFCore

KFCore's default configuration is backend-neutral. CUDA, TensorRT, and ONNX
Runtime execution capabilities are opt-in.

## Default SDK build

With Salts and SaltsUtils discoverable by the active toolchain or
`CMAKE_PREFIX_PATH`:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix <install-prefix>
```

The default values are:

```text
KFCORE_ENABLE_CUDA=OFF
KFCORE_ENABLE_TENSORRT=OFF
KFCORE_ENABLE_ONNXRUNTIME=OFF
```

`SALTS_ROOT` and `SALTS_UTILS_ROOT` may be used as developer hints, but
installed KFCore packages do not require those environment variables.

## Execution capabilities

Tracked reusable capability presets live in
`presets/RuntimeCapabilities.json`:

- `runtime-ort-cpu`;
- `runtime-ort-cuda`;
- `runtime-tensorrt-ort-cpu`;
- `runtime-tensorrt-only`.

The tracked `CMakeUserPresets.json` defines the project's configure, build,
test, and install entry presets, including local SDK locations.
`CMakeUserPresets.json.example` shows smaller profiles for other environments.

Equivalent command-line switches can also be used directly. For example,
TensorRT requires both:

```text
KFCORE_ENABLE_CUDA=ON
KFCORE_ENABLE_TENSORRT=ON
```

ONNX Runtime CPU execution requires:

```text
KFCORE_ENABLE_ONNXRUNTIME=ON
KFCORE_ONNXRUNTIME_ENABLE_CUDA=OFF
```

When CUDA-backed static KFCore targets are enabled, consumers also need a
discoverable `CUDAToolkit`.

## Consuming the install tree

KFCore exports normal CMake package targets:

```cmake
find_package(KFCore CONFIG REQUIRED)

target_link_libraries(app PRIVATE
    KFCore::runtime_core
    KFCore::kalman)
```

Dependency lookup uses standard CMake package discovery, so a toolchain file or
`CMAKE_PREFIX_PATH` is sufficient when Salts and SaltsUtils are installed.

## Runtime plugins

Execution backends are dynamic plugins. The default install directory is:

```text
<install-prefix>/plugins
```

Load a specific plugin explicitly:

```cpp
kfcore::runtime::Runtime runtime;
runtime.load_backend("<install-prefix>/plugins/kfcore_backend_tensorrt.dll");
```

Or enumerate one application-controlled directory:

```cpp
runtime.load_backends_from("<install-prefix>/plugins");
```

KFCore does not search arbitrary system DLL/shared-library paths for execution
plugins.

## Installed-package gate

The repository's package contract generates an install-tree
`KFCoreConfig.cmake`, clears KFCore-specific dependency-root environment
variables, and configures a real C/C++ consumer using only normal package-prefix
discovery. This is the expected portability boundary for the SDK.
