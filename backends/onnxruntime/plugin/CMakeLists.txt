if(WIN32)
  set(_kfcore_ort_runtime_relative "${ONNXRuntime_RUNTIME_RELATIVE_PATH}")
else()
  file(REAL_PATH "$ENV{ONNXRUNTIME_ROOT}" _kfcore_ort_root)
  file(RELATIVE_PATH _kfcore_ort_runtime_relative
    "${_kfcore_ort_root}" "${ONNXRuntime_LIBRARY}")
endif()

add_library(kfcore_backend_onnxruntime SHARED
  src/plugin.cpp)
target_compile_features(kfcore_backend_onnxruntime PRIVATE cxx_std_17)
target_include_directories(kfcore_backend_onnxruntime PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/abi/include
  ${ONNXRuntime_INCLUDE_DIR})
target_link_libraries(kfcore_backend_onnxruntime PRIVATE
  KFCore::runtime_abi
  ${CMAKE_DL_LIBS})
target_compile_definitions(kfcore_backend_onnxruntime PRIVATE
  "KFCORE_ONNX_RUNTIME_RELATIVE_PATH=\"${_kfcore_ort_runtime_relative}\"")
if(KFCORE_ONNXRUNTIME_ENABLE_CUDA)
  target_link_libraries(kfcore_backend_onnxruntime PRIVATE CUDA::cudart_static)
  target_compile_definitions(kfcore_backend_onnxruntime PRIVATE
    KFCORE_ONNX_CUDA
    KFCORE_ONNX_CUDA_ALLOW_CPU_NODES=$<BOOL:${KFCORE_ONNXRUNTIME_ALLOW_CPU_NODES}>)
endif()
set_target_properties(kfcore_backend_onnxruntime PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  OUTPUT_NAME "kfcore_backend_onnxruntime"
  FOLDER "backends/onnxruntime/plugin")
cmake_config_target(kfcore_backend_onnxruntime
  PLUGIN
  FOLDER "backends/onnxruntime/plugin")
