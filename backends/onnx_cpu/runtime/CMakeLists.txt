add_library(kfcore_runtime_onnx SHARED
  src/error.cpp
  src/runtime.cpp)
target_include_directories(kfcore_runtime_onnx PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_runtime_onnx PUBLIC cxx_std_17)
target_link_libraries(kfcore_runtime_onnx PRIVATE ONNXRuntime::ONNXRuntime)
if(WIN32)
  target_compile_definitions(kfcore_runtime_onnx PRIVATE
    "KFCORE_ONNX_RUNTIME_RELATIVE_PATH=\"${ONNXRuntime_RUNTIME_RELATIVE_PATH}\"")
endif()
if(KFCORE_ONNXRUNTIME_ENABLE_CUDA)
  set(KFCORE_ONNX_CUDA_DEVICE_ID "0" CACHE STRING
      "CUDA device ordinal used only by the transitional direct ONNX Runtime wrapper")
  set(KFCORE_ONNX_CUDA_MEMORY_LIMIT "2147483648" CACHE STRING
      "CUDA arena limit used only by the transitional direct ONNX Runtime wrapper")
  if(NOT KFCORE_ONNX_CUDA_DEVICE_ID MATCHES "^(0|[1-9][0-9]*)$" OR
     NOT KFCORE_ONNX_CUDA_MEMORY_LIMIT MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR
      "ONNX Runtime CUDA device must be nonnegative and memory limit must be positive integers")
  endif()
  target_compile_definitions(kfcore_runtime_onnx PRIVATE
    KFCORE_ONNX_CUDA
    KFCORE_ONNX_CUDA_ALLOW_CPU_NODES=$<BOOL:${KFCORE_ONNXRUNTIME_ALLOW_CPU_NODES}>
    "KFCORE_ONNX_CUDA_DEVICE_ID=\"${KFCORE_ONNX_CUDA_DEVICE_ID}\""
    "KFCORE_ONNX_CUDA_MEMORY_LIMIT=\"${KFCORE_ONNX_CUDA_MEMORY_LIMIT}\"")
endif()
set_target_properties(kfcore_runtime_onnx PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "backends/onnx_cpu/runtime")
cmake_config_target(kfcore_runtime_onnx
  ALIAS KFCore::runtime_onnx
  EXPORT_NAME runtime_onnx
  EXPORT_SET KFCoreTargets
  FOLDER "backends/onnx_cpu/runtime")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
