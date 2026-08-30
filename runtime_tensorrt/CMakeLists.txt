find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)

add_library(kfcore_runtime_tensorrt SHARED
  src/cuda_buffer.cpp
  src/cuda_device.cpp
  src/engine.cpp
  src/engine_file.cpp
  src/error.cpp
  src/executor.cpp
  src/tensor_validation.cpp)
target_include_directories(kfcore_runtime_tensorrt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_runtime_tensorrt PUBLIC cxx_std_17)
target_link_libraries(kfcore_runtime_tensorrt PRIVATE
  CUDA::cudart
  TensorRT::nvinfer
  TensorRT::nvinfer_plugin)
set_target_properties(kfcore_runtime_tensorrt PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "runtime_tensorrt"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_runtime_tensorrt
  ALIAS KFCore::runtime_tensorrt
  EXPORT_NAME runtime_tensorrt
  EXPORT_SET KFCoreTargets
  FOLDER "runtime_tensorrt")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
