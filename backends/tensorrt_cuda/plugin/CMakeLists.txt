add_library(kfcore_backend_tensorrt SHARED
  src/plugin.cpp)
target_compile_features(kfcore_backend_tensorrt PRIVATE cxx_std_17)
target_link_libraries(kfcore_backend_tensorrt PRIVATE
  KFCore::runtime_abi
  KFCore::runtime_tensorrt
  TensorRT::nvinfer
  CUDA::cudart)
target_include_directories(kfcore_backend_tensorrt PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime/abi/include)
set_target_properties(kfcore_backend_tensorrt PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  OUTPUT_NAME "kfcore_backend_tensorrt"
  FOLDER "backends/tensorrt_cuda/plugin")
if(UNIX AND NOT APPLE)
  set_target_properties(kfcore_backend_tensorrt PROPERTIES
    INSTALL_RPATH "$ORIGIN")
endif()
cmake_config_target(kfcore_backend_tensorrt
  PLUGIN
  FOLDER "backends/tensorrt_cuda/plugin")
