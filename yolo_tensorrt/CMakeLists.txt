add_library(kfcore_yolo_tensorrt SHARED
  src/cuda_buffer.cpp
  src/detector_helpers.cpp
  src/detector.cpp
  src/engine_contract.cpp
  src/engine_file.cpp
  src/engine.cpp)
target_include_directories(kfcore_yolo_tensorrt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${KFCORE_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/yolo_core/src)
target_link_libraries(kfcore_yolo_tensorrt
  PUBLIC KFCore::yolo_core
  PRIVATE
    KFCore::image_processor_cuda
    TensorRT::nvinfer
    TensorRT::nvinfer_plugin
    CUDA::cudart)
target_compile_features(kfcore_yolo_tensorrt PUBLIC cxx_std_17)
set_target_properties(kfcore_yolo_tensorrt PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "yolo_tensorrt")
cmake_config_target(kfcore_yolo_tensorrt
  ALIAS KFCore::yolo_tensorrt
  EXPORT_NAME yolo_tensorrt
  EXPORT_SET KFCoreTargets
  FOLDER "yolo_tensorrt")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
