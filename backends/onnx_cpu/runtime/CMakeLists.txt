add_library(kfcore_runtime_onnx SHARED
  src/error.cpp
  src/runtime.cpp)
target_include_directories(kfcore_runtime_onnx PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_runtime_onnx PUBLIC cxx_std_17)
target_link_libraries(kfcore_runtime_onnx PRIVATE ONNXRuntime::ONNXRuntime)
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
