add_library(kfcore_hand_models_cpu SHARED
  src/cpu.cpp
  src/runtime_support.cpp)
target_include_directories(kfcore_hand_models_cpu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${KFCORE_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vision/core/hand_models/src)
target_compile_features(kfcore_hand_models_cpu PUBLIC cxx_std_17)
target_link_libraries(kfcore_hand_models_cpu
  PUBLIC KFCore::hand_model_core
  PRIVATE KFCore::image_processor_cpu KFCore::runtime_onnx)
set_target_properties(kfcore_hand_models_cpu PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "backends/onnx_cpu/hand_models")
cmake_config_target(kfcore_hand_models_cpu
  ALIAS KFCore::hand_models_cpu
  EXPORT_NAME hand_models_cpu
  EXPORT_SET KFCoreTargets
  FOLDER "backends/onnx_cpu/hand_models")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
