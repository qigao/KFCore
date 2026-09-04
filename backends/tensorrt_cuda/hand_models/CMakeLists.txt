add_library(kfcore_hand_models_cuda SHARED
  src/tensorrt.cpp
  src/tensorrt_models.cpp)
target_include_directories(kfcore_hand_models_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${KFCORE_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vision/core/hand_models/src)
target_compile_features(kfcore_hand_models_cuda PUBLIC cxx_std_17)
target_link_libraries(kfcore_hand_models_cuda
  PUBLIC KFCore::hand_model_core
  PRIVATE
    KFCore::image_processor_cuda
    KFCore::runtime_tensorrt)
set_target_properties(kfcore_hand_models_cuda PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "backends/tensorrt_cuda/hand_models")
cmake_config_target(kfcore_hand_models_cuda
  ALIAS KFCore::hand_models_cuda
  EXPORT_NAME hand_models_cuda
  EXPORT_SET KFCoreTargets
  FOLDER "backends/tensorrt_cuda/hand_models")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
