add_library(kfcore_face_models_cuda SHARED
  src/age_gender.cpp
  src/arcface.cpp
  src/contracts.cpp
  src/face68.cpp
  src/facemesh.cpp
  src/gfpgan.cpp
  src/inswapper.cpp)
target_include_directories(kfcore_face_models_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${KFCORE_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vision/core/face_models/src)
target_compile_features(kfcore_face_models_cuda PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_models_cuda
  PUBLIC KFCore::face_model_core KFCore::runtime_tensorrt
  PRIVATE KFCore::image_processor_cuda KFCore::yolo_tensorrt Salts::Core)
set_target_properties(kfcore_face_models_cuda PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "backends/tensorrt_cuda/face_models")
cmake_config_target(kfcore_face_models_cuda
  ALIAS KFCore::face_models_cuda
  EXPORT_NAME face_models_cuda
  EXPORT_SET KFCoreTargets
  FOLDER "backends/tensorrt_cuda/face_models")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
