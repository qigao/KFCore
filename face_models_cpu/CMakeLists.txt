add_library(kfcore_face_models_cpu SHARED
  src/age_gender.cpp
  src/arcface.cpp
  src/facemesh.cpp
  src/support.cpp)
target_sources(kfcore_face_models_cpu PRIVATE
  src/face68.cpp
  src/gfpgan.cpp
  src/inswapper.cpp)
target_include_directories(kfcore_face_models_cpu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${KFCORE_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/face_model_core/src)
target_compile_features(kfcore_face_models_cpu PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_models_cpu
  PUBLIC KFCore::face_model_core
  PRIVATE KFCore::runtime_onnx KFCore::image_processor_cpu)
set_target_properties(kfcore_face_models_cpu PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "face_models_cpu")
cmake_config_target(kfcore_face_models_cpu
  ALIAS KFCore::face_models_cpu
  EXPORT_NAME face_models_cpu
  EXPORT_SET KFCoreTargets
  FOLDER "face_models_cpu")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
