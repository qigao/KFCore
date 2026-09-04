add_library(kfcore_face_applications_cpu SHARED
  src/composer.cpp
  src/cpu.cpp
  src/geometry.cpp)
target_sources(kfcore_face_applications_cpu PRIVATE src/preprocess.cpp)
target_include_directories(kfcore_face_applications_cpu
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_face_applications_cpu PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_applications_cpu PUBLIC
  KFCore::image_processor_cpu
  KFCore::face_model_core
  KFCore::face_models_cpu)
set_target_properties(kfcore_face_applications_cpu PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
  FOLDER "applications/onnx_cpu"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_face_applications_cpu
  ALIAS KFCore::face_applications_cpu
  EXPORT_NAME face_applications_cpu
  EXPORT_SET KFCoreTargets
  FOLDER "applications/onnx_cpu")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
