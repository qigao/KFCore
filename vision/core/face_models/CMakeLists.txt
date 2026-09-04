add_library(kfcore_face_model_core SHARED
  src/decode.cpp
  src/error.cpp
  src/facemesh_decode.cpp
  src/facemesh_geometry.cpp
  src/facemesh_pipeline.cpp
  src/inswapper_embedding.cpp)
target_include_directories(kfcore_face_model_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_face_model_core PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_model_core
  PUBLIC KFCore::image_processor_core
  PRIVATE Salts::Core)
set_target_properties(kfcore_face_model_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "vision/core/face_models")
cmake_config_target(kfcore_face_model_core
  ALIAS KFCore::face_model_core
  EXPORT_NAME face_model_core
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/face_models")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
