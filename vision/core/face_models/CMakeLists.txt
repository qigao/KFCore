add_library(kfcore_face_model_core STATIC
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
  FOLDER "vision/core/face_models")
cmake_config_target(kfcore_face_model_core
  ALIAS KFCore::face_model_core
  EXPORT_NAME face_model_core
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/face_models")

add_library(kfcore_face_model_runtime STATIC
  src/runtime.cpp
  src/prepared_models.cpp)
target_include_directories(kfcore_face_model_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_face_model_runtime PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_model_runtime
  PUBLIC
    KFCore::face_model_core
    KFCore::runtime_core
  PRIVATE
    KFCore::image_processor_cpu)
set_target_properties(kfcore_face_model_runtime PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision/core/face_models")
cmake_config_target(kfcore_face_model_runtime
  ALIAS KFCore::face_model_runtime
  EXPORT_NAME face_model_runtime
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/face_models")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
