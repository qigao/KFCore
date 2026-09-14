add_library(kfcore_whole_body_pipeline SHARED
  src/whole_body.cpp)
target_include_directories(kfcore_whole_body_pipeline
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_whole_body_pipeline PUBLIC cxx_std_17)
target_link_libraries(kfcore_whole_body_pipeline
  PUBLIC
    KFCore::image_processor_core
    KFCore::yolo_runtime
    KFCore::pose_core)
set_target_properties(kfcore_whole_body_pipeline PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "vision/pipelines/whole_body")
cmake_config_target(kfcore_whole_body_pipeline
  ALIAS KFCore::whole_body_pipeline
  EXPORT_NAME whole_body_pipeline
  EXPORT_SET KFCoreTargets
  FOLDER "vision/pipelines/whole_body")

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
