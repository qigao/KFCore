add_library(kfcore_scene_graph_pipeline STATIC
  src/scene_graph.cpp
  src/scene_graph_detail.cpp)
target_include_directories(kfcore_scene_graph_pipeline
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_scene_graph_pipeline PUBLIC cxx_std_17)
target_link_libraries(kfcore_scene_graph_pipeline
  PUBLIC
    KFCore::yolo_runtime
    KFCore::relation_core)
set_target_properties(kfcore_scene_graph_pipeline PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision/pipelines/scene_graph")
cmake_config_target(kfcore_scene_graph_pipeline
  ALIAS KFCore::scene_graph_pipeline
  EXPORT_NAME scene_graph_pipeline
  EXPORT_SET KFCoreTargets
  FOLDER "vision/pipelines/scene_graph")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
