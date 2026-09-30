add_library(kfcore_scene_interaction STATIC
  src/scene_features.cpp
  src/interaction.cpp
  src/pipeline.cpp)
target_include_directories(kfcore_scene_interaction
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_scene_interaction PUBLIC cxx_std_17)
target_link_libraries(kfcore_scene_interaction
  PUBLIC
    KFCore::scene_graph_pipeline
    KFCore::esn)
set_target_properties(kfcore_scene_interaction PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision/interaction")
cmake_config_target(kfcore_scene_interaction
  ALIAS KFCore::scene_interaction
  EXPORT_NAME scene_interaction
  EXPORT_SET KFCoreTargets
  FOLDER "vision/interaction")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
