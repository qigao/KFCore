add_library(kfcore_scene_interaction STATIC
  src/scene_features.cpp
  src/interaction.cpp
  src/pipeline.cpp
  src/latency_report.cpp)
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

if(KFCORE_BUILD_RELEASED_SCENE_BEHAVIOR_QUALIFICATION)
  add_executable(kfcore_released_scene_behavior_latency
    tests/released_scene_behavior_latency.cpp)
  target_compile_features(
    kfcore_released_scene_behavior_latency PRIVATE cxx_std_17)
  target_link_libraries(
    kfcore_released_scene_behavior_latency PRIVATE
      KFCore::scene_interaction)
  set_target_properties(
    kfcore_released_scene_behavior_latency PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "vision/interaction/qualification")
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
