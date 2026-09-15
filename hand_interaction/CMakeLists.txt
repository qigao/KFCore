add_library(kfcore_hand_interaction STATIC
  src/gesture_graph.cpp
  src/hand_interaction.cpp
  src/hand_identity.cpp
  src/hand_shape_descriptor.cpp
  src/primitive_extractor.cpp)
target_include_directories(kfcore_hand_interaction
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_hand_interaction PUBLIC cxx_std_17)
target_link_libraries(kfcore_hand_interaction
  PUBLIC
    KFCore::hand_model_core
    KFCore::hand_gesture
    KFCore::thig)
set_target_properties(kfcore_hand_interaction PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision/interaction/hand")
cmake_config_target(kfcore_hand_interaction
  ALIAS KFCore::hand_interaction
  EXPORT_NAME hand_interaction
  EXPORT_SET KFCoreTargets
  FOLDER "vision/interaction/hand")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERNS "*.hpp")
