add_library(kfcore_hand_interaction STATIC
  src/gesture_graph.cpp
  src/hand_interaction.cpp
  src/hand_identity.cpp
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
    KFCore::vision_model_core
    KFCore::thig)
set_target_properties(kfcore_hand_interaction PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "hand_interaction")
cmake_config_target(kfcore_hand_interaction
  ALIAS KFCore::hand_interaction
  EXPORT_NAME hand_interaction
  EXPORT_SET KFCoreTargets
  FOLDER "hand_interaction")

if(BUILD_TESTS)
  add_executable(test_hand_primitive_extractor
    tests/test_primitive_extractor.cpp)
  target_include_directories(test_hand_primitive_extractor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_hand_primitive_extractor PRIVATE
    KFCore::hand_interaction
    TurboUtils::TinyTest)
  target_compile_features(test_hand_primitive_extractor PRIVATE cxx_std_17)
  add_test(NAME test_hand_primitive_extractor
    COMMAND test_hand_primitive_extractor)

  add_executable(test_hand_interaction
    tests/test_hand_interaction.cpp)
  target_include_directories(test_hand_interaction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_hand_interaction PRIVATE
    KFCore::hand_interaction
    TurboUtils::TinyTest)
  target_compile_features(test_hand_interaction PRIVATE cxx_std_17)
  add_test(NAME test_hand_interaction COMMAND test_hand_interaction)
endif()

cmake_install_headers(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT Development)
