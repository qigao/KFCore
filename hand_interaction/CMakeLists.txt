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

if(KFCORE_BUILD_HAND_INTERACTION_EXAMPLES)
  if(NOT TARGET TurboUtils::Capture)
    message(FATAL_ERROR
      "KFCORE_BUILD_HAND_INTERACTION_EXAMPLES requires a TurboUtils SDK "
      "installed with the Capture component")
  endif()
  find_package(OpenCVLite REQUIRED COMPONENTS core imgproc highgui)
endif()

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

  add_executable(test_hand_interaction_demo_cli
    tests/test_hand_interaction_demo_cli.cpp
    examples/hand_interaction_demo_cli.cpp)
  target_include_directories(test_hand_interaction_demo_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples)
  target_link_libraries(test_hand_interaction_demo_cli PRIVATE
    TurboUtils::Core
    TurboUtils::TinyTest)
  target_compile_features(test_hand_interaction_demo_cli PRIVATE cxx_std_17)
  add_test(NAME test_hand_interaction_demo_cli
    COMMAND test_hand_interaction_demo_cli)

  add_executable(test_hand_interaction_demo_capture
    tests/test_hand_interaction_demo_capture.cpp
    examples/hand_interaction_demo_capture.cpp)
  target_include_directories(test_hand_interaction_demo_capture PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples)
  target_link_libraries(test_hand_interaction_demo_capture PRIVATE
    TurboUtils::Capture
    TurboUtils::Core
    TurboUtils::TinyTest)
  target_compile_features(test_hand_interaction_demo_capture PRIVATE cxx_std_17)
  add_test(NAME test_hand_interaction_demo_capture
    COMMAND test_hand_interaction_demo_capture)

  if(KFCORE_BUILD_HAND_INTERACTION_EXAMPLES)
    add_executable(test_hand_interaction_demo_ui
      tests/test_hand_interaction_demo_ui.cpp
      examples/hand_interaction_demo_capture.cpp
      examples/hand_interaction_demo_frame.cpp
      examples/hand_interaction_demo_ui.cpp)
    target_include_directories(test_hand_interaction_demo_ui PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    target_link_libraries(test_hand_interaction_demo_ui PRIVATE
      KFCore::hand_interaction
      OpenCVLite::core
      OpenCVLite::imgproc
      TurboUtils::Capture
      TurboUtils::Core
      TurboUtils::TinyTest)
    target_compile_features(test_hand_interaction_demo_ui PRIVATE cxx_std_17)
    add_test(NAME test_hand_interaction_demo_ui
      COMMAND test_hand_interaction_demo_ui)
  endif()
endif()

cmake_install_headers(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERNS "*.hpp")
