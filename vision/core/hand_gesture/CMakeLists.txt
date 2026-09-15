add_library(kfcore_hand_gesture STATIC
  src/feature_encoder.cpp)
target_include_directories(kfcore_hand_gesture
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_hand_gesture PUBLIC cxx_std_17)
target_link_libraries(kfcore_hand_gesture
  PUBLIC KFCore::hand_model_core)
set_target_properties(kfcore_hand_gesture PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision/core/hand_gesture")
cmake_config_target(kfcore_hand_gesture
  ALIAS KFCore::hand_gesture
  EXPORT_NAME hand_gesture
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/hand_gesture")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
