add_library(kfcore_hand_model_core SHARED
  src/decode.cpp
  src/error.cpp
  src/geometry.cpp
  src/hand_appearance.cpp
  src/pipeline.cpp)
target_include_directories(kfcore_hand_model_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_hand_model_core PUBLIC cxx_std_17)
target_link_libraries(kfcore_hand_model_core
  PUBLIC KFCore::image_processor_core
  PRIVATE KFCore::trackers)
set_target_properties(kfcore_hand_model_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "vision/core/hand_models")
cmake_config_target(kfcore_hand_model_core
  ALIAS KFCore::hand_model_core
  EXPORT_NAME hand_model_core
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/hand_models")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
