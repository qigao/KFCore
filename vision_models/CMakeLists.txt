add_library(kfcore_vision_model_core SHARED
  src/decode.cpp
  src/error.cpp
  src/geometry.cpp
  src/pipeline.cpp)
target_include_directories(kfcore_vision_model_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_vision_model_core PUBLIC cxx_std_17)
target_link_libraries(kfcore_vision_model_core
  PUBLIC KFCore::image_core
  PRIVATE KFCore::trackers)
set_target_properties(kfcore_vision_model_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision_models"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_vision_model_core
  ALIAS KFCore::vision_model_core
  EXPORT_NAME vision_model_core
  EXPORT_SET KFCoreTargets
  FOLDER "vision_models")

if(BUILD_TESTS)
  add_executable(test_vision_model_geometry tests/test_geometry.cpp)
  target_include_directories(test_vision_model_geometry PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_vision_model_geometry PRIVATE
    KFCore::vision_model_core
    TurboUtils::TinyTest)
  target_compile_features(test_vision_model_geometry PRIVATE cxx_std_17)
  add_test(NAME test_vision_model_geometry COMMAND test_vision_model_geometry)

  add_executable(test_vision_model_decode tests/test_decode.cpp)
  target_include_directories(test_vision_model_decode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_vision_model_decode PRIVATE
    KFCore::vision_model_core
    TurboUtils::TinyTest)
  target_compile_features(test_vision_model_decode PRIVATE cxx_std_17)
  add_test(NAME test_vision_model_decode COMMAND test_vision_model_decode)

  add_executable(test_vision_model_hand_pipeline tests/test_hand_pipeline.cpp)
  target_include_directories(test_vision_model_hand_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_vision_model_hand_pipeline PRIVATE
    KFCore::vision_model_core
    TurboUtils::TinyTest)
  target_compile_features(test_vision_model_hand_pipeline PRIVATE cxx_std_17)
  add_test(NAME test_vision_model_hand_pipeline COMMAND test_vision_model_hand_pipeline)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
