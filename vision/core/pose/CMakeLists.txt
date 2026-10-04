add_library(kfcore_pose_core STATIC
  src/rtmw.cpp
  src/schema.cpp
  src/simcc_decode.cpp
  src/heatmap_decode.cpp
  src/rtmw_preprocess.cpp
  src/pose_semantic_config.cpp)
target_include_directories(kfcore_pose_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_pose_core PUBLIC cxx_std_17)
target_link_libraries(kfcore_pose_core
  PUBLIC KFCore::runtime_core KFCore::image_processor_core
  PRIVATE KFCore::image_processor_cpu Salts::DataBind)
set_target_properties(kfcore_pose_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision/core/pose")
cmake_config_target(kfcore_pose_core
  ALIAS KFCore::pose_core
  EXPORT_NAME pose_core
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/pose")

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")

if(BUILD_TESTS OR KFCORE_BUILD_POSE_RTMW_ORT_CPU_QUALIFICATION)
  add_subdirectory(tests)
endif()
