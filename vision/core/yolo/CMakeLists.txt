add_library(kfcore_yolo_core STATIC
  src/compact_nms.cpp
  src/efficient_nms.cpp
  src/raw_yolo.cpp
  src/tracking.cpp)
target_include_directories(kfcore_yolo_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_yolo_core PUBLIC cxx_std_17)
target_link_libraries(kfcore_yolo_core
  PUBLIC KFCore::image_processor_core
  PRIVATE KFCore::trackers)
set_target_properties(kfcore_yolo_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision/core/yolo")
cmake_config_target(kfcore_yolo_core
  ALIAS KFCore::yolo_core
  EXPORT_NAME yolo_core
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/yolo")

add_library(kfcore_yolo_runtime STATIC
  src/runtime_detector.cpp
  src/runtime_detector_facade.cpp
  src/runtime_detector_image_view.cpp)
target_include_directories(kfcore_yolo_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_yolo_runtime PUBLIC cxx_std_17)
target_link_libraries(kfcore_yolo_runtime
  PUBLIC
    KFCore::yolo_core
    KFCore::runtime_core
  PRIVATE
    KFCore::image_processor_cpu)
set_target_properties(kfcore_yolo_runtime PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision/core/yolo")
cmake_config_target(kfcore_yolo_runtime
  ALIAS KFCore::yolo_runtime
  EXPORT_NAME yolo_runtime
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/yolo")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
