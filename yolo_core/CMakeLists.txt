add_library(kfcore_yolo_core SHARED
  src/compact_nms.cpp
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
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "yolo_core")
cmake_config_target(kfcore_yolo_core
  ALIAS KFCore::yolo_core
  EXPORT_NAME yolo_core
  EXPORT_SET KFCoreTargets
  FOLDER "yolo_core")

find_package(Stb REQUIRED)

if(BUILD_EXAMPLES)
  if(NOT TARGET TurboParser::Capture)
    message(FATAL_ERROR
      "BUILD_EXAMPLES requires a TurboParser SDK installed with Capture")
  endif()
  if(WIN32)
    set(_kfcore_yolo_domain_window_source
      examples/yolo_domain_window_win32.cpp)
  else()
    set(_kfcore_yolo_domain_window_source
      examples/yolo_domain_window_stub.cpp)
  endif()

  add_executable(yolov8_domain_demo
    examples/yolov8_domain_demo.cpp
    examples/yolo_domain_capture.cpp
    examples/yolo_domain_cli.cpp
    examples/yolo_domain_frame.cpp
    examples/yolo_domain_profile.cpp
    examples/yolo_domain_ui.cpp
    ${_kfcore_yolo_domain_window_source})
  target_include_directories(yolov8_domain_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    ${Stb_INCLUDE_DIR})
  target_link_libraries(yolov8_domain_demo PRIVATE
    KFCore::yolo_core
    KFCore::yolo_onnx
    KFCore::yolo_tensorrt
    KFCore::image_processor_cpu
    TurboParser::Capture)
  if(WIN32)
    target_link_libraries(yolov8_domain_demo PRIVATE user32 gdi32)
  endif()
  target_compile_features(yolov8_domain_demo PRIVATE cxx_std_17)
  cmake_config_target(yolov8_domain_demo
    NO_INSTALL FOLDER "yolo_core/examples")
endif()

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
