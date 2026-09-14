add_library(kfcore_yolo_core SHARED
  src/compact_nms.cpp
  src/efficient_nms.cpp
  src/raw_yolo.cpp
  src/tracking.cpp)
target_include_directories(kfcore_yolo_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${KFCORE_GENERATED_INCLUDE_DIR}>
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
  FOLDER "vision/core/yolo")
cmake_config_target(kfcore_yolo_core
  ALIAS KFCore::yolo_core
  EXPORT_NAME yolo_core
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/yolo")

add_library(kfcore_yolo_runtime SHARED
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
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "vision/core/yolo")
cmake_config_target(kfcore_yolo_runtime
  ALIAS KFCore::yolo_runtime
  EXPORT_NAME yolo_runtime
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/yolo")

if(BUILD_EXAMPLES AND KFCORE_ENABLE_LEGACY_MODEL_BACKENDS)
  find_package(Stb REQUIRED)
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
    KFCore::image_processor_cpu
    Salts::Capture)
  if(KFCORE_ENABLE_ONNXRUNTIME)
    target_link_libraries(yolov8_domain_demo PRIVATE KFCore::yolo_onnx)
  endif()
  if(KFCORE_ENABLE_TENSORRT)
    target_link_libraries(yolov8_domain_demo PRIVATE KFCore::yolo_tensorrt)
  endif()
  target_compile_definitions(yolov8_domain_demo PRIVATE
    KFCORE_DEMO_ENABLE_CUDA=$<BOOL:${KFCORE_ENABLE_CUDA}>
    KFCORE_DEMO_ENABLE_TENSORRT=$<BOOL:${KFCORE_ENABLE_TENSORRT}>
    KFCORE_DEMO_ENABLE_ONNX_CPU=$<BOOL:${KFCORE_ENABLE_ONNXRUNTIME}>)
  if(WIN32)
    target_link_libraries(yolov8_domain_demo PRIVATE user32 gdi32)
  endif()
  target_compile_features(yolov8_domain_demo PRIVATE cxx_std_17)
  cmake_config_target(yolov8_domain_demo
    NO_INSTALL FOLDER "vision/core/yolo/examples")
endif()

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
