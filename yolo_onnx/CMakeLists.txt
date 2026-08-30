add_library(kfcore_yolo_onnx SHARED src/onnx.cpp)
target_include_directories(kfcore_yolo_onnx
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_SOURCE_DIR}/yolo_core/src)
target_compile_features(kfcore_yolo_onnx PUBLIC cxx_std_17)
target_link_libraries(kfcore_yolo_onnx
  PUBLIC KFCore::yolo_core
  PRIVATE KFCore::image_processor_cpu KFCore::runtime_onnx)
set_target_properties(kfcore_yolo_onnx PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "yolo_onnx")
cmake_config_target(kfcore_yolo_onnx
  ALIAS KFCore::yolo_onnx
  EXPORT_NAME yolo_onnx
  EXPORT_SET KFCoreTargets
  FOLDER "yolo_onnx")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
