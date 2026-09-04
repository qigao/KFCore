add_library(kfcore_image_processor_core SHARED src/error.cpp)
target_include_directories(kfcore_image_processor_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_image_processor_core PUBLIC cxx_std_17)
set_target_properties(kfcore_image_processor_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "vision/core/image_processor")
cmake_config_target(kfcore_image_processor_core
  ALIAS KFCore::image_processor_core
  EXPORT_NAME image_processor_core
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/image_processor")

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
