add_library(kfcore_image_processor_cpu STATIC src/cpu.cpp)
target_include_directories(kfcore_image_processor_cpu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/image_processor/src)
target_compile_features(kfcore_image_processor_cpu PUBLIC cxx_std_17)
target_link_libraries(kfcore_image_processor_cpu PUBLIC KFCore::image_processor_core)
set_target_properties(kfcore_image_processor_cpu PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision/image/cpu")
cmake_config_target(kfcore_image_processor_cpu
  ALIAS KFCore::image_processor_cpu
  EXPORT_NAME image_processor_cpu
  EXPORT_SET KFCoreTargets
  FOLDER "vision/image/cpu")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
