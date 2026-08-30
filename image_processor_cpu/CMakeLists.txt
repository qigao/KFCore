add_library(kfcore_image_processor_cpu SHARED src/cpu.cpp)
target_include_directories(kfcore_image_processor_cpu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_SOURCE_DIR}/image_processor_core/src)
target_compile_features(kfcore_image_processor_cpu PUBLIC cxx_std_17)
target_link_libraries(kfcore_image_processor_cpu PUBLIC KFCore::image_processor_core)
set_target_properties(kfcore_image_processor_cpu PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "image_processor_cpu")
cmake_config_target(kfcore_image_processor_cpu
  ALIAS KFCore::image_processor_cpu
  EXPORT_NAME image_processor_cpu
  EXPORT_SET KFCoreTargets
  FOLDER "image_processor_cpu")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
