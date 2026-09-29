add_library(kfcore_relation_core STATIC
  src/decode.cpp
  src/relate_anything.cpp)
target_include_directories(kfcore_relation_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_relation_core PUBLIC cxx_std_17)
target_link_libraries(kfcore_relation_core
  PUBLIC
    KFCore::runtime_core
    KFCore::image_processor_core
  PRIVATE
    KFCore::image_processor_cpu)
set_target_properties(kfcore_relation_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision/core/relation")
cmake_config_target(kfcore_relation_core
  ALIAS KFCore::relation_core
  EXPORT_NAME relation_core
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/relation")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
