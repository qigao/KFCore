add_library(kfcore_runtime_core STATIC
  src/dynamic_library.cpp
  src/plugin.cpp)
target_include_directories(kfcore_runtime_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_runtime_core PUBLIC cxx_std_17)
target_link_libraries(kfcore_runtime_core PUBLIC KFCore::runtime_abi)
if(CMAKE_DL_LIBS)
  target_link_libraries(kfcore_runtime_core PRIVATE ${CMAKE_DL_LIBS})
endif()
set_target_properties(kfcore_runtime_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "runtime/core")
cmake_config_target(kfcore_runtime_core
  ALIAS KFCore::runtime_core
  EXPORT_NAME runtime_core
  EXPORT_SET KFCoreTargets
  FOLDER "runtime/core")

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
