add_library(kfcore_runtime_abi INTERFACE)
target_include_directories(kfcore_runtime_abi INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
cmake_config_target(kfcore_runtime_abi
  ALIAS KFCore::runtime_abi
  EXPORT_NAME runtime_abi
  EXPORT_SET KFCoreTargets
  FOLDER "runtime/abi")

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.h")
