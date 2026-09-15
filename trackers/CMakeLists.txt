include(CMakePackageConfigHelpers)

add_library(trackers STATIC src/tracker.c)
target_include_directories(
  trackers PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                  $<INSTALL_INTERFACE:include>)

target_link_libraries(trackers PUBLIC KFCore::kfcore)
target_link_libraries(trackers PRIVATE KFCore::miniblas_internal)

set_target_properties(
  trackers
  PROPERTIES C_STANDARD 11
             C_STANDARD_REQUIRED ON
             C_EXTENSIONS OFF
             EXPORT_NAME trackers)
cmake_config_target(trackers
  ALIAS KFCore::trackers
  EXPORT_NAME trackers
  EXPORT_SET trackersTargets
  FOLDER "trackers")

if(BUILD_EXAMPLES)
  add_executable(example examples/basic_tracking.c)
  target_link_libraries(example PRIVATE trackers)
  cmake_config_target(example NO_INSTALL FOLDER "trackers/examples")
endif()

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERNS "*.h")

set(trackers_config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/trackers")
set(trackers_config "${CMAKE_CURRENT_BINARY_DIR}/trackersConfig.cmake")
set(trackers_version_config "${CMAKE_CURRENT_BINARY_DIR}/trackersConfigVersion.cmake")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/trackersConfig.cmake.in"
  "${trackers_config}"
  INSTALL_DESTINATION "${trackers_config_install_dir}")

write_basic_package_version_file(
  "${trackers_version_config}"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(
  EXPORT trackersTargets
  NAMESPACE KFCore::
  DESTINATION "${trackers_config_install_dir}")

install(
  FILES "${trackers_config}" "${trackers_version_config}"
  DESTINATION "${trackers_config_install_dir}")
