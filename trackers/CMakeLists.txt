include(CMakePackageConfigHelpers)

add_library(trackers SHARED src/tracker.c)
add_library(KFCore::trackers ALIAS trackers)
set_target_properties(trackers PROPERTIES
    FOLDER "trackers" )
target_include_directories(
  trackers PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                  $<INSTALL_INTERFACE:include>)

target_link_libraries(trackers PUBLIC KFCore::kfcore)
target_link_libraries(trackers PRIVATE miniblas)

set_target_properties(
  trackers
  PROPERTIES C_STANDARD 11
             C_STANDARD_REQUIRED ON
             C_EXTENSIONS OFF
             EXPORT_NAME trackers
             WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(BUILD_EXAMPLES)
  add_executable(example examples/basic_tracking.c)
  target_link_libraries(example PRIVATE trackers)
  if(WIN32)
    add_custom_command(
      TARGET example
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              $<TARGET_FILE:KFCore::kfcore> $<TARGET_FILE_DIR:example>)
  endif()
endif()

if(BUILD_TESTS)

  add_executable(test_tracker tests/test_trackers.c)
  target_link_libraries(test_tracker PRIVATE trackers TurboUtils::TinyTest)
  if(WIN32)
    add_custom_command(
      TARGET test_tracker
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              $<TARGET_FILE:KFCore::kfcore> $<TARGET_FILE_DIR:test_tracker>)
  endif()
  add_test(NAME test_tracker COMMAND test_tracker)
  set_target_properties(test_tracker PROPERTIES
    FOLDER "trackers" )
endif()

install(
  TARGETS trackers
  EXPORT trackersTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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
