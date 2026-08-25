cmake_minimum_required(VERSION 3.20)

project(TrackersInstalledConsumer LANGUAGES C)

set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)

if(NOT DEFINED KFCORE_EXPECTED_KFCORE_DIR OR NOT DEFINED KFCORE_EXPECTED_TRACKERS_DIR)
  message(FATAL_ERROR "KFCORE_EXPECTED_KFCORE_DIR and KFCORE_EXPECTED_TRACKERS_DIR are required")
endif()

find_package(trackers CONFIG REQUIRED
  PATHS "${KFCORE_EXPECTED_TRACKERS_DIR}"
  NO_DEFAULT_PATH)

file(REAL_PATH "${KFCORE_EXPECTED_KFCORE_DIR}" _expected_kfcore_dir)
file(REAL_PATH "${KFCore_DIR}" _resolved_kfcore_dir)
file(REAL_PATH "${KFCORE_EXPECTED_TRACKERS_DIR}" _expected_trackers_dir)
file(REAL_PATH "${trackers_DIR}" _resolved_trackers_dir)
if(NOT _resolved_kfcore_dir STREQUAL _expected_kfcore_dir OR
   NOT _resolved_trackers_dir STREQUAL _expected_trackers_dir)
  message(FATAL_ERROR "Consumer resolved KFCore or trackers outside its install prefix")
endif()

add_executable(trackers_installed_consumer main.c)
target_link_libraries(trackers_installed_consumer PRIVATE KFCore::trackers)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/consumer-target-file-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:trackers_installed_consumer>")
