cmake_minimum_required(VERSION 3.20)

project(KFCoreKalmanTargetContract VERSION 0.1.0 LANGUAGES C)

if(NOT DEFINED KFCORE_SOURCE_DIR OR KFCORE_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "KFCORE_SOURCE_DIR is required")
endif()

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

list(APPEND CMAKE_MODULE_PATH "${KFCORE_SOURCE_DIR}/cmake")
include(GNUInstallDirs)
include(CmakeUtils)

add_library(kfcore_miniblas_contract INTERFACE)
add_library(KFCore::miniblas_internal ALIAS kfcore_miniblas_contract)

add_subdirectory("${KFCORE_SOURCE_DIR}/kalman" kalman)

if(NOT TARGET KFCore::kalman)
  message(FATAL_ERROR "KFCore::kalman alias was not created")
endif()
if(NOT TARGET kfcore_kalman)
  message(FATAL_ERROR "kfcore_kalman implementation target was not created")
endif()

get_target_property(_kalman_export_name kfcore_kalman EXPORT_NAME)
if(NOT _kalman_export_name STREQUAL "kalman")
  message(FATAL_ERROR
    "kfcore_kalman export name must be 'kalman', got '${_kalman_export_name}'")
endif()

add_subdirectory("${KFCORE_SOURCE_DIR}/trackers" trackers)

get_target_property(_tracker_links trackers INTERFACE_LINK_LIBRARIES)
if(NOT "KFCore::kalman" IN_LIST _tracker_links)
  message(FATAL_ERROR
    "trackers public link interface must contain KFCore::kalman: ${_tracker_links}")
endif()

message(STATUS "KFCore Kalman target contract passed")
