cmake_minimum_required(VERSION 3.20)

project(YoloTrackingInstalledConsumer LANGUAGES CXX)

set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)

if(NOT DEFINED KFCORE_EXPECTED_KFCORE_DIR OR NOT DEFINED KFCORE_EXPECTED_TRACKERS_DIR)
  message(FATAL_ERROR "KFCORE_EXPECTED_KFCORE_DIR and KFCORE_EXPECTED_TRACKERS_DIR are required")
endif()
if(NOT DEFINED KFCORE_CONSUMER_MODE)
  message(FATAL_ERROR "Installed-consumer mode is required")
endif()

if(KFCORE_CONSUMER_MODE STREQUAL "dependency_first")
  list(PREPEND CMAKE_MODULE_PATH "${KFCORE_EXPECTED_KFCORE_DIR}")
  find_package(CUDAToolkit REQUIRED)
  find_package(TensorRT REQUIRED MODULE)
  find_package(OpenCVLite REQUIRED MODULE COMPONENTS core imgproc)
endif()

find_package(KFCore CONFIG REQUIRED
  PATHS "${KFCORE_EXPECTED_KFCORE_DIR}"
  NO_DEFAULT_PATH)
if(KFCORE_CONSUMER_MODE STREQUAL "repeated_kfcore")
  find_package(KFCore CONFIG REQUIRED
    PATHS "${KFCORE_EXPECTED_KFCORE_DIR}"
    NO_DEFAULT_PATH)
endif()

file(REAL_PATH "${KFCORE_EXPECTED_KFCORE_DIR}" _expected_kfcore_dir)
file(REAL_PATH "${KFCore_DIR}" _resolved_kfcore_dir)
file(REAL_PATH "${KFCORE_EXPECTED_TRACKERS_DIR}" _expected_trackers_dir)
file(REAL_PATH "${trackers_DIR}" _resolved_trackers_dir)
if(NOT _resolved_kfcore_dir STREQUAL _expected_kfcore_dir OR
   NOT _resolved_trackers_dir STREQUAL _expected_trackers_dir)
  message(FATAL_ERROR "Consumer resolved KFCore or trackers outside its install prefix")
endif()

add_executable(yolo_tracking_installed_consumer main.cpp)
target_link_libraries(yolo_tracking_installed_consumer PRIVATE KFCore::yolo_tracking)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/consumer-target-file-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:yolo_tracking_installed_consumer>")
