cmake_minimum_required(VERSION 3.20)

project(FindOpenCVLiteFixture NONE)

if(NOT DEFINED FIND_OPENCV_LITE_MODULE_DIR)
  message(FATAL_ERROR "FIND_OPENCV_LITE_MODULE_DIR is required")
endif()

list(PREPEND CMAKE_MODULE_PATH "${FIND_OPENCV_LITE_MODULE_DIR}")
find_package(OpenCVLite REQUIRED)

get_target_property(_core_implib OpenCVLite::core IMPORTED_IMPLIB_RELEASE)
get_target_property(_core_location OpenCVLite::core IMPORTED_LOCATION)
get_target_property(_core_type OpenCVLite::core TYPE)
if(WIN32)
  if(NOT _core_type STREQUAL "SHARED_LIBRARY")
    message(FATAL_ERROR "OpenCV Lite target must model a Windows DLL: ${_core_type}")
  endif()
  if(NOT _core_implib MATCHES "\\.lib$")
    message(FATAL_ERROR "OpenCV Lite release target has no import library: ${_core_implib}")
  endif()
  if(NOT _core_location STREQUAL "_core_location-NOTFOUND")
    message(FATAL_ERROR
      "OpenCV Lite must not set generic IMPORTED_LOCATION on Windows: ${_core_location}")
  endif()
elseif(NOT _core_location MATCHES "opencv_core4130")
  message(FATAL_ERROR "OpenCV Lite target has no resolved Unix library: ${_core_location}")
endif()
