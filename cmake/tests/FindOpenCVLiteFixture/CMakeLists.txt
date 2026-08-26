cmake_minimum_required(VERSION 3.20)

project(FindOpenCVLiteFixture NONE)

if(NOT DEFINED FIND_OPENCV_LITE_MODULE_DIR)
  message(FATAL_ERROR "FIND_OPENCV_LITE_MODULE_DIR is required")
endif()
if(NOT DEFINED FIND_OPENCV_LITE_SCENARIO)
  set(FIND_OPENCV_LITE_SCENARIO discover)
endif()
if(NOT DEFINED FIND_OPENCV_LITE_COMPONENTS)
  set(FIND_OPENCV_LITE_COMPONENTS "core,imgproc,imgcodecs")
endif()
string(REPLACE "," ";" _fixture_requested_components
  "${FIND_OPENCV_LITE_COMPONENTS}")

list(PREPEND CMAKE_MODULE_PATH "${FIND_OPENCV_LITE_MODULE_DIR}")

function(_fixture_add_opencv_target component)
  if(NOT DEFINED FIND_OPENCV_LITE_PRESEEDED_ROOT)
    message(FATAL_ERROR "FIND_OPENCV_LITE_PRESEEDED_ROOT is required for preseeded targets")
  endif()
  if(WIN32)
    if(FIND_OPENCV_LITE_SCENARIO STREQUAL "abi_mismatch")
      set(_fixture_opencv_suffix 3416)
    else()
      set(_fixture_opencv_suffix 4130)
    endif()
    add_library(OpenCVLite::${component} SHARED IMPORTED)
    set_target_properties(OpenCVLite::${component} PROPERTIES
      IMPORTED_CONFIGURATIONS RELEASE
      IMPORTED_IMPLIB_RELEASE
        "${FIND_OPENCV_LITE_PRESEEDED_ROOT}/lib/opencv_${component}${_fixture_opencv_suffix}.lib"
      IMPORTED_LOCATION_RELEASE
        "${FIND_OPENCV_LITE_PRESEEDED_ROOT}/bin/opencv_${component}${_fixture_opencv_suffix}.dll"
      INTERFACE_INCLUDE_DIRECTORIES "${FIND_OPENCV_LITE_PRESEEDED_ROOT}/include")
  else()
    if(FIND_OPENCV_LITE_SCENARIO STREQUAL "abi_mismatch")
      set(_fixture_opencv_library "libopencv_${component}.so.3.4")
    else()
      set(_fixture_opencv_library "libopencv_${component}4130.a")
    endif()
    add_library(OpenCVLite::${component} UNKNOWN IMPORTED)
    set_target_properties(OpenCVLite::${component} PROPERTIES
      IMPORTED_LOCATION
        "${FIND_OPENCV_LITE_PRESEEDED_ROOT}/lib/${_fixture_opencv_library}"
      INTERFACE_INCLUDE_DIRECTORIES "${FIND_OPENCV_LITE_PRESEEDED_ROOT}/include")
  endif()
endfunction()

if(FIND_OPENCV_LITE_SCENARIO MATCHES
   "^(dependency_first|partial|root_mismatch|abi_mismatch)$")
  _fixture_add_opencv_target(core)
  if(NOT FIND_OPENCV_LITE_SCENARIO STREQUAL "partial")
    _fixture_add_opencv_target(imgproc)
    _fixture_add_opencv_target(imgcodecs)
  endif()
endif()

find_package(OpenCVLite REQUIRED COMPONENTS ${_fixture_requested_components})
if(FIND_OPENCV_LITE_SCENARIO STREQUAL "repeated")
  find_package(OpenCVLite REQUIRED COMPONENTS ${_fixture_requested_components})
endif()

foreach(_fixture_component IN LISTS _fixture_requested_components)
  if(NOT TARGET OpenCVLite::${_fixture_component})
    message(FATAL_ERROR "Requested target was not exported: OpenCVLite::${_fixture_component}")
  endif()
endforeach()
if(DEFINED FIND_OPENCV_LITE_EXPECT_ABSENT)
  string(REPLACE "," ";" _fixture_absent_components
    "${FIND_OPENCV_LITE_EXPECT_ABSENT}")
  foreach(_fixture_component IN LISTS _fixture_absent_components)
    if(TARGET OpenCVLite::${_fixture_component})
      message(FATAL_ERROR
        "Unrequested target was exported: OpenCVLite::${_fixture_component}")
    endif()
  endforeach()
endif()

if(TARGET OpenCVLite::core)
  get_target_property(_core_implib OpenCVLite::core IMPORTED_IMPLIB_RELEASE)
  get_target_property(_core_location OpenCVLite::core IMPORTED_LOCATION)
  get_target_property(_core_location_release OpenCVLite::core IMPORTED_LOCATION_RELEASE)
  get_target_property(_core_type OpenCVLite::core TYPE)
endif()
if(WIN32 AND TARGET OpenCVLite::core)
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
  if(NOT _core_location_release MATCHES "[/\\\\]bin[/\\\\]opencv_core4130\\.dll$")
    message(FATAL_ERROR "OpenCV Lite release runtime is not the expected DLL: ${_core_location_release}")
  endif()
  file(REAL_PATH "$ENV{OPENCV_LITE_ROOT}" _opencv_lite_root)
  file(REAL_PATH "${_core_location_release}" _opencv_lite_runtime)
  cmake_path(IS_PREFIX _opencv_lite_root "${_opencv_lite_runtime}" NORMALIZE _runtime_contained)
  if(NOT _runtime_contained)
    message(FATAL_ERROR "OpenCV Lite release runtime escapes OPENCV_LITE_ROOT")
  endif()
elseif(TARGET OpenCVLite::core AND
       NOT FIND_OPENCV_LITE_SCENARIO STREQUAL "abi_mismatch" AND
       NOT _core_location MATCHES "opencv_core4130")
  message(FATAL_ERROR "OpenCV Lite target has no resolved Unix library: ${_core_location}")
endif()
