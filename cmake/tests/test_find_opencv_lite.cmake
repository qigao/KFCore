if(NOT DEFINED KFCORE_SOURCE_DIR OR NOT DEFINED KFCORE_TEST_BINARY_DIR)
  message(FATAL_ERROR "KFCORE_SOURCE_DIR and KFCORE_TEST_BINARY_DIR are required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../SafeTestDirectory.cmake")

set(_fixture_source "${KFCORE_SOURCE_DIR}/cmake/tests/FindOpenCVLiteFixture")
set(_fixture_root "${KFCORE_TEST_BINARY_DIR}/find_opencv_lite")
kfcore_reset_test_directory("${KFCORE_TEST_BINARY_DIR}" "${_fixture_root}")
file(MAKE_DIRECTORY "${_fixture_root}/sdk/include/opencv2/core"
  "${_fixture_root}/sdk/lib" "${_fixture_root}/sdk/bin")
file(WRITE "${_fixture_root}/sdk/include/opencv2/core.hpp" "#pragma once\n")
file(WRITE "${_fixture_root}/sdk/include/opencv2/core/version.hpp"
  "#define CV_VERSION_MAJOR 4\n#define CV_VERSION_MINOR 13\n#define CV_VERSION_REVISION 0\n")
foreach(_component IN ITEMS core imgproc imgcodecs highgui)
  if(WIN32)
    file(WRITE "${_fixture_root}/sdk/lib/opencv_${_component}4130.lib" "")
    file(WRITE "${_fixture_root}/sdk/bin/opencv_${_component}4130.dll" "")
  else()
    file(WRITE "${_fixture_root}/sdk/lib/libopencv_${_component}4130.a" "")
  endif()
endforeach()

function(_find_opencv_expect_success name scenario sdk_root preseeded_root components absent)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "OPENCV_LITE_ROOT=${sdk_root}"
      "${CMAKE_COMMAND}" -S "${_fixture_source}" -B "${_fixture_root}/${name}"
      "-DFIND_OPENCV_LITE_MODULE_DIR=${KFCORE_SOURCE_DIR}/cmake"
      "-DFIND_OPENCV_LITE_SCENARIO=${scenario}"
      "-DFIND_OPENCV_LITE_PRESEEDED_ROOT=${preseeded_root}"
      "-DFIND_OPENCV_LITE_COMPONENTS=${components}"
      "-DFIND_OPENCV_LITE_EXPECT_ABSENT=${absent}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${name} should configure successfully:\n${_output}\n${_error}")
  endif()
endfunction()

function(_find_opencv_expect_failure name scenario sdk_root preseeded_root components expected_error)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "OPENCV_LITE_ROOT=${sdk_root}"
      "${CMAKE_COMMAND}" -S "${_fixture_source}" -B "${_fixture_root}/${name}"
      "-DFIND_OPENCV_LITE_MODULE_DIR=${KFCORE_SOURCE_DIR}/cmake"
      "-DFIND_OPENCV_LITE_SCENARIO=${scenario}"
      "-DFIND_OPENCV_LITE_PRESEEDED_ROOT=${preseeded_root}"
      "-DFIND_OPENCV_LITE_COMPONENTS=${components}"
      ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
  if(_result EQUAL 0 OR NOT "${_output}\n${_error}" MATCHES "${expected_error}")
    message(FATAL_ERROR
      "${name} should have been rejected with ${expected_error}:\n${_output}\n${_error}")
  endif()
endfunction()

_find_opencv_expect_success(discover discover "${_fixture_root}/sdk"
  "${_fixture_root}/sdk" "core,imgproc,imgcodecs,highgui" "")
_find_opencv_expect_success(dependency_first dependency_first "${_fixture_root}/sdk"
  "${_fixture_root}/sdk" "core,imgproc,imgcodecs,highgui" "")
_find_opencv_expect_success(repeated repeated "${_fixture_root}/sdk"
  "${_fixture_root}/sdk" "core,imgproc,imgcodecs,highgui" "")
_find_opencv_expect_failure(partial partial "${_fixture_root}/sdk"
  "${_fixture_root}/sdk" "core,imgproc,imgcodecs,highgui" "partial imported targets")

set(_poison_root "${_fixture_root}/poison")
file(MAKE_DIRECTORY "${_poison_root}/include/opencv2/core"
  "${_poison_root}/lib" "${_poison_root}/bin")
file(WRITE "${_poison_root}/include/opencv2/core.hpp" "#pragma once\n")
file(WRITE "${_poison_root}/include/opencv2/core/version.hpp"
  "#define CV_VERSION_MAJOR 4\n#define CV_VERSION_MINOR 13\n#define CV_VERSION_REVISION 0\n")
foreach(_component IN ITEMS core imgproc imgcodecs highgui)
  if(WIN32)
    file(WRITE "${_poison_root}/lib/opencv_${_component}4130.lib" "")
    file(WRITE "${_poison_root}/bin/opencv_${_component}4130.dll" "")
  else()
    file(WRITE "${_poison_root}/lib/libopencv_${_component}4130.a" "")
  endif()
endforeach()

set(_abi_mismatch_root "${_fixture_root}/abi_mismatch_sdk")
file(MAKE_DIRECTORY "${_abi_mismatch_root}/include/opencv2/core"
  "${_abi_mismatch_root}/lib" "${_abi_mismatch_root}/bin")
file(WRITE "${_abi_mismatch_root}/include/opencv2/core.hpp" "#pragma once\n")
file(WRITE "${_abi_mismatch_root}/include/opencv2/core/version.hpp"
  "#define CV_VERSION_MAJOR 4\n#define CV_VERSION_MINOR 13\n#define CV_VERSION_REVISION 0\n")
foreach(_component IN ITEMS core imgproc imgcodecs)
  file(WRITE "${_abi_mismatch_root}/lib/opencv_${_component}3416.lib" "")
  file(WRITE "${_abi_mismatch_root}/bin/opencv_${_component}3416.dll" "")
  file(WRITE "${_abi_mismatch_root}/lib/libopencv_${_component}.so.3.4" "")
endforeach()
_find_opencv_expect_failure(
  abi_mismatch_windows abi_mismatch "${_abi_mismatch_root}"
  "${_abi_mismatch_root}" "core,imgproc,imgcodecs"
  "does not match OpenCV Lite header")
_find_opencv_expect_failure(
  abi_mismatch_unix abi_mismatch "${_abi_mismatch_root}"
  "${_abi_mismatch_root}" "core,imgproc,imgcodecs"
  "does not match OpenCV Lite header"
  -DCMAKE_SYSTEM_NAME=Linux)
_find_opencv_expect_failure(root_mismatch root_mismatch "${_fixture_root}/sdk"
  "${_poison_root}" "core,imgproc,imgcodecs,highgui" "resolved outside")

function(_find_opencv_write_sdk root)
  file(MAKE_DIRECTORY "${root}/include/opencv2/core" "${root}/lib" "${root}/bin")
  file(WRITE "${root}/include/opencv2/core.hpp" "#pragma once\n")
  file(WRITE "${root}/include/opencv2/core/version.hpp"
    "#define CV_VERSION_MAJOR 4\n#define CV_VERSION_MINOR 13\n#define CV_VERSION_REVISION 0\n")
  foreach(_component IN LISTS ARGN)
    if(WIN32)
      file(WRITE "${root}/lib/opencv_${_component}4130.lib" "")
      file(WRITE "${root}/bin/opencv_${_component}4130.dll" "")
    else()
      file(WRITE "${root}/lib/libopencv_${_component}4130.a" "")
    endif()
  endforeach()
endfunction()

set(_adapter_root "${_fixture_root}/adapter_sdk")
_find_opencv_write_sdk("${_adapter_root}" core imgproc)
_find_opencv_expect_success(adapter_components discover "${_adapter_root}"
  "${_adapter_root}" "core,imgproc" "imgcodecs")

set(_imgcodecs_root "${_fixture_root}/imgcodecs_sdk")
_find_opencv_write_sdk("${_imgcodecs_root}" imgcodecs)
_find_opencv_expect_success(imgcodecs_component discover "${_imgcodecs_root}"
  "${_imgcodecs_root}" "imgcodecs" "core,imgproc,highgui")

set(_highgui_root "${_fixture_root}/highgui_sdk")
_find_opencv_write_sdk("${_highgui_root}" highgui)
_find_opencv_expect_success(highgui_component discover "${_highgui_root}"
  "${_highgui_root}" "highgui" "core,imgproc,imgcodecs")
