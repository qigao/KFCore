if(NOT DEFINED KFCORE_SOURCE_DIR OR NOT DEFINED KFCORE_TEST_BINARY_DIR)
  message(FATAL_ERROR "KFCORE_SOURCE_DIR and KFCORE_TEST_BINARY_DIR are required")
endif()

set(_fixture_source "${KFCORE_SOURCE_DIR}/cmake/tests/FindOpenCVLiteFixture")
set(_fixture_root "${KFCORE_TEST_BINARY_DIR}/find_opencv_lite")
file(REMOVE_RECURSE "${_fixture_root}")
file(MAKE_DIRECTORY "${_fixture_root}/sdk/include/opencv2/core"
  "${_fixture_root}/sdk/lib" "${_fixture_root}/sdk/bin")
file(WRITE "${_fixture_root}/sdk/include/opencv2/core.hpp" "#pragma once\n")
file(WRITE "${_fixture_root}/sdk/include/opencv2/core/version.hpp"
  "#define CV_VERSION_MAJOR 4\n#define CV_VERSION_MINOR 13\n#define CV_VERSION_REVISION 0\n")
foreach(_component IN ITEMS core imgproc imgcodecs)
  if(WIN32)
    file(WRITE "${_fixture_root}/sdk/lib/opencv_${_component}4130.lib" "")
    file(WRITE "${_fixture_root}/sdk/bin/opencv_${_component}4130.dll" "")
  else()
    file(WRITE "${_fixture_root}/sdk/lib/libopencv_${_component}4130.a" "")
  endif()
endforeach()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "OPENCV_LITE_ROOT=${_fixture_root}/sdk"
    "${CMAKE_COMMAND}" -S "${_fixture_source}" -B "${_fixture_root}/build"
    "-DFIND_OPENCV_LITE_MODULE_DIR=${KFCORE_SOURCE_DIR}/cmake"
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _output
  ERROR_VARIABLE _error)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "OpenCV Lite fixture should configure successfully:\n${_output}\n${_error}")
endif()
