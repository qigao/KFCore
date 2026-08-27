if(NOT DEFINED KFCORE_SOURCE_DIR OR NOT DEFINED KFCORE_TEST_BINARY_DIR)
  message(FATAL_ERROR "KFCORE_SOURCE_DIR and KFCORE_TEST_BINARY_DIR are required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../SafeTestDirectory.cmake")
set(_source "${KFCORE_SOURCE_DIR}/cmake/tests/FindONNXRuntimeFixture")
set(_root "${KFCORE_TEST_BINARY_DIR}/find_onnxruntime")
kfcore_reset_test_directory("${KFCORE_TEST_BINARY_DIR}" "${_root}")
file(MAKE_DIRECTORY "${_root}/sdk/include" "${_root}/sdk/lib")
file(WRITE "${_root}/sdk/include/onnxruntime_cxx_api.h" "#pragma once\n")
if(WIN32)
  file(WRITE "${_root}/sdk/lib/onnxruntime.lib" "")
  file(WRITE "${_root}/sdk/lib/onnxruntime.dll" "")
else()
  file(WRITE "${_root}/sdk/lib/libonnxruntime.so" "")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "ONNXRUNTIME_ROOT=${_root}/sdk"
    "${CMAKE_COMMAND}" -S "${_source}" -B "${_root}/valid"
    "-DFIND_ONNXRUNTIME_MODULE_DIR=${KFCORE_SOURCE_DIR}/cmake"
  RESULT_VARIABLE _valid_result
  OUTPUT_VARIABLE _valid_output
  ERROR_VARIABLE _valid_error)
if(NOT _valid_result EQUAL 0)
  message(FATAL_ERROR "valid ONNX Runtime SDK was rejected:\n${_valid_output}\n${_valid_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "ONNXRUNTIME_ROOT=${_root}/missing"
    "${CMAKE_COMMAND}" -S "${_source}" -B "${_root}/invalid"
    "-DFIND_ONNXRUNTIME_MODULE_DIR=${KFCORE_SOURCE_DIR}/cmake"
  RESULT_VARIABLE _invalid_result
  OUTPUT_VARIABLE _invalid_output
  ERROR_VARIABLE _invalid_error)
if(_invalid_result EQUAL 0 OR
   NOT "${_invalid_output}\n${_invalid_error}" MATCHES "ONNXRUNTIME_ROOT")
  message(FATAL_ERROR "missing ONNX Runtime root was not rejected")
endif()
