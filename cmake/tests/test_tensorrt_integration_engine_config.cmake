if(NOT DEFINED KFCORE_SOURCE_DIR OR NOT DEFINED KFCORE_TEST_BINARY_DIR)
  message(FATAL_ERROR "KFCORE_SOURCE_DIR and KFCORE_TEST_BINARY_DIR are required")
endif()

set(_fixture_source
  "${KFCORE_SOURCE_DIR}/cmake/tests/TensorRtIntegrationEngineFixture")
set(_fixture_root "${KFCORE_TEST_BINARY_DIR}/tensorrt_integration_engine_config")
file(REMOVE_RECURSE "${_fixture_root}")
file(MAKE_DIRECTORY "${_fixture_root}")
set(_valid_engine "${_fixture_root}/trusted.engine")
file(WRITE "${_valid_engine}" "fixture")

function(_integration_engine_configure name engine_path)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      -S "${_fixture_source}"
      -B "${_fixture_root}/${name}"
      "-DKFCORE_VALIDATION_MODULE_DIR=${KFCORE_SOURCE_DIR}/cmake"
      "-DKFCORE_FIXTURE_ENGINE_PATH=${engine_path}"
      ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
  set(${name}_RESULT "${_result}" PARENT_SCOPE)
  set(${name}_OUTPUT "${_output}\n${_error}" PARENT_SCOPE)
endfunction()

_integration_engine_configure(valid "${_valid_engine}"
  "-DKFCORE_FIXTURE_EXPECTED_PATH=${_valid_engine}")
if(NOT valid_RESULT EQUAL 0)
  message(FATAL_ERROR "A valid engine file should configure:\n${valid_OUTPUT}")
endif()
execute_process(
  COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${_fixture_root}/valid"
    -C Debug --output-on-failure
  RESULT_VARIABLE _valid_ctest_result
  OUTPUT_VARIABLE _valid_ctest_output
  ERROR_VARIABLE _valid_ctest_error)
if(NOT _valid_ctest_result EQUAL 0)
  message(FATAL_ERROR
    "CTest should receive the validated engine path:\n"
    "${_valid_ctest_output}\n${_valid_ctest_error}")
endif()

_integration_engine_configure(empty "")
if(empty_RESULT EQUAL 0 OR
   NOT empty_OUTPUT MATCHES "KFCORE_TENSORRT_TEST_ENGINE must not be empty")
  message(FATAL_ERROR
    "An empty engine path should fail at configure time:\n${empty_OUTPUT}")
endif()

_integration_engine_configure(nonexistent "${_fixture_root}/missing.engine")
if(nonexistent_RESULT EQUAL 0 OR
   NOT nonexistent_OUTPUT MATCHES "must name an existing file")
  message(FATAL_ERROR
    "A nonexistent engine path should fail at configure time:\n${nonexistent_OUTPUT}")
endif()

_integration_engine_configure(directory "${_fixture_root}")
if(directory_RESULT EQUAL 0 OR
   NOT directory_OUTPUT MATCHES "must name an existing file")
  message(FATAL_ERROR
    "An engine directory should fail at configure time:\n${directory_OUTPUT}")
endif()
