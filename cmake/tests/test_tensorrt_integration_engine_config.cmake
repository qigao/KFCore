if(NOT DEFINED KFCORE_SOURCE_DIR OR NOT DEFINED KFCORE_TEST_BINARY_DIR)
  message(FATAL_ERROR "KFCORE_SOURCE_DIR and KFCORE_TEST_BINARY_DIR are required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../SafeTestDirectory.cmake")

set(_fixture_source
  "${KFCORE_SOURCE_DIR}/cmake/tests/TensorRtIntegrationEngineFixture")
set(_fixture_root "${KFCORE_TEST_BINARY_DIR}/tensorrt_integration_engine_config")
kfcore_reset_test_directory("${KFCORE_TEST_BINARY_DIR}" "${_fixture_root}")
set(_valid_engine "${_fixture_root}/trusted.engine")
set(_valid_yolo11_face_engine "${_fixture_root}/yolo11-face.engine")
file(WRITE "${_valid_engine}" "fixture")
file(WRITE "${_valid_yolo11_face_engine}" "fixture")

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

function(_assert_fixture_test_registered configuration_name test_name)
  execute_process(
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${_fixture_root}/${configuration_name}"
      -C Debug -N -R "^${test_name}$"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0 OR NOT _output MATCHES "Total Tests: 1")
    message(FATAL_ERROR
      "CTest should register exactly ${test_name}:\n${_output}\n${_error}")
  endif()
endfunction()

function(_assert_fixture_test_count configuration_name expected_count)
  execute_process(
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${_fixture_root}/${configuration_name}"
      -C Debug -N
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0 OR NOT _output MATCHES "Total Tests: ${expected_count}")
    message(FATAL_ERROR
      "CTest should register ${expected_count} tests for ${configuration_name}:\n"
      "${_output}\n${_error}")
  endif()
endfunction()

function(_run_fixture_tests configuration_name)
  execute_process(
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${_fixture_root}/${configuration_name}"
      -C Debug --output-on-failure
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "CTest should receive the validated engine path for ${configuration_name}:\n"
      "${_output}\n${_error}")
  endif()
endfunction()

_integration_engine_configure(valid "${_valid_engine}"
  "-DKFCORE_FIXTURE_ENGINE_PATH_YOLO11_FACE=${_valid_yolo11_face_engine}"
  "-DKFCORE_FIXTURE_EXPECTED_PATH=${_valid_engine}")
if(NOT valid_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Two valid engine files should configure and register distinct CTests:\n${valid_OUTPUT}")
endif()
_assert_fixture_test_count(valid 2)
_assert_fixture_test_registered(valid fixture_tensorrt_integration)
_assert_fixture_test_registered(valid fixture_tensorrt_integration_yolo11_face)
_run_fixture_tests(valid)

_integration_engine_configure(empty_face "${_valid_engine}"
  "-DKFCORE_FIXTURE_ENGINE_PATH_YOLO11_FACE=")
if(NOT empty_face_RESULT EQUAL 0)
  message(FATAL_ERROR
    "An empty optional face engine should register only the primary CTest:\n${empty_face_OUTPUT}")
endif()
_assert_fixture_test_count(empty_face 1)
_assert_fixture_test_registered(empty_face fixture_tensorrt_integration)
_run_fixture_tests(empty_face)

_integration_engine_configure(missing_face "${_valid_engine}"
  "-DKFCORE_FIXTURE_ENGINE_PATH_YOLO11_FACE=${_fixture_root}/missing-face.engine")
if(missing_face_RESULT EQUAL 0 OR
   NOT missing_face_OUTPUT MATCHES
       "KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE must name an existing file")
  message(FATAL_ERROR
    "A nonexistent optional face engine should fail at configure time:\n${missing_face_OUTPUT}")
endif()

_integration_engine_configure(empty ""
  "-DKFCORE_FIXTURE_ENGINE_PATH_YOLO11_FACE=${_valid_yolo11_face_engine}")
if(empty_RESULT EQUAL 0 OR
   NOT empty_OUTPUT MATCHES "KFCORE_TENSORRT_TEST_ENGINE must not be empty")
  message(FATAL_ERROR
    "An empty engine path should fail at configure time:\n${empty_OUTPUT}")
endif()

_integration_engine_configure(nonexistent "${_fixture_root}/missing.engine"
  "-DKFCORE_FIXTURE_ENGINE_PATH_YOLO11_FACE=${_valid_yolo11_face_engine}")
if(nonexistent_RESULT EQUAL 0 OR
   NOT nonexistent_OUTPUT MATCHES "must name an existing file")
  message(FATAL_ERROR
    "A nonexistent engine path should fail at configure time:\n${nonexistent_OUTPUT}")
endif()

_integration_engine_configure(directory "${_fixture_root}"
  "-DKFCORE_FIXTURE_ENGINE_PATH_YOLO11_FACE=${_valid_yolo11_face_engine}")
if(directory_RESULT EQUAL 0 OR
   NOT directory_OUTPUT MATCHES "must name an existing file")
  message(FATAL_ERROR
    "An engine directory should fail at configure time:\n${directory_OUTPUT}")
endif()

set(_runtime_fixture_source
  "${KFCORE_SOURCE_DIR}/cmake/tests/TensorRtRuntimeEngineFixture")
set(_runtime_fixture_root "${_fixture_root}/runtime")
file(MAKE_DIRECTORY "${_runtime_fixture_root}")
set(_runtime_valid_engine "${_runtime_fixture_root}/trusted.engine")
set(_runtime_zero_engine "${_runtime_fixture_root}/zero.engine")
file(WRITE "${_runtime_valid_engine}" "fixture")
file(WRITE "${_runtime_zero_engine}" "")
set(_runtime_relative_engine
  "../TensorRtIntegrationEngineFixture/check_engine_environment.cmake")
set(_runtime_relative_expected
  "${KFCORE_SOURCE_DIR}/cmake/tests/TensorRtIntegrationEngineFixture/check_engine_environment.cmake")

function(_runtime_engine_configure name engine_path)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      -S "${_runtime_fixture_source}"
      -B "${_runtime_fixture_root}/${name}"
      "-DKFCORE_VALIDATION_MODULE_DIR=${KFCORE_SOURCE_DIR}/cmake"
      "-DKFCORE_FIXTURE_RAW_ENGINE_PATH=${engine_path}"
      ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
  set(${name}_RESULT "${_result}" PARENT_SCOPE)
  set(${name}_OUTPUT "${_output}\n${_error}" PARENT_SCOPE)
endfunction()

_runtime_engine_configure(runtime_empty "" -DKFCORE_FIXTURE_EXPECT_EMPTY=ON)
if(NOT runtime_empty_RESULT EQUAL 0 OR
   NOT runtime_empty_OUTPUT MATCHES "is empty.*will not be registered")
  message(FATAL_ERROR
    "An empty runtime engine should report and skip registration:\n${runtime_empty_OUTPUT}")
endif()

_runtime_engine_configure(runtime_directory "${_runtime_fixture_root}")
if(runtime_directory_RESULT EQUAL 0 OR
   NOT runtime_directory_OUTPUT MATCHES "existing non-empty regular file")
  message(FATAL_ERROR
    "A runtime engine directory should fail at configure time:\n${runtime_directory_OUTPUT}")
endif()

_runtime_engine_configure(runtime_zero "${_runtime_zero_engine}")
if(runtime_zero_RESULT EQUAL 0 OR
   NOT runtime_zero_OUTPUT MATCHES "existing non-empty regular file")
  message(FATAL_ERROR
    "A zero-byte runtime engine should fail at configure time:\n${runtime_zero_OUTPUT}")
endif()

_runtime_engine_configure(runtime_relative "${_runtime_relative_engine}"
  "-DKFCORE_FIXTURE_EXPECTED_PATH=${_runtime_relative_expected}")
if(NOT runtime_relative_RESULT EQUAL 0)
  message(FATAL_ERROR
    "A relative non-empty runtime engine should validate:\n${runtime_relative_OUTPUT}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "KFCORE_FIXTURE_ENGINE_PATH=${_runtime_valid_engine}"
    "${CMAKE_COMMAND}"
      -S "${_runtime_fixture_source}"
      -B "${_runtime_fixture_root}/runtime_env_cache"
      "-DKFCORE_VALIDATION_MODULE_DIR=${KFCORE_SOURCE_DIR}/cmake"
      "-DKFCORE_FIXTURE_EXPECTED_PATH=${_runtime_valid_engine}"
  RESULT_VARIABLE runtime_env_cache_RESULT
  OUTPUT_VARIABLE runtime_env_cache_OUTPUT
  ERROR_VARIABLE runtime_env_cache_ERROR)
if(NOT runtime_env_cache_RESULT EQUAL 0)
  message(FATAL_ERROR
    "The same-named parent environment value should initialize the engine cache:\n"
    "${runtime_env_cache_OUTPUT}\n${runtime_env_cache_ERROR}")
endif()
