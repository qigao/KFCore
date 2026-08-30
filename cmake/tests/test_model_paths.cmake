if(NOT DEFINED KFCORE_SOURCE_DIR OR NOT DEFINED KFCORE_TEST_BINARY_DIR)
  message(FATAL_ERROR "KFCORE_SOURCE_DIR and KFCORE_TEST_BINARY_DIR are required")
endif()

include("${KFCORE_SOURCE_DIR}/cmake/SafeTestDirectory.cmake")
set(_test_root "${KFCORE_TEST_BINARY_DIR}/model_paths")
kfcore_reset_test_directory("${KFCORE_TEST_BINARY_DIR}" "${_test_root}")
set(_model_root "${_test_root}/models")
file(MAKE_DIRECTORY "${_model_root}/nested")
file(WRITE "${_model_root}/nested/model.onnx" "model")

set(_fixture "${KFCORE_SOURCE_DIR}/cmake/tests/ModelPathFixture.cmake")
set(_module "${KFCORE_SOURCE_DIR}/cmake/ModelPaths.cmake")

function(_run_fixture name relative_path optional expected_success expected_pattern)
  set(_result_file "${_test_root}/${name}.txt")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DKFCORE_MODEL_PATH_MODULE=${_module}"
      "-DKFCORE_MODEL_ROOT=${_model_root}"
      "-DKFCORE_MODEL_RELATIVE=${relative_path}"
      "-DKFCORE_MODEL_OPTIONAL=${optional}"
      "-DKFCORE_MODEL_RESULT=${_result_file}"
      -P "${_fixture}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
  if(expected_success)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR "${name} unexpectedly failed:\n${_output}\n${_error}")
    endif()
    file(READ "${_result_file}" _resolved)
    if("${relative_path}" STREQUAL "")
      set(_expected "")
    else()
      file(REAL_PATH "${_model_root}/nested/model.onnx" _expected)
    endif()
    if(NOT _resolved STREQUAL _expected)
      message(FATAL_ERROR "${name} resolved '${_resolved}', expected '${_expected}'")
    endif()
  else()
    if(_result EQUAL 0)
      message(FATAL_ERROR "${name} unexpectedly succeeded")
    endif()
    if(NOT "${_output}\n${_error}" MATCHES "${expected_pattern}")
      message(FATAL_ERROR
        "${name} did not report '${expected_pattern}':\n${_output}\n${_error}")
    endif()
  endif()
endfunction()

_run_fixture(valid_relative "nested/model.onnx" FALSE TRUE "")
_run_fixture(optional_relative "nested/model.onnx" TRUE TRUE "")
_run_fixture(optional_empty "" TRUE TRUE "")
_run_fixture(absolute_relative "${_model_root}/nested/model.onnx" FALSE FALSE
  "relative model path must not be absolute")
_run_fixture(parent_escape "../outside.onnx" FALSE FALSE
  "relative model path escapes KFCORE_MODEL_ROOT")
_run_fixture(missing_file "nested/missing.onnx" FALSE FALSE
  "resolved model path is not an existing file")
