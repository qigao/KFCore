function(kfcore_validate_tensorrt_integration_engine configured_path output_variable)
  set(_kfcore_engine_variable "KFCORE_TENSORRT_TEST_ENGINE")
  if(ARGC GREATER 2 AND NOT "${ARGV2}" STREQUAL "")
    set(_kfcore_engine_variable "${ARGV2}")
  endif()

  if("${configured_path}" STREQUAL "")
    message(FATAL_ERROR
      "${_kfcore_engine_variable} must not be empty when "
      "KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS=ON")
  endif()

  set(_kfcore_engine_candidate "${configured_path}")
  cmake_path(ABSOLUTE_PATH _kfcore_engine_candidate
    BASE_DIRECTORY "${CMAKE_SOURCE_DIR}"
    NORMALIZE
    OUTPUT_VARIABLE _kfcore_engine_absolute)
  if(NOT EXISTS "${_kfcore_engine_absolute}" OR
     IS_DIRECTORY "${_kfcore_engine_absolute}")
    message(FATAL_ERROR
      "${_kfcore_engine_variable} must name an existing file: "
      "${_kfcore_engine_absolute}")
  endif()

  file(REAL_PATH "${_kfcore_engine_absolute}" _kfcore_engine_real)
  set(${output_variable} "${_kfcore_engine_real}" PARENT_SCOPE)
endfunction()

function(kfcore_validate_optional_tensorrt_integration_engine configured_path output_variable)
  if("${configured_path}" STREQUAL "")
    set(${output_variable} "" PARENT_SCOPE)
    return()
  endif()

  kfcore_validate_tensorrt_integration_engine(
    "${configured_path}" _validated_engine "${ARGV2}")
  set(${output_variable} "${_validated_engine}" PARENT_SCOPE)
endfunction()

function(kfcore_set_tensorrt_integration_test_engine test_name validated_path)
  if(NOT TEST "${test_name}")
    message(FATAL_ERROR
      "Cannot configure TensorRT integration engine for unknown CTest test: "
      "${test_name}")
  endif()

  set_tests_properties("${test_name}" PROPERTIES
    ENVIRONMENT "KFCORE_TENSORRT_TEST_ENGINE=${validated_path}")
endfunction()

function(kfcore_add_tensorrt_integration_test test_name validated_path)
  if(ARGC LESS 3 OR "${ARGV2}" STREQUAL "")
    message(FATAL_ERROR
      "Cannot register TensorRT integration test without a command: ${test_name}")
  endif()

  add_test(NAME "${test_name}" COMMAND ${ARGN})
  kfcore_set_tensorrt_integration_test_engine("${test_name}" "${validated_path}")
endfunction()

function(kfcore_add_optional_tensorrt_integration_test test_name validated_path)
  if("${validated_path}" STREQUAL "")
    return()
  endif()

  kfcore_add_tensorrt_integration_test("${test_name}" "${validated_path}" ${ARGN})
endfunction()

function(kfcore_validate_optional_tensorrt_runtime_engine
         configured_path cache_variable model_name output_variable)
  if("${configured_path}" STREQUAL "")
    message(STATUS
      "${cache_variable} is empty; ${model_name} real-engine integration tests "
      "will not be registered")
    set(${output_variable} "" PARENT_SCOPE)
    return()
  endif()

  set(_kfcore_runtime_engine_candidate "${configured_path}")
  cmake_path(ABSOLUTE_PATH _kfcore_runtime_engine_candidate
    BASE_DIRECTORY "${CMAKE_SOURCE_DIR}"
    NORMALIZE
    OUTPUT_VARIABLE _kfcore_runtime_engine_absolute)
  if(NOT EXISTS "${_kfcore_runtime_engine_absolute}" OR
     IS_DIRECTORY "${_kfcore_runtime_engine_absolute}")
    message(FATAL_ERROR
      "${cache_variable} must name an existing non-empty regular file: "
      "${_kfcore_runtime_engine_absolute}")
  endif()

  file(SIZE "${_kfcore_runtime_engine_absolute}" _kfcore_runtime_engine_size)
  if(_kfcore_runtime_engine_size LESS_EQUAL 0)
    message(FATAL_ERROR
      "${cache_variable} must name an existing non-empty regular file: "
      "${_kfcore_runtime_engine_absolute}")
  endif()

  file(REAL_PATH "${_kfcore_runtime_engine_absolute}" _kfcore_runtime_engine_real)
  message(STATUS
    "${model_name} real-engine integration tests use: ${_kfcore_runtime_engine_real}")
  set(${output_variable} "${_kfcore_runtime_engine_real}" PARENT_SCOPE)
endfunction()
