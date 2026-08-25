function(kfcore_validate_tensorrt_integration_engine configured_path output_variable)
  if("${configured_path}" STREQUAL "")
    message(FATAL_ERROR
      "KFCORE_TENSORRT_TEST_ENGINE must not be empty when "
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
      "KFCORE_TENSORRT_TEST_ENGINE must name an existing file: "
      "${_kfcore_engine_absolute}")
  endif()

  file(REAL_PATH "${_kfcore_engine_absolute}" _kfcore_engine_real)
  set(${output_variable} "${_kfcore_engine_real}" PARENT_SCOPE)
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
