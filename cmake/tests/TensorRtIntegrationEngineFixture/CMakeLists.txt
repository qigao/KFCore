cmake_minimum_required(VERSION 3.20)

project(TensorRtIntegrationEngineFixture NONE)

enable_testing()

if(NOT DEFINED KFCORE_VALIDATION_MODULE_DIR OR
   NOT DEFINED KFCORE_FIXTURE_ENGINE_PATH)
  message(FATAL_ERROR
    "KFCORE_VALIDATION_MODULE_DIR and KFCORE_FIXTURE_ENGINE_PATH are required")
endif()

list(PREPEND CMAKE_MODULE_PATH "${KFCORE_VALIDATION_MODULE_DIR}")
include(ValidateTensorRtIntegrationEngine)

kfcore_validate_tensorrt_integration_engine(
  "${KFCORE_FIXTURE_ENGINE_PATH}" _validated_engine)
add_test(NAME fixture_tensorrt_integration
  COMMAND "${CMAKE_COMMAND}"
    "-DEXPECTED_ENGINE=${_validated_engine}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check_engine_environment.cmake")
kfcore_set_tensorrt_integration_test_engine(
  fixture_tensorrt_integration "${_validated_engine}")
get_test_property(
  fixture_tensorrt_integration ENVIRONMENT _fixture_test_environment)
if(NOT _fixture_test_environment STREQUAL
   "KFCORE_TENSORRT_TEST_ENGINE=${_validated_engine}")
  message(FATAL_ERROR
    "CTest did not receive the validated engine path: "
    "${_fixture_test_environment}")
endif()
if(DEFINED KFCORE_FIXTURE_EXPECTED_PATH)
  file(REAL_PATH "${KFCORE_FIXTURE_EXPECTED_PATH}" _expected_engine)
  if(NOT _validated_engine STREQUAL _expected_engine)
    message(FATAL_ERROR
      "Unexpected validated engine path: ${_validated_engine}; expected ${_expected_engine}")
  endif()
endif()
