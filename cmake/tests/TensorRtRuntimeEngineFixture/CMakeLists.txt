cmake_minimum_required(VERSION 3.20)

project(TensorRtRuntimeEngineFixture NONE)

if(NOT DEFINED KFCORE_VALIDATION_MODULE_DIR)
  message(FATAL_ERROR "KFCORE_VALIDATION_MODULE_DIR is required")
endif()

set(KFCORE_FIXTURE_ENGINE_PATH "$ENV{KFCORE_FIXTURE_ENGINE_PATH}" CACHE FILEPATH
  "Optional fixture engine path")
include("${KFCORE_VALIDATION_MODULE_DIR}/ValidateTensorRtIntegrationEngine.cmake")

set(_configured_engine_path "${KFCORE_FIXTURE_ENGINE_PATH}")
if(DEFINED KFCORE_FIXTURE_RAW_ENGINE_PATH)
  set(_configured_engine_path "${KFCORE_FIXTURE_RAW_ENGINE_PATH}")
endif()

kfcore_validate_optional_tensorrt_runtime_engine(
  "${_configured_engine_path}"
  KFCORE_FIXTURE_ENGINE_PATH
  FixtureModel
  _validated_engine)

if(DEFINED KFCORE_FIXTURE_EXPECT_EMPTY)
  if(NOT "${_validated_engine}" STREQUAL "")
    message(FATAL_ERROR
      "Expected no validated runtime engine, got: ${_validated_engine}")
  endif()
elseif(DEFINED KFCORE_FIXTURE_EXPECTED_PATH)
  file(REAL_PATH "${KFCORE_FIXTURE_EXPECTED_PATH}" _expected_engine)
  if(NOT "${_validated_engine}" STREQUAL "${_expected_engine}")
    message(FATAL_ERROR
      "Unexpected validated runtime engine: ${_validated_engine}; expected ${_expected_engine}")
  endif()
endif()
