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
kfcore_validate_optional_tensorrt_integration_engine(
  "${KFCORE_FIXTURE_ENGINE_PATH_YOLO11_FACE}" _validated_yolo11_face_engine
  KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE)

kfcore_add_tensorrt_integration_test(
  fixture_tensorrt_integration
  "${_validated_engine}"
  "${CMAKE_COMMAND}"
  "-DEXPECTED_ENGINE=${_validated_engine}"
  "-DEXPECTED_TEST_NAME=fixture_tensorrt_integration"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/check_engine_environment.cmake")
kfcore_add_optional_tensorrt_integration_test(
  fixture_tensorrt_integration_yolo11_face
  "${_validated_yolo11_face_engine}"
  "${CMAKE_COMMAND}"
  "-DEXPECTED_ENGINE=${_validated_yolo11_face_engine}"
  "-DEXPECTED_TEST_NAME=fixture_tensorrt_integration_yolo11_face"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/check_engine_environment.cmake")
if(DEFINED KFCORE_FIXTURE_EXPECTED_PATH)
  file(REAL_PATH "${KFCORE_FIXTURE_EXPECTED_PATH}" _expected_engine)
  if(NOT _validated_engine STREQUAL _expected_engine)
    message(FATAL_ERROR
      "Unexpected validated engine path: ${_validated_engine}; expected ${_expected_engine}")
  endif()
endif()
