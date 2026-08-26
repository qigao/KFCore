if(NOT DEFINED KFCORE_SOURCE_DIR OR NOT DEFINED KFCORE_TEST_BINARY_DIR)
  message(FATAL_ERROR "KFCORE_SOURCE_DIR and KFCORE_TEST_BINARY_DIR are required")
endif()

set(_fixture_root "${KFCORE_TEST_BINARY_DIR}/safe_test_directory")
set(_allowed_root "${_fixture_root}/allowed-root")
set(_safe_target "${_allowed_root}/safe-target")
set(_fixture_script "${KFCORE_SOURCE_DIR}/cmake/tests/SafeTestDirectoryFixture.cmake")
set(_safety_module "${KFCORE_SOURCE_DIR}/cmake/SafeTestDirectory.cmake")

file(MAKE_DIRECTORY "${_safe_target}/nested")
file(WRITE "${_safe_target}/nested/stale.txt" "stale")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DKFCORE_SAFETY_MODULE=${_safety_module}"
    "-DKFCORE_ALLOWED_ROOT=${_allowed_root}"
    "-DKFCORE_TARGET_DIR=${_safe_target}"
    -P "${_fixture_script}"
  RESULT_VARIABLE _safe_result
  OUTPUT_VARIABLE _safe_output
  ERROR_VARIABLE _safe_error)
if(NOT _safe_result EQUAL 0 OR
   NOT IS_DIRECTORY "${_safe_target}" OR
   EXISTS "${_safe_target}/nested/stale.txt")
  message(FATAL_ERROR
    "A strict descendant should be reset safely:\n${_safe_output}\n${_safe_error}")
endif()

set(_outside_sentinel "${_fixture_root}/outside-sentinel.txt")
file(WRITE "${_outside_sentinel}" "preserve")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DKFCORE_SAFETY_MODULE=${_safety_module}"
    "-DKFCORE_ALLOWED_ROOT=${_allowed_root}"
    "-DKFCORE_TARGET_DIR=${_fixture_root}"
    -P "${_fixture_script}"
  RESULT_VARIABLE _unsafe_result
  OUTPUT_VARIABLE _unsafe_output
  ERROR_VARIABLE _unsafe_error)
if(_unsafe_result EQUAL 0 OR
   NOT "${_unsafe_output}\n${_unsafe_error}" MATCHES "outside the allowed test root" OR
   NOT EXISTS "${_outside_sentinel}")
  message(FATAL_ERROR
    "An ancestor target should be rejected without deletion:\n"
    "${_unsafe_output}\n${_unsafe_error}")
endif()
