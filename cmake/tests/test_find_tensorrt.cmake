if(NOT DEFINED KFCORE_SOURCE_DIR)
  message(FATAL_ERROR "KFCORE_SOURCE_DIR is required")
endif()
if(NOT DEFINED KFCORE_TEST_BINARY_DIR)
  message(FATAL_ERROR "KFCORE_TEST_BINARY_DIR is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../SafeTestDirectory.cmake")

set(_fixture_source "${KFCORE_SOURCE_DIR}/cmake/tests/FindTensorRTFixture")
set(_fixture_root "${KFCORE_TEST_BINARY_DIR}/find_tensorrt")
kfcore_reset_test_directory("${KFCORE_TEST_BINARY_DIR}" "${_fixture_root}")

function(_find_tensorrt_write_fixture name version_contents)
  set(_root "${_fixture_root}/${name}/sdk")
  file(MAKE_DIRECTORY "${_root}/include" "${_root}/lib")
  file(WRITE "${_root}/include/NvInfer.h" "#pragma once\n")
  file(WRITE "${_root}/include/NvInferPlugin.h" "#pragma once\n")
  file(WRITE "${_root}/include/NvInferVersion.h" "${version_contents}\n")
  foreach(_major IN ITEMS 10 11)
    foreach(_library IN ITEMS nvinfer nvinfer_plugin)
      file(WRITE "${_root}/lib/${_library}_${_major}.lib" "")
      file(WRITE "${_root}/lib/lib${_library}_${_major}.a" "")
      file(WRITE "${_root}/lib/lib${_library}.so.${_major}" "")
    endforeach()
  endforeach()
  foreach(_library IN ITEMS nvinfer nvinfer_plugin)
    file(WRITE "${_root}/lib/${_library}.lib" "")
    file(WRITE "${_root}/lib/lib${_library}.a" "")
    file(WRITE "${_root}/lib/lib${_library}.so" "")
  endforeach()
  set(${name}_ROOT "${_root}" PARENT_SCOPE)
endfunction()

function(_find_tensorrt_expect_success name expected_version scenario)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "TENSORRT_ROOT=${${name}_ROOT}"
      "${CMAKE_COMMAND}" -S "${_fixture_source}" -B "${_fixture_root}/${name}/build"
      "-DFIND_TENSORRT_MODULE_DIR=${KFCORE_SOURCE_DIR}/cmake"
      "-DFIND_TENSORRT_EXPECTED_VERSION=${expected_version}"
      "-DFIND_TENSORRT_SCENARIO=${scenario}"
      "-DFIND_TENSORRT_PRESEEDED_ROOT=${${name}_PRESEEDED_ROOT}"
      ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${name} should configure successfully:\n${_output}\n${_error}")
  endif()
endfunction()

function(_find_tensorrt_expect_failure name expected_error scenario)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "TENSORRT_ROOT=${${name}_ROOT}"
      "${CMAKE_COMMAND}" -S "${_fixture_source}" -B "${_fixture_root}/${name}/build"
      "-DFIND_TENSORRT_MODULE_DIR=${KFCORE_SOURCE_DIR}/cmake"
      "-DFIND_TENSORRT_EXPECTED_VERSION=0.0.0"
      "-DFIND_TENSORRT_SCENARIO=${scenario}"
      "-DFIND_TENSORRT_PRESEEDED_ROOT=${${name}_PRESEEDED_ROOT}"
      ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
  if(_result EQUAL 0 OR NOT "${_output}\n${_error}" MATCHES "${expected_error}")
    message(FATAL_ERROR
      "${name} should have been rejected with ${expected_error}:\n${_output}\n${_error}")
  endif()
endfunction()

_find_tensorrt_write_fixture(direct_numeric
  "#define NV_TENSORRT_MAJOR 10\n#define NV_TENSORRT_MINOR 9\n#define NV_TENSORRT_PATCH 0")
set(direct_numeric_PRESEEDED_ROOT "${direct_numeric_ROOT}")
_find_tensorrt_expect_success(direct_numeric "10.9.0" discover)

_find_tensorrt_write_fixture(enterprise_indirect
  "#define TRT_MAJOR_ENTERPRISE 11\n#define TRT_MINOR_ENTERPRISE 2\n#define TRT_PATCH_ENTERPRISE 1\n#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE\n#define NV_TENSORRT_MINOR TRT_MINOR_ENTERPRISE\n#define NV_TENSORRT_PATCH TRT_PATCH_ENTERPRISE")
set(enterprise_indirect_PRESEEDED_ROOT "${enterprise_indirect_ROOT}")
_find_tensorrt_expect_success(enterprise_indirect "11.2.1" discover)

_find_tensorrt_write_fixture(dependency_first
  "#define NV_TENSORRT_MAJOR 11\n#define NV_TENSORRT_MINOR 2\n#define NV_TENSORRT_PATCH 1")
set(dependency_first_PRESEEDED_ROOT "${dependency_first_ROOT}")
_find_tensorrt_expect_success(dependency_first "11.2.1" dependency_first)

_find_tensorrt_write_fixture(repeated
  "#define NV_TENSORRT_MAJOR 11\n#define NV_TENSORRT_MINOR 2\n#define NV_TENSORRT_PATCH 1")
set(repeated_PRESEEDED_ROOT "${repeated_ROOT}")
_find_tensorrt_expect_success(repeated "11.2.1" repeated)

_find_tensorrt_write_fixture(unversioned
  "#define NV_TENSORRT_MAJOR 11\n#define NV_TENSORRT_MINOR 2\n#define NV_TENSORRT_PATCH 1")
set(unversioned_PRESEEDED_ROOT "${unversioned_ROOT}")
_find_tensorrt_expect_success(unversioned "11.2.1" unversioned)
set(unversioned_unix_ROOT "${unversioned_ROOT}")
set(unversioned_unix_PRESEEDED_ROOT "${unversioned_ROOT}")
_find_tensorrt_expect_success(
  unversioned_unix "11.2.1" unversioned -DCMAKE_SYSTEM_NAME=Linux)

_find_tensorrt_write_fixture(abi_mismatch
  "#define NV_TENSORRT_MAJOR 11\n#define NV_TENSORRT_MINOR 2\n#define NV_TENSORRT_PATCH 1")
set(abi_mismatch_windows_ROOT "${abi_mismatch_ROOT}")
set(abi_mismatch_windows_PRESEEDED_ROOT "${abi_mismatch_ROOT}")
_find_tensorrt_expect_failure(
  abi_mismatch_windows "does not match TensorRT major" abi_mismatch)
set(abi_mismatch_unix_ROOT "${abi_mismatch_ROOT}")
set(abi_mismatch_unix_PRESEEDED_ROOT "${abi_mismatch_ROOT}")
_find_tensorrt_expect_failure(
  abi_mismatch_unix "does not match TensorRT major" abi_mismatch
  -DCMAKE_SYSTEM_NAME=Linux)

_find_tensorrt_write_fixture(partial
  "#define NV_TENSORRT_MAJOR 11\n#define NV_TENSORRT_MINOR 2\n#define NV_TENSORRT_PATCH 1")
set(partial_PRESEEDED_ROOT "${partial_ROOT}")
_find_tensorrt_expect_failure(partial "partial imported targets" partial)

_find_tensorrt_write_fixture(root_mismatch
  "#define NV_TENSORRT_MAJOR 11\n#define NV_TENSORRT_MINOR 2\n#define NV_TENSORRT_PATCH 1")
_find_tensorrt_write_fixture(root_mismatch_poison
  "#define NV_TENSORRT_MAJOR 11\n#define NV_TENSORRT_MINOR 2\n#define NV_TENSORRT_PATCH 1")
set(root_mismatch_PRESEEDED_ROOT "${root_mismatch_poison_ROOT}")
_find_tensorrt_expect_failure(root_mismatch "resolved outside" root_mismatch)

_find_tensorrt_write_fixture(unresolvable
  "#define NV_TENSORRT_MAJOR TRT_MAJOR_UNDEFINED\n#define NV_TENSORRT_MINOR 2\n#define NV_TENSORRT_PATCH 1")
set(unresolvable_PRESEEDED_ROOT "${unresolvable_ROOT}")
_find_tensorrt_expect_failure(unresolvable "unparseable macro" discover)

_find_tensorrt_write_fixture(unsupported
  "#define NV_TENSORRT_MAJOR 9\n#define NV_TENSORRT_MINOR 0\n#define NV_TENSORRT_PATCH 0")
set(unsupported_PRESEEDED_ROOT "${unsupported_ROOT}")
_find_tensorrt_expect_failure(unsupported "Unsupported TensorRT major version 9" discover)
