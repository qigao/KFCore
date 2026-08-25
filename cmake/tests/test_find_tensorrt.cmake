if(NOT DEFINED KFCORE_SOURCE_DIR)
  message(FATAL_ERROR "KFCORE_SOURCE_DIR is required")
endif()
if(NOT DEFINED KFCORE_TEST_BINARY_DIR)
  message(FATAL_ERROR "KFCORE_TEST_BINARY_DIR is required")
endif()

set(_fixture_source "${KFCORE_SOURCE_DIR}/cmake/tests/FindTensorRTFixture")
set(_fixture_root "${KFCORE_TEST_BINARY_DIR}/find_tensorrt")
file(REMOVE_RECURSE "${_fixture_root}")
file(MAKE_DIRECTORY "${_fixture_root}")

function(_find_tensorrt_write_fixture name version_contents)
  set(_root "${_fixture_root}/${name}/sdk")
  file(MAKE_DIRECTORY "${_root}/include" "${_root}/lib")
  file(WRITE "${_root}/include/NvInfer.h" "#pragma once\n")
  file(WRITE "${_root}/include/NvInferPlugin.h" "#pragma once\n")
  file(WRITE "${_root}/include/NvInferVersion.h" "${version_contents}\n")
  if(WIN32)
    set(_library_prefix "")
    set(_library_suffix ".lib")
  else()
    set(_library_prefix "lib")
    set(_library_suffix ".a")
  endif()
  foreach(_major IN ITEMS 10 11)
    file(WRITE "${_root}/lib/${_library_prefix}nvinfer_${_major}${_library_suffix}" "")
    file(WRITE "${_root}/lib/${_library_prefix}nvinfer_plugin_${_major}${_library_suffix}" "")
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
