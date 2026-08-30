if(NOT DEFINED KFCORE_BINARY_DIR OR NOT DEFINED KFCORE_CONSUMER_SOURCE_DIR OR
   NOT DEFINED KFCORE_CONSUMER_BINARY_DIR OR NOT DEFINED KFCORE_CONSUMER_INSTALL_PREFIX OR
   NOT DEFINED KFCORE_CONSUMER_KFCORE_DIR OR
   NOT DEFINED KFCORE_EXPECT_SIFT_POPSIFT)
  message(FATAL_ERROR "SIFT installed-consumer test inputs are required")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../SafeTestDirectory.cmake")
kfcore_require_environment_directory(CUDA_TOOLKIT_ROOT)
kfcore_get_parent_build_config("${KFCORE_BINARY_DIR}" _consumer_build_config)
set(_consumer_test_root "${KFCORE_BINARY_DIR}/tests")
file(MAKE_DIRECTORY "${_consumer_test_root}")
kfcore_reset_test_directory("${_consumer_test_root}" "${KFCORE_CONSUMER_BINARY_DIR}")
kfcore_reset_test_directory("${_consumer_test_root}" "${KFCORE_CONSUMER_INSTALL_PREFIX}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${KFCORE_BINARY_DIR}"
    --config "${_consumer_build_config}"
    --prefix "${KFCORE_CONSUMER_INSTALL_PREFIX}"
  RESULT_VARIABLE _install_result
  OUTPUT_VARIABLE _install_output
  ERROR_VARIABLE _install_error)
if(NOT _install_result EQUAL 0)
  message(FATAL_ERROR "KFCore install failed:\n${_install_output}\n${_install_error}")
endif()

if(KFCORE_EXPECT_SIFT_POPSIFT)
  foreach(_required_metadata COPYING.md UPSTREAM.md)
    if(NOT EXISTS
       "${KFCORE_CONSUMER_INSTALL_PREFIX}/share/licenses/KFCore/popsift/${_required_metadata}")
      message(FATAL_ERROR "Installed CUDA SIFT metadata is missing: ${_required_metadata}")
    endif()
  endforeach()
  if(NOT DEFINED KFCORE_EXPECTED_SOURCE_REVISION OR
     "${KFCORE_EXPECTED_SOURCE_REVISION}" STREQUAL "")
    message(FATAL_ERROR "Expected CUDA SIFT source revision is required")
  endif()
  set(_popsift_upstream
    "${KFCORE_CONSUMER_INSTALL_PREFIX}/share/licenses/KFCore/popsift/UPSTREAM.md")
  file(READ "${_popsift_upstream}" _popsift_upstream_content)
  string(FIND "${_popsift_upstream_content}"
    "KFCore source revision: `${KFCORE_EXPECTED_SOURCE_REVISION}`"
    _popsift_revision_index)
  string(FIND "${_popsift_upstream_content}"
    "/tree/${KFCORE_EXPECTED_SOURCE_REVISION}/sift/vendor/popsift"
    _popsift_url_index)
  if(_popsift_revision_index EQUAL -1 OR _popsift_url_index EQUAL -1 OR
     "${_popsift_upstream_content}" MATCHES "<KFCORE|@KFCORE")
    message(FATAL_ERROR
      "Installed CUDA SIFT provenance does not contain the configured immutable revision")
  endif()
  if(EXISTS "${KFCORE_CONSUMER_INSTALL_PREFIX}/bin/popsift.dll")
    message(FATAL_ERROR "Installed CUDA SIFT backend unexpectedly deploys popsift.dll")
  endif()
  if(EXISTS "${KFCORE_CONSUMER_INSTALL_PREFIX}/include/popsift")
    message(FATAL_ERROR "Installed CUDA SIFT backend unexpectedly exposes internal headers")
  endif()
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${KFCORE_CONSUMER_SOURCE_DIR}"
    -B "${KFCORE_CONSUMER_BINARY_DIR}"
    "-DKFCore_DIR=${KFCORE_CONSUMER_KFCORE_DIR}"
    "-DKFCORE_EXPECTED_KFCORE_DIR=${KFCORE_CONSUMER_KFCORE_DIR}"
    "-DKFCORE_EXPECT_SIFT_POPSIFT=${KFCORE_EXPECT_SIFT_POPSIFT}"
    "-DCMAKE_BUILD_TYPE=${_consumer_build_config}"
    -DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE
    -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE
    -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=TRUE
  RESULT_VARIABLE _configure_result
  OUTPUT_VARIABLE _configure_output
  ERROR_VARIABLE _configure_error)
if(NOT _configure_result EQUAL 0)
  message(FATAL_ERROR
    "SIFT installed consumer configure failed:\n${_configure_output}\n${_configure_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${KFCORE_CONSUMER_BINARY_DIR}"
    --config "${_consumer_build_config}"
  RESULT_VARIABLE _build_result
  OUTPUT_VARIABLE _build_output
  ERROR_VARIABLE _build_error)
if(NOT _build_result EQUAL 0)
  message(FATAL_ERROR
    "SIFT installed consumer build failed:\n${_build_output}\n${_build_error}")
endif()

set(_consumer_target_file
  "${KFCORE_CONSUMER_BINARY_DIR}/consumer-target-file-${_consumer_build_config}.txt")
if(NOT EXISTS "${_consumer_target_file}")
  message(FATAL_ERROR "Consumer did not generate its target file path")
endif()
file(READ "${_consumer_target_file}" _consumer_executable)
string(STRIP "${_consumer_executable}" _consumer_executable)
if(NOT EXISTS "${_consumer_executable}")
  message(FATAL_ERROR "Consumer executable does not exist: ${_consumer_executable}")
endif()

if(WIN32)
  set(_runtime_environment
    "PATH=${KFCORE_CONSUMER_INSTALL_PREFIX}/bin;$ENV{CUDA_TOOLKIT_ROOT}/bin;$ENV{SystemRoot}/System32;$ENV{SystemRoot}")
else()
  set(_runtime_environment
    "LD_LIBRARY_PATH=${KFCORE_CONSUMER_INSTALL_PREFIX}/lib:$ENV{CUDA_TOOLKIT_ROOT}/lib64")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "${_runtime_environment}" "${_consumer_executable}"
  RESULT_VARIABLE _run_result
  OUTPUT_VARIABLE _run_output
  ERROR_VARIABLE _run_error)
if(NOT _run_result EQUAL 0)
  message(FATAL_ERROR
    "SIFT installed consumer failed:\n${_run_output}\n${_run_error}")
endif()

if(KFCORE_EXPECT_SIFT_POPSIFT)
  set(_popsift_consumer_target_file
    "${KFCORE_CONSUMER_BINARY_DIR}/popsift-consumer-target-file-${_consumer_build_config}.txt")
  if(NOT EXISTS "${_popsift_consumer_target_file}")
    message(FATAL_ERROR "PopSift consumer did not generate its target file path")
  endif()
  file(READ "${_popsift_consumer_target_file}" _popsift_consumer_executable)
  string(STRIP "${_popsift_consumer_executable}" _popsift_consumer_executable)
  if(NOT EXISTS "${_popsift_consumer_executable}")
    message(FATAL_ERROR
      "PopSift consumer executable does not exist: ${_popsift_consumer_executable}")
  endif()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "${_runtime_environment}"
      "${_popsift_consumer_executable}"
    RESULT_VARIABLE _popsift_run_result
    OUTPUT_VARIABLE _popsift_run_output
    ERROR_VARIABLE _popsift_run_error)
  if(NOT _popsift_run_result EQUAL 0)
    message(FATAL_ERROR
      "Installed PopSift consumer failed:\n${_popsift_run_output}\n${_popsift_run_error}")
  endif()
endif()
