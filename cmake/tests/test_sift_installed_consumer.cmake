if(NOT DEFINED KFCORE_BINARY_DIR OR NOT DEFINED KFCORE_CONSUMER_SOURCE_DIR OR
   NOT DEFINED KFCORE_CONSUMER_BINARY_DIR OR NOT DEFINED KFCORE_CONSUMER_INSTALL_PREFIX OR
   NOT DEFINED KFCORE_CONSUMER_KFCORE_DIR OR NOT DEFINED KFCORE_CONSUMER_BUILD_CONFIG OR
   NOT DEFINED KFCORE_CONSUMER_CUDATOOLKIT_ROOT OR
   NOT DEFINED KFCORE_EXPECT_SIFT_POPSIFT)
  message(FATAL_ERROR "SIFT installed-consumer test inputs are required")
endif()
if("${KFCORE_CONSUMER_BUILD_CONFIG}" STREQUAL "" OR
   NOT "${KFCORE_CONSUMER_BUILD_CONFIG}" MATCHES "^[A-Za-z0-9_.+-]+$")
  message(FATAL_ERROR "KFCORE_CONSUMER_BUILD_CONFIG must name a CMake configuration")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../SafeTestDirectory.cmake")
set(_consumer_test_root "${KFCORE_BINARY_DIR}/tests")
file(MAKE_DIRECTORY "${_consumer_test_root}")
kfcore_reset_test_directory("${_consumer_test_root}" "${KFCORE_CONSUMER_BINARY_DIR}")
kfcore_reset_test_directory("${_consumer_test_root}" "${KFCORE_CONSUMER_INSTALL_PREFIX}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${KFCORE_BINARY_DIR}"
    --config "${KFCORE_CONSUMER_BUILD_CONFIG}"
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
    "-DCUDAToolkit_ROOT=${KFCORE_CONSUMER_CUDATOOLKIT_ROOT}"
    "-DCMAKE_BUILD_TYPE=${KFCORE_CONSUMER_BUILD_CONFIG}"
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
    --config "${KFCORE_CONSUMER_BUILD_CONFIG}"
  RESULT_VARIABLE _build_result
  OUTPUT_VARIABLE _build_output
  ERROR_VARIABLE _build_error)
if(NOT _build_result EQUAL 0)
  message(FATAL_ERROR
    "SIFT installed consumer build failed:\n${_build_output}\n${_build_error}")
endif()

set(_consumer_target_file
  "${KFCORE_CONSUMER_BINARY_DIR}/consumer-target-file-${KFCORE_CONSUMER_BUILD_CONFIG}.txt")
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
    "PATH=${KFCORE_CONSUMER_INSTALL_PREFIX}/bin;${KFCORE_CONSUMER_CUDATOOLKIT_ROOT}/bin;$ENV{SystemRoot}/System32;$ENV{SystemRoot}")
else()
  set(_runtime_environment
    "LD_LIBRARY_PATH=${KFCORE_CONSUMER_INSTALL_PREFIX}/lib:${KFCORE_CONSUMER_CUDATOOLKIT_ROOT}/lib64")
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
    "${KFCORE_CONSUMER_BINARY_DIR}/popsift-consumer-target-file-${KFCORE_CONSUMER_BUILD_CONFIG}.txt")
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
