if(NOT DEFINED KFCORE_BINARY_DIR OR
   NOT DEFINED KFCORE_CONSUMER_SOURCE_DIR OR
   NOT DEFINED KFCORE_CONSUMER_BINARY_DIR OR
   NOT DEFINED KFCORE_CONSUMER_INSTALL_PREFIX OR
   NOT DEFINED KFCORE_CONSUMER_KFCORE_DIR)
  message(FATAL_ERROR "Face application installed-consumer test paths are required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../SafeTestDirectory.cmake")
kfcore_require_environment_directory(CUDA_TOOLKIT_ROOT)
kfcore_require_environment_directory(TENSORRT_ROOT)
kfcore_require_environment_directory(TURBOUTILS_ROOT)
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

execute_process(
  COMMAND "${CMAKE_COMMAND}" -G Ninja -S "${KFCORE_CONSUMER_SOURCE_DIR}"
    -B "${KFCORE_CONSUMER_BINARY_DIR}"
    "-DKFCore_DIR=${KFCORE_CONSUMER_KFCORE_DIR}"
    "-DKFCORE_EXPECTED_KFCORE_DIR=${KFCORE_CONSUMER_KFCORE_DIR}"
    "-DCMAKE_BUILD_TYPE=${_consumer_build_config}"
    -DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE
    -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE
    -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=TRUE
  RESULT_VARIABLE _configure_result
  OUTPUT_VARIABLE _configure_output
  ERROR_VARIABLE _configure_error)
if(NOT _configure_result EQUAL 0)
  message(FATAL_ERROR
    "Face application installed consumer configure failed:\n"
    "${_configure_output}\n${_configure_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${KFCORE_CONSUMER_BINARY_DIR}"
    --config "${_consumer_build_config}"
  RESULT_VARIABLE _build_result
  OUTPUT_VARIABLE _build_output
  ERROR_VARIABLE _build_error)
if(NOT _build_result EQUAL 0)
  message(FATAL_ERROR
    "Face application installed consumer build failed:\n${_build_output}\n${_build_error}")
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
    "PATH=${KFCORE_CONSUMER_INSTALL_PREFIX}/bin;$ENV{TENSORRT_ROOT}/bin;$ENV{CUDA_TOOLKIT_ROOT}/bin;$ENV{TURBOUTILS_ROOT}/bin;$ENV{SystemRoot}/System32;$ENV{SystemRoot}")
else()
  set(_runtime_environment
    "LD_LIBRARY_PATH=${KFCORE_CONSUMER_INSTALL_PREFIX}/lib:$ENV{TENSORRT_ROOT}/lib:$ENV{CUDA_TOOLKIT_ROOT}/lib64:$ENV{TURBOUTILS_ROOT}/lib")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "${_runtime_environment}" "${_consumer_executable}"
  RESULT_VARIABLE _run_result
  OUTPUT_VARIABLE _run_output
  ERROR_VARIABLE _run_error)
if(NOT _run_result EQUAL 0)
  message(FATAL_ERROR
    "Face application installed consumer failed:\n${_run_output}\n${_run_error}")
endif()
