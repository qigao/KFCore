if(NOT DEFINED KFCORE_BINARY_DIR OR
   NOT DEFINED KFCORE_CONSUMER_SOURCE_DIR OR
   NOT DEFINED KFCORE_CONSUMER_BINARY_DIR OR
   NOT DEFINED KFCORE_CONSUMER_INSTALL_PREFIX OR
   NOT DEFINED KFCORE_CONSUMER_KFCORE_DIR OR
   NOT DEFINED KFCORE_CONSUMER_BUILD_CONFIG OR
   NOT DEFINED KFCORE_CONSUMER_NINJA_EXECUTABLE OR
   NOT DEFINED KFCORE_CONSUMER_EXPECT_CPU OR
   NOT DEFINED KFCORE_CONSUMER_EXPECT_TENSORRT OR
   NOT DEFINED KFCORE_CONSUMER_EXPECT_HAND_INTERACTION)
  message(FATAL_ERROR "Vision-model installed-consumer test paths are required")
endif()
if(NOT IS_ABSOLUTE "${KFCORE_CONSUMER_NINJA_EXECUTABLE}" OR
   NOT EXISTS "${KFCORE_CONSUMER_NINJA_EXECUTABLE}" OR
   IS_DIRECTORY "${KFCORE_CONSUMER_NINJA_EXECUTABLE}")
  message(FATAL_ERROR "Vision-model consumer requires a resolved Ninja executable")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../SafeTestDirectory.cmake")
set(_consumer_test_root "${KFCORE_BINARY_DIR}/tests")
file(MAKE_DIRECTORY "${_consumer_test_root}")
kfcore_reset_test_directory(
  "${_consumer_test_root}" "${KFCORE_CONSUMER_BINARY_DIR}")
kfcore_reset_test_directory(
  "${_consumer_test_root}" "${KFCORE_CONSUMER_INSTALL_PREFIX}")

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

set(_configure_arguments
  -G Ninja
  -S "${KFCORE_CONSUMER_SOURCE_DIR}"
  -B "${KFCORE_CONSUMER_BINARY_DIR}"
  "-DCMAKE_MAKE_PROGRAM=${KFCORE_CONSUMER_NINJA_EXECUTABLE}"
  "-DKFCore_DIR=${KFCORE_CONSUMER_KFCORE_DIR}"
  "-DKFCORE_EXPECTED_KFCORE_DIR=${KFCORE_CONSUMER_KFCORE_DIR}"
  "-DKFCORE_EXPECT_CPU=${KFCORE_CONSUMER_EXPECT_CPU}"
  "-DKFCORE_EXPECT_TENSORRT=${KFCORE_CONSUMER_EXPECT_TENSORRT}"
  "-DKFCORE_EXPECT_HAND_INTERACTION=${KFCORE_CONSUMER_EXPECT_HAND_INTERACTION}"
  "-DCMAKE_BUILD_TYPE=${KFCORE_CONSUMER_BUILD_CONFIG}"
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE
  -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE
  -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=TRUE)
if(KFCORE_CONSUMER_EXPECT_CPU)
  list(APPEND _configure_arguments
    "-DONNXRUNTIME_ROOT=${KFCORE_CONSUMER_ONNXRUNTIME_ROOT}")
endif()
if(KFCORE_CONSUMER_EXPECT_TENSORRT)
  list(APPEND _configure_arguments
    "-DTENSORRT_ROOT=${KFCORE_CONSUMER_TENSORRT_ROOT}"
    "-DCUDAToolkit_ROOT=${KFCORE_CONSUMER_CUDATOOLKIT_ROOT}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" ${_configure_arguments}
  RESULT_VARIABLE _configure_result
  OUTPUT_VARIABLE _configure_output
  ERROR_VARIABLE _configure_error)
if(NOT _configure_result EQUAL 0)
  message(FATAL_ERROR
    "Vision-model installed consumer configure failed:\n"
    "${_configure_output}\n${_configure_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${KFCORE_CONSUMER_BINARY_DIR}"
    --config "${KFCORE_CONSUMER_BUILD_CONFIG}"
  RESULT_VARIABLE _build_result
  OUTPUT_VARIABLE _build_output
  ERROR_VARIABLE _build_error)
if(NOT _build_result EQUAL 0)
  message(FATAL_ERROR
    "Vision-model installed consumer build failed:\n${_build_output}\n${_build_error}")
endif()

set(_consumer_target_file
  "${KFCORE_CONSUMER_BINARY_DIR}/consumer-target-file-${KFCORE_CONSUMER_BUILD_CONFIG}.txt")
if(NOT EXISTS "${_consumer_target_file}")
  message(FATAL_ERROR "Vision-model consumer did not generate its target file path")
endif()
file(READ "${_consumer_target_file}" _consumer_executable)
string(STRIP "${_consumer_executable}" _consumer_executable)
if(NOT EXISTS "${_consumer_executable}")
  message(FATAL_ERROR "Vision-model consumer executable does not exist")
endif()

if(WIN32)
  set(_consumer_runtime_path "${KFCORE_CONSUMER_INSTALL_PREFIX}/bin")
  if(KFCORE_CONSUMER_EXPECT_CPU)
    string(APPEND _consumer_runtime_path
      ";${KFCORE_CONSUMER_ONNXRUNTIME_ROOT}/lib")
  endif()
  if(KFCORE_CONSUMER_EXPECT_TENSORRT)
    string(APPEND _consumer_runtime_path
      ";${KFCORE_CONSUMER_TENSORRT_ROOT}/bin;${KFCORE_CONSUMER_CUDATOOLKIT_ROOT}/bin")
  endif()
  string(APPEND _consumer_runtime_path ";$ENV{SystemRoot}/System32;$ENV{SystemRoot}")
  set(_runtime_environment "PATH=${_consumer_runtime_path}")
else()
  set(_consumer_runtime_path "${KFCORE_CONSUMER_INSTALL_PREFIX}/lib")
  if(KFCORE_CONSUMER_EXPECT_CPU)
    string(APPEND _consumer_runtime_path
      ":${KFCORE_CONSUMER_ONNXRUNTIME_ROOT}/lib")
  endif()
  if(KFCORE_CONSUMER_EXPECT_TENSORRT)
    string(APPEND _consumer_runtime_path
      ":${KFCORE_CONSUMER_TENSORRT_ROOT}/lib:${KFCORE_CONSUMER_TENSORRT_ROOT}/lib64:${KFCORE_CONSUMER_CUDATOOLKIT_ROOT}/lib64")
  endif()
  set(_runtime_environment "LD_LIBRARY_PATH=${_consumer_runtime_path}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "${_runtime_environment}" "${_consumer_executable}"
  RESULT_VARIABLE _run_result
  OUTPUT_VARIABLE _run_output
  ERROR_VARIABLE _run_error)
if(NOT _run_result EQUAL 0)
  message(FATAL_ERROR
    "Vision-model installed consumer failed:\n${_run_output}\n${_run_error}")
endif()
