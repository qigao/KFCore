if(NOT DEFINED KFCORE_BINARY_DIR OR NOT DEFINED KFCORE_CONSUMER_SOURCE_DIR OR
   NOT DEFINED KFCORE_CONSUMER_BINARY_DIR OR NOT DEFINED KFCORE_CONSUMER_INSTALL_PREFIX OR
   NOT DEFINED KFCORE_CONSUMER_KFCORE_DIR OR NOT DEFINED KFCORE_CONSUMER_TRACKERS_DIR OR
   NOT DEFINED KFCORE_CONSUMER_ENABLE_SANITIZER_ADDRESS OR
   NOT DEFINED KFCORE_CONSUMER_BUILD_CONFIG OR
   NOT DEFINED KFCORE_CONSUMER_MODE OR
   NOT DEFINED KFCORE_CONSUMER_HAS_TENSORRT_YOLO OR
   NOT DEFINED KFCORE_CONSUMER_HAS_YOLO_OPENCV)
  message(FATAL_ERROR "Installed-consumer test paths are required")
endif()
if(NOT KFCORE_CONSUMER_MODE MATCHES
   "^(kfcore_first|trackers_first|dependency_first|repeated_kfcore)$")
  message(FATAL_ERROR "Unknown installed-consumer mode: ${KFCORE_CONSUMER_MODE}")
endif()
foreach(_feature IN ITEMS
    KFCORE_CONSUMER_HAS_TENSORRT_YOLO
    KFCORE_CONSUMER_HAS_YOLO_OPENCV)
  if(NOT "${${_feature}}" STREQUAL "ON" AND
     NOT "${${_feature}}" STREQUAL "OFF")
    message(FATAL_ERROR "${_feature} must be an explicit ON or OFF value")
  endif()
endforeach()
if(NOT "${KFCORE_CONSUMER_ENABLE_SANITIZER_ADDRESS}" STREQUAL "ON" AND
   NOT "${KFCORE_CONSUMER_ENABLE_SANITIZER_ADDRESS}" STREQUAL "OFF")
  message(FATAL_ERROR
    "KFCORE_CONSUMER_ENABLE_SANITIZER_ADDRESS must be an explicit ON or OFF value")
endif()
if("${KFCORE_CONSUMER_BUILD_CONFIG}" STREQUAL "" OR
   NOT "${KFCORE_CONSUMER_BUILD_CONFIG}" MATCHES "^[A-Za-z0-9_.+-]+$")
  message(FATAL_ERROR
    "KFCORE_CONSUMER_BUILD_CONFIG must name a non-empty CMake configuration")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../SafeTestDirectory.cmake")
set(_consumer_test_root "${KFCORE_BINARY_DIR}/tests")
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

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${KFCORE_CONSUMER_SOURCE_DIR}"
    -B "${KFCORE_CONSUMER_BINARY_DIR}"
    "-DKFCore_DIR=${KFCORE_CONSUMER_KFCORE_DIR}"
    "-Dtrackers_DIR=${KFCORE_CONSUMER_TRACKERS_DIR}"
    "-DKFCORE_EXPECTED_KFCORE_DIR=${KFCORE_CONSUMER_KFCORE_DIR}"
    "-DKFCORE_EXPECTED_TRACKERS_DIR=${KFCORE_CONSUMER_TRACKERS_DIR}"
    "-DKFCORE_CONSUMER_MODE=${KFCORE_CONSUMER_MODE}"
    "-DKFCORE_CONSUMER_HAS_TENSORRT_YOLO=${KFCORE_CONSUMER_HAS_TENSORRT_YOLO}"
    "-DKFCORE_CONSUMER_HAS_YOLO_OPENCV=${KFCORE_CONSUMER_HAS_YOLO_OPENCV}"
    "-DCMAKE_BUILD_TYPE=${KFCORE_CONSUMER_BUILD_CONFIG}"
    -DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE
    -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE
    -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=TRUE
  RESULT_VARIABLE _configure_result
  OUTPUT_VARIABLE _configure_output
  ERROR_VARIABLE _configure_error)
if(NOT _configure_result EQUAL 0)
  message(FATAL_ERROR "Installed consumer configure failed:\n${_configure_output}\n${_configure_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${KFCORE_CONSUMER_BINARY_DIR}"
    --config "${KFCORE_CONSUMER_BUILD_CONFIG}"
  RESULT_VARIABLE _build_result
  OUTPUT_VARIABLE _build_output
  ERROR_VARIABLE _build_error)
if(NOT _build_result EQUAL 0)
  message(FATAL_ERROR "Installed consumer build failed:\n${_build_output}\n${_build_error}")
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

file(REAL_PATH "${KFCORE_BINARY_DIR}" _kfcore_binary_dir)
file(REAL_PATH "${KFCORE_CONSUMER_INSTALL_PREFIX}" _consumer_install_prefix)
if(WIN32)
  set(_consumer_runtime_path
    "${_consumer_install_prefix}/bin;$ENV{SystemRoot}/System32;$ENV{SystemRoot}")
  if("${KFCORE_CONSUMER_ENABLE_SANITIZER_ADDRESS}" STREQUAL "ON")
    unset(_asan_runtime CACHE)
    find_file(_asan_runtime
      NAMES clang_rt.asan_dynamic-x86_64.dll
      PATHS $ENV{PATH}
      NO_DEFAULT_PATH)
    if(NOT _asan_runtime)
      message(FATAL_ERROR "The Visual Studio AddressSanitizer runtime is required when ENABLE_SANITIZER_ADDRESS is ON")
    endif()
    get_filename_component(_asan_runtime_dir "${_asan_runtime}" DIRECTORY)
    set(_consumer_runtime_path
      "${_consumer_install_prefix}/bin;${_asan_runtime_dir};$ENV{SystemRoot}/System32;$ENV{SystemRoot}")
  endif()
  set(_runtime_environment "PATH=${_consumer_runtime_path}")
else()
  set(_consumer_runtime_path "${_consumer_install_prefix}/lib")
  set(_runtime_environment "LD_LIBRARY_PATH=${_consumer_runtime_path}")
endif()
string(FIND "${_consumer_runtime_path}" "${_kfcore_binary_dir}/bin" _build_bin_index)
if(NOT _build_bin_index EQUAL -1)
  message(FATAL_ERROR "Consumer runtime path must not contain the KFCore build bin directory")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "${_runtime_environment}"
    "${_consumer_executable}"
  RESULT_VARIABLE _run_result
  OUTPUT_VARIABLE _run_output
  ERROR_VARIABLE _run_error)
if(NOT _run_result EQUAL 0)
  message(FATAL_ERROR "Installed consumer failed:\n${_run_output}\n${_run_error}")
endif()
