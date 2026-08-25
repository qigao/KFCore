if(NOT DEFINED KFCORE_BINARY_DIR OR NOT DEFINED KFCORE_CONSUMER_SOURCE_DIR OR
   NOT DEFINED KFCORE_CONSUMER_BINARY_DIR OR NOT DEFINED KFCORE_CONSUMER_INSTALL_PREFIX)
  message(FATAL_ERROR "Installed-consumer test paths are required")
endif()

file(REMOVE_RECURSE "${KFCORE_CONSUMER_BINARY_DIR}" "${KFCORE_CONSUMER_INSTALL_PREFIX}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${KFCORE_BINARY_DIR}"
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
    "-DCMAKE_PREFIX_PATH=${KFCORE_CONSUMER_INSTALL_PREFIX}"
  RESULT_VARIABLE _configure_result
  OUTPUT_VARIABLE _configure_output
  ERROR_VARIABLE _configure_error)
if(NOT _configure_result EQUAL 0)
  message(FATAL_ERROR "Installed consumer configure failed:\n${_configure_output}\n${_configure_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${KFCORE_CONSUMER_BINARY_DIR}"
  RESULT_VARIABLE _build_result
  OUTPUT_VARIABLE _build_output
  ERROR_VARIABLE _build_error)
if(NOT _build_result EQUAL 0)
  message(FATAL_ERROR "Installed consumer build failed:\n${_build_output}\n${_build_error}")
endif()

file(GLOB_RECURSE _consumer_executables
  "${KFCORE_CONSUMER_BINARY_DIR}/yolo_tracking_installed_consumer.exe")
list(LENGTH _consumer_executables _consumer_executable_count)
if(NOT _consumer_executable_count EQUAL 1)
  message(FATAL_ERROR "Expected one installed-consumer executable, found ${_consumer_executable_count}")
endif()

list(GET _consumer_executables 0 _consumer_executable)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "PATH=${KFCORE_CONSUMER_INSTALL_PREFIX}/bin;$ENV{PATH}"
    "${_consumer_executable}"
  RESULT_VARIABLE _run_result
  OUTPUT_VARIABLE _run_output
  ERROR_VARIABLE _run_error)
if(NOT _run_result EQUAL 0)
  message(FATAL_ERROR "Installed consumer failed:\n${_run_output}\n${_run_error}")
endif()
