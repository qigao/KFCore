if(KFCORE_NINJA_MISSING_PROBE)
  set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)
  set(CMAKE_FIND_USE_CMAKE_PATH FALSE)
  set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)
  set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)

  include("${CMAKE_CURRENT_LIST_DIR}/../RequireNinja.cmake")
  kfcore_require_ninja(_ninja_executable)
  return()
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -DKFCORE_NINJA_MISSING_PROBE=TRUE
    -P "${CMAKE_CURRENT_LIST_FILE}"
  RESULT_VARIABLE _probe_result
  OUTPUT_VARIABLE _probe_output
  ERROR_VARIABLE _probe_error)
if(_probe_result EQUAL 0)
  message(FATAL_ERROR "Missing-Ninja probe unexpectedly succeeded")
endif()
set(_probe_log "${_probe_output}\n${_probe_error}")
if(NOT _probe_log MATCHES
   "TensorRT runtime installed-consumer test requires Ninja")
  message(FATAL_ERROR
    "Missing-Ninja probe did not report the actionable diagnostic:\n${_probe_log}")
endif()
