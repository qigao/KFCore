include(FindPackageHandleStandardArgs)

if(NOT DEFINED ENV{ONNXRUNTIME_ROOT} OR "$ENV{ONNXRUNTIME_ROOT}" STREQUAL "")
  message(FATAL_ERROR
    "ONNXRUNTIME_ROOT is required when KFCore ONNX Runtime support is enabled")
endif()
if(NOT IS_DIRECTORY "$ENV{ONNXRUNTIME_ROOT}")
  message(FATAL_ERROR "ONNXRUNTIME_ROOT is not a directory: $ENV{ONNXRUNTIME_ROOT}")
endif()

file(REAL_PATH "$ENV{ONNXRUNTIME_ROOT}" _ONNXRuntime_root)
cmake_path(SET _ONNXRuntime_root NORMALIZE "${_ONNXRuntime_root}")

function(_onnxruntime_require_contained path description)
  file(REAL_PATH "${path}" _resolved)
  cmake_path(SET _resolved NORMALIZE "${_resolved}")
  cmake_path(IS_PREFIX _ONNXRuntime_root "${_resolved}" NORMALIZE _contained)
  if(NOT _contained)
    message(FATAL_ERROR
      "ONNX Runtime ${description} resolved outside ONNXRUNTIME_ROOT: ${_resolved}")
  endif()
endfunction()

unset(_ONNXRuntime_include CACHE)
find_path(_ONNXRuntime_include
  NAMES onnxruntime_cxx_api.h
  PATHS "${_ONNXRuntime_root}/include"
  NO_DEFAULT_PATH)
if(NOT _ONNXRuntime_include)
  message(FATAL_ERROR
    "ONNXRUNTIME_ROOT does not contain include/onnxruntime_cxx_api.h")
endif()
_onnxruntime_require_contained("${_ONNXRuntime_include}" "include directory")

unset(_ONNXRuntime_library CACHE)
find_library(_ONNXRuntime_library
  NAMES onnxruntime
  PATHS "${_ONNXRuntime_root}/lib" "${_ONNXRuntime_root}/lib64"
  NO_DEFAULT_PATH)
if(NOT _ONNXRuntime_library)
  message(FATAL_ERROR "ONNXRUNTIME_ROOT does not contain the onnxruntime library")
endif()
_onnxruntime_require_contained("${_ONNXRuntime_library}" "library")

if(KFCORE_ENABLE_ONNX_CUDA OR KFCore_ONNX_CUDA_ENABLED)
  foreach(_provider IN ITEMS cuda shared)
    unset(_ONNXRuntime_provider CACHE)
    if(WIN32)
      set(_provider_filename "onnxruntime_providers_${_provider}.dll")
    else()
      set(_provider_filename "libonnxruntime_providers_${_provider}.so")
    endif()
    find_file(_ONNXRuntime_provider NAMES "${_provider_filename}"
      PATHS "${_ONNXRuntime_root}/lib" "${_ONNXRuntime_root}/lib64"
      NO_DEFAULT_PATH)
    if(NOT _ONNXRuntime_provider)
      message(FATAL_ERROR "ONNX CUDA requires ${_provider_filename} in ONNXRUNTIME_ROOT; a CPU SDK is insufficient")
    endif()
    _onnxruntime_require_contained("${_ONNXRuntime_provider}" "${_provider} provider")
  endforeach()
endif()

if(EXISTS "${_ONNXRuntime_root}/VERSION_NUMBER")
  file(STRINGS "${_ONNXRuntime_root}/VERSION_NUMBER" ONNXRuntime_VERSION LIMIT_COUNT 1)
endif()

if(NOT TARGET ONNXRuntime::ONNXRuntime)
  if(WIN32)
    unset(_ONNXRuntime_runtime CACHE)
    find_file(_ONNXRuntime_runtime
      NAMES onnxruntime.dll
      PATHS "${_ONNXRuntime_root}/lib" "${_ONNXRuntime_root}/bin"
      NO_DEFAULT_PATH)
    if(NOT _ONNXRuntime_runtime)
      message(FATAL_ERROR "ONNXRUNTIME_ROOT does not contain onnxruntime.dll")
    endif()
    _onnxruntime_require_contained("${_ONNXRuntime_runtime}" "runtime")
    file(RELATIVE_PATH ONNXRuntime_RUNTIME_RELATIVE_PATH
      "${_ONNXRuntime_root}" "${_ONNXRuntime_runtime}")
    add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
      IMPORTED_IMPLIB "${_ONNXRuntime_library}"
      IMPORTED_LOCATION "${_ONNXRuntime_runtime}"
      INTERFACE_INCLUDE_DIRECTORIES "${_ONNXRuntime_include}")
  else()
    add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
      IMPORTED_LOCATION "${_ONNXRuntime_library}"
      INTERFACE_INCLUDE_DIRECTORIES "${_ONNXRuntime_include}")
  endif()
endif()

set(ONNXRuntime_INCLUDE_DIR "${_ONNXRuntime_include}")
set(ONNXRuntime_LIBRARY "${_ONNXRuntime_library}")
find_package_handle_standard_args(ONNXRuntime
  REQUIRED_VARS ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY
  VERSION_VAR ONNXRuntime_VERSION)

mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
