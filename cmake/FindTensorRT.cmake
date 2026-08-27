include(FindPackageHandleStandardArgs)

if(NOT DEFINED ENV{TENSORRT_ROOT} OR "$ENV{TENSORRT_ROOT}" STREQUAL "")
  message(FATAL_ERROR
    "TENSORRT_ROOT is required when KFCore TensorRT support is enabled")
endif()

if(NOT IS_DIRECTORY "$ENV{TENSORRT_ROOT}")
  message(FATAL_ERROR
    "TENSORRT_ROOT is not a directory: $ENV{TENSORRT_ROOT}")
endif()

file(REAL_PATH "$ENV{TENSORRT_ROOT}" _TensorRT_root)
cmake_path(SET _TensorRT_root NORMALIZE "${_TensorRT_root}")

function(_tensorrt_require_contained resolved_path description)
  file(REAL_PATH "${resolved_path}" _resolved_real)
  cmake_path(SET _resolved_real NORMALIZE "${_resolved_real}")
  cmake_path(IS_PREFIX _TensorRT_root "${_resolved_real}" NORMALIZE _is_contained)
  if(NOT _is_contained)
    message(FATAL_ERROR
      "TensorRT ${description} resolved outside TENSORRT_ROOT: ${_resolved_real}")
  endif()
endfunction()

unset(TensorRT_INCLUDE_DIR CACHE)
find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS
    "${_TensorRT_root}/include"
    "${_TensorRT_root}/targets/x86_64-linux-gnu/include"
  NO_DEFAULT_PATH)

if(NOT TensorRT_INCLUDE_DIR)
  message(FATAL_ERROR
    "TENSORRT_ROOT does not contain NvInfer.h: ${_TensorRT_root}")
endif()
_tensorrt_require_contained("${TensorRT_INCLUDE_DIR}" "include directory")

set(_TensorRT_version_header "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
if(NOT EXISTS "${_TensorRT_version_header}")
  message(FATAL_ERROR
    "TENSORRT_ROOT does not contain NvInferVersion.h: ${TensorRT_INCLUDE_DIR}")
endif()
_tensorrt_require_contained("${_TensorRT_version_header}" "version header")

if(NOT EXISTS "${TensorRT_INCLUDE_DIR}/NvInferPlugin.h")
  message(FATAL_ERROR
    "TENSORRT_ROOT does not contain NvInferPlugin.h: ${TensorRT_INCLUDE_DIR}")
endif()
_tensorrt_require_contained("${TensorRT_INCLUDE_DIR}/NvInferPlugin.h" "plugin header")

file(READ "${_TensorRT_version_header}" _TensorRT_version_contents)

function(_tensorrt_read_version_component component output_variable)
  string(REGEX MATCH
    "#[ \t]*define[ \t]+NV_TENSORRT_${component}[ \t]+([A-Za-z_][A-Za-z0-9_]*|[0-9]+)"
    _TensorRT_component_match
    "${_TensorRT_version_contents}")
  if(NOT _TensorRT_component_match)
    message(FATAL_ERROR
      "NvInferVersion.h does not define a parseable NV_TENSORRT_${component}")
  endif()

  set(_TensorRT_component_value "${CMAKE_MATCH_1}")
  if(NOT _TensorRT_component_value MATCHES "^[0-9]+$")
    string(REGEX MATCH
      "#[ \t]*define[ \t]+${_TensorRT_component_value}[ \t]+([0-9]+)"
      _TensorRT_indirect_match
      "${_TensorRT_version_contents}")
    if(NOT _TensorRT_indirect_match)
      message(FATAL_ERROR
        "NvInferVersion.h defines NV_TENSORRT_${component} through an unparseable macro: "
        "${_TensorRT_component_value}")
    endif()
    set(_TensorRT_component_value "${CMAKE_MATCH_1}")
  endif()

  set(${output_variable} "${_TensorRT_component_value}" PARENT_SCOPE)
endfunction()

foreach(_component IN ITEMS MAJOR MINOR PATCH)
  _tensorrt_read_version_component(${_component} TensorRT_VERSION_${_component})
endforeach()

if(NOT TensorRT_VERSION_MAJOR EQUAL 10 AND NOT TensorRT_VERSION_MAJOR EQUAL 11)
  message(FATAL_ERROR
    "Unsupported TensorRT major version ${TensorRT_VERSION_MAJOR}; expected 10 or 11")
endif()
set(TensorRT_VERSION
  "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")

set(_TensorRT_library_paths
  "${_TensorRT_root}/lib"
  "${_TensorRT_root}/lib64"
  "${_TensorRT_root}/lib/x64"
  "${_TensorRT_root}/targets/x86_64-linux-gnu/lib")

function(_tensorrt_validate_library_abi target library_stem library_path)
  file(REAL_PATH "${library_path}" _TensorRT_library_real)
  get_filename_component(_TensorRT_filename "${_TensorRT_library_real}" NAME)
  set(_TensorRT_expected_filenames
    "${library_stem}.lib"
    "${library_stem}_${TensorRT_VERSION_MAJOR}.lib"
    "${library_stem}.dll"
    "${library_stem}_${TensorRT_VERSION_MAJOR}.dll"
    "lib${library_stem}.a"
    "lib${library_stem}_${TensorRT_VERSION_MAJOR}.a"
    "lib${library_stem}.so"
    "lib${library_stem}_${TensorRT_VERSION_MAJOR}.so"
    "lib${library_stem}.dylib"
    "lib${library_stem}_${TensorRT_VERSION_MAJOR}.dylib")
  if(_TensorRT_filename IN_LIST _TensorRT_expected_filenames OR
     _TensorRT_filename MATCHES
       "^lib${library_stem}\\.so\\.${TensorRT_VERSION_MAJOR}(\\.[0-9]+)*$" OR
     _TensorRT_filename MATCHES
       "^lib${library_stem}\\.${TensorRT_VERSION_MAJOR}(\\.[0-9]+)*\\.dylib$")
    return()
  endif()
  message(FATAL_ERROR
    "${target} imported location does not match TensorRT major version "
    "${TensorRT_VERSION_MAJOR}: ${_TensorRT_library_real}")
endfunction()

function(_tensorrt_validate_existing_target target library_stem output_library)
  get_target_property(_TensorRT_imported "${target}" IMPORTED)
  if(NOT _TensorRT_imported)
    message(FATAL_ERROR "${target} is not a verifiable imported target")
  endif()

  get_target_property(_TensorRT_includes "${target}" INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _TensorRT_includes)
    message(FATAL_ERROR "${target} has no verifiable include directory")
  endif()
  set(_TensorRT_has_expected_include FALSE)
  foreach(_TensorRT_include IN LISTS _TensorRT_includes)
    if(_TensorRT_include MATCHES "\\$<")
      message(FATAL_ERROR "${target} has an unverified generator-expression include")
    endif()
    if(NOT IS_DIRECTORY "${_TensorRT_include}")
      message(FATAL_ERROR "${target} include directory does not exist: ${_TensorRT_include}")
    endif()
    _tensorrt_require_contained("${_TensorRT_include}" "${target} include directory")
    if(EXISTS "${_TensorRT_include}/NvInfer.h" AND
       EXISTS "${_TensorRT_include}/NvInferVersion.h")
      set(_TensorRT_has_expected_include TRUE)
    endif()
  endforeach()
  if(NOT _TensorRT_has_expected_include)
    message(FATAL_ERROR "${target} does not expose the validated TensorRT headers")
  endif()

  set(_TensorRT_location_properties IMPORTED_LOCATION IMPORTED_IMPLIB)
  get_target_property(_TensorRT_configurations "${target}" IMPORTED_CONFIGURATIONS)
  list(APPEND _TensorRT_configurations DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  list(REMOVE_DUPLICATES _TensorRT_configurations)
  foreach(_TensorRT_configuration IN LISTS _TensorRT_configurations)
    string(TOUPPER "${_TensorRT_configuration}" _TensorRT_configuration_upper)
    list(APPEND _TensorRT_location_properties
      "IMPORTED_LOCATION_${_TensorRT_configuration_upper}"
      "IMPORTED_IMPLIB_${_TensorRT_configuration_upper}")
  endforeach()
  list(REMOVE_DUPLICATES _TensorRT_location_properties)

  set(_TensorRT_representative_library)
  foreach(_TensorRT_property IN LISTS _TensorRT_location_properties)
    get_target_property(_TensorRT_locations "${target}" "${_TensorRT_property}")
    if(NOT _TensorRT_locations)
      continue()
    endif()
    foreach(_TensorRT_location IN LISTS _TensorRT_locations)
      if(_TensorRT_location MATCHES "\\$<")
        message(FATAL_ERROR "${target} has an unverified generator-expression location")
      endif()
      if(NOT EXISTS "${_TensorRT_location}")
        message(FATAL_ERROR "${target} imported location does not exist: ${_TensorRT_location}")
      endif()
      _tensorrt_require_contained("${_TensorRT_location}" "${target} imported location")
      _tensorrt_validate_library_abi(
        "${target}" "${library_stem}" "${_TensorRT_location}")
      if(NOT _TensorRT_representative_library)
        set(_TensorRT_representative_library "${_TensorRT_location}")
      endif()
    endforeach()
  endforeach()
  if(NOT _TensorRT_representative_library)
    message(FATAL_ERROR "${target} has no verifiable imported library location")
  endif()
  set(${output_library} "${_TensorRT_representative_library}" PARENT_SCOPE)
endfunction()

set(_TensorRT_reuse_targets FALSE)
if(TARGET TensorRT::nvinfer OR TARGET TensorRT::nvinfer_plugin)
  if(NOT TARGET TensorRT::nvinfer OR NOT TARGET TensorRT::nvinfer_plugin)
    message(FATAL_ERROR
      "FindTensorRT rejects partial imported targets; both TensorRT::nvinfer and "
      "TensorRT::nvinfer_plugin are required")
  endif()
  _tensorrt_validate_existing_target(
    TensorRT::nvinfer nvinfer TensorRT_NVINFER_LIBRARY)
  _tensorrt_validate_existing_target(
    TensorRT::nvinfer_plugin nvinfer_plugin TensorRT_NVINFER_PLUGIN_LIBRARY)
  set(_TensorRT_reuse_targets TRUE)
else()
  unset(TensorRT_NVINFER_LIBRARY CACHE)
  find_library(TensorRT_NVINFER_LIBRARY
    NAMES "nvinfer_${TensorRT_VERSION_MAJOR}" nvinfer
    PATHS ${_TensorRT_library_paths}
    NO_DEFAULT_PATH)

  unset(TensorRT_NVINFER_PLUGIN_LIBRARY CACHE)
  find_library(TensorRT_NVINFER_PLUGIN_LIBRARY
    NAMES "nvinfer_plugin_${TensorRT_VERSION_MAJOR}" nvinfer_plugin
    PATHS ${_TensorRT_library_paths}
    NO_DEFAULT_PATH)

  if(NOT TensorRT_NVINFER_LIBRARY)
    message(FATAL_ERROR
      "TENSORRT_ROOT does not contain the nvinfer library: ${_TensorRT_root}")
  endif()
  _tensorrt_require_contained("${TensorRT_NVINFER_LIBRARY}" "nvinfer library")

  if(NOT TensorRT_NVINFER_PLUGIN_LIBRARY)
    message(FATAL_ERROR
      "TENSORRT_ROOT does not contain the nvinfer_plugin library: ${_TensorRT_root}")
  endif()
  _tensorrt_require_contained(
    "${TensorRT_NVINFER_PLUGIN_LIBRARY}" "nvinfer_plugin library")
endif()

find_package_handle_standard_args(TensorRT
  REQUIRED_VARS
    TensorRT_INCLUDE_DIR
    TensorRT_NVINFER_LIBRARY
    TensorRT_NVINFER_PLUGIN_LIBRARY
  VERSION_VAR TensorRT_VERSION)

if(TensorRT_FOUND AND NOT _TensorRT_reuse_targets)
  add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
  set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
  add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
  set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
    IMPORTED_LOCATION "${TensorRT_NVINFER_PLUGIN_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES TensorRT::nvinfer)
endif()

mark_as_advanced(
  TensorRT_INCLUDE_DIR
  TensorRT_NVINFER_LIBRARY
  TensorRT_NVINFER_PLUGIN_LIBRARY)
