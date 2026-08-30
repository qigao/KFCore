include(FindPackageHandleStandardArgs)

set(_OpenCVLite_supported_components core imgproc imgcodecs highgui)
if(OpenCVLite_FIND_COMPONENTS)
  set(_OpenCVLite_requested_components ${OpenCVLite_FIND_COMPONENTS})
else()
  set(_OpenCVLite_requested_components core imgproc)
  set(OpenCVLite_FIND_COMPONENTS ${_OpenCVLite_requested_components})
  foreach(_OpenCVLite_component IN LISTS _OpenCVLite_requested_components)
    set(OpenCVLite_FIND_REQUIRED_${_OpenCVLite_component} ${OpenCVLite_FIND_REQUIRED})
  endforeach()
endif()
list(LENGTH _OpenCVLite_requested_components _OpenCVLite_requested_count)
if(_OpenCVLite_requested_count EQUAL 0)
  message(FATAL_ERROR "FindOpenCVLite requires at least one component")
endif()
set(_OpenCVLite_unique_components ${_OpenCVLite_requested_components})
list(REMOVE_DUPLICATES _OpenCVLite_unique_components)
list(LENGTH _OpenCVLite_unique_components _OpenCVLite_unique_count)
if(NOT _OpenCVLite_unique_count EQUAL _OpenCVLite_requested_count)
  message(FATAL_ERROR "FindOpenCVLite rejects duplicate requested components")
endif()
foreach(_OpenCVLite_component IN LISTS _OpenCVLite_requested_components)
  if(NOT _OpenCVLite_component IN_LIST _OpenCVLite_supported_components)
    message(FATAL_ERROR "FindOpenCVLite does not support component: ${_OpenCVLite_component}")
  endif()
endforeach()

if(NOT TARGET OpenCVLite::core AND NOT TARGET OpenCVLite::imgproc AND
   NOT TARGET OpenCVLite::imgcodecs AND NOT TARGET OpenCVLite::highgui)
  foreach(_OpenCVLite_predefined IN ITEMS
      OpenCVLite_INCLUDE_DIR
      OpenCVLite_CORE_RELEASE_LIBRARY
      OpenCVLite_CORE_DEBUG_LIBRARY
      OpenCVLite_IMGPROC_RELEASE_LIBRARY
      OpenCVLite_IMGPROC_DEBUG_LIBRARY
      OpenCVLite_IMGCODECS_RELEASE_LIBRARY
      OpenCVLite_IMGCODECS_DEBUG_LIBRARY
      OpenCVLite_HIGHGUI_RELEASE_LIBRARY
      OpenCVLite_HIGHGUI_DEBUG_LIBRARY)
    if(DEFINED ${_OpenCVLite_predefined} OR DEFINED CACHE{${_OpenCVLite_predefined}})
      message(FATAL_ERROR
        "FindOpenCVLite refuses a predefined result: ${_OpenCVLite_predefined}")
    endif()
  endforeach()
endif()

if(NOT DEFINED ENV{OPENCV_LITE_ROOT} OR "$ENV{OPENCV_LITE_ROOT}" STREQUAL "")
  message(FATAL_ERROR "OPENCV_LITE_ROOT is required by KFCore OpenCV Lite support")
endif()
if(NOT IS_DIRECTORY "$ENV{OPENCV_LITE_ROOT}")
  message(FATAL_ERROR "OPENCV_LITE_ROOT is not a directory: $ENV{OPENCV_LITE_ROOT}")
endif()

file(REAL_PATH "$ENV{OPENCV_LITE_ROOT}" _OpenCVLite_root)
cmake_path(SET _OpenCVLite_root NORMALIZE "${_OpenCVLite_root}")

function(_opencv_lite_require_contained resolved_path description)
  file(REAL_PATH "${resolved_path}" _OpenCVLite_resolved)
  cmake_path(SET _OpenCVLite_resolved NORMALIZE "${_OpenCVLite_resolved}")
  cmake_path(IS_PREFIX _OpenCVLite_root "${_OpenCVLite_resolved}" NORMALIZE _OpenCVLite_contained)
  if(NOT _OpenCVLite_contained)
    message(FATAL_ERROR
      "OpenCV Lite ${description} resolved outside OPENCV_LITE_ROOT: ${_OpenCVLite_resolved}")
  endif()
endfunction()

unset(_OpenCVLite_discovered_include_dir)
unset(_OpenCVLite_discovered_include_dir CACHE)
set(_OpenCVLite_discovered_include_dir "_OpenCVLite_discovered_include_dir-NOTFOUND")
find_path(_OpenCVLite_discovered_include_dir
  NAMES opencv2/core.hpp
  PATHS "${_OpenCVLite_root}/include"
  NO_DEFAULT_PATH)
set(OpenCVLite_INCLUDE_DIR "${_OpenCVLite_discovered_include_dir}")
unset(_OpenCVLite_discovered_include_dir CACHE)
if(NOT OpenCVLite_INCLUDE_DIR)
  message(FATAL_ERROR "OPENCV_LITE_ROOT does not contain opencv2/core.hpp: ${_OpenCVLite_root}")
endif()
_opencv_lite_require_contained("${OpenCVLite_INCLUDE_DIR}" "include directory")

set(_OpenCVLite_version_header "${OpenCVLite_INCLUDE_DIR}/opencv2/core/version.hpp")
if(NOT EXISTS "${_OpenCVLite_version_header}")
  message(FATAL_ERROR "OPENCV_LITE_ROOT does not contain opencv2/core/version.hpp")
endif()
_opencv_lite_require_contained("${_OpenCVLite_version_header}" "version header")
file(READ "${_OpenCVLite_version_header}" _OpenCVLite_version_contents)
foreach(_OpenCVLite_version_part IN ITEMS MAJOR MINOR REVISION)
  string(REGEX MATCH
    "#[ \t]*define[ \t]+CV_VERSION_${_OpenCVLite_version_part}[ \t]+([0-9]+)"
    _OpenCVLite_version_match
    "${_OpenCVLite_version_contents}")
  if(NOT _OpenCVLite_version_match)
    message(FATAL_ERROR
      "opencv2/core/version.hpp does not define CV_VERSION_${_OpenCVLite_version_part}")
  endif()
  set(_OpenCVLite_VERSION_${_OpenCVLite_version_part} "${CMAKE_MATCH_1}")
endforeach()
set(_OpenCVLite_version_suffix
  "${_OpenCVLite_VERSION_MAJOR}${_OpenCVLite_VERSION_MINOR}${_OpenCVLite_VERSION_REVISION}")
set(OpenCVLite_VERSION
  "${_OpenCVLite_VERSION_MAJOR}.${_OpenCVLite_VERSION_MINOR}.${_OpenCVLite_VERSION_REVISION}")

function(_opencv_lite_validate_library_abi target component library_path)
  file(REAL_PATH "${library_path}" _OpenCVLite_library_real)
  get_filename_component(_OpenCVLite_filename "${_OpenCVLite_library_real}" NAME)
  set(_OpenCVLite_stems
    "opencv_${component}"
    "opencv_${component}d"
    "opencv_${component}${_OpenCVLite_version_suffix}"
    "opencv_${component}${_OpenCVLite_version_suffix}d")
  set(_OpenCVLite_expected_filenames)
  foreach(_OpenCVLite_stem IN LISTS _OpenCVLite_stems)
    list(APPEND _OpenCVLite_expected_filenames
      "${_OpenCVLite_stem}.lib"
      "${_OpenCVLite_stem}.dll"
      "lib${_OpenCVLite_stem}.a"
      "lib${_OpenCVLite_stem}.so"
      "lib${_OpenCVLite_stem}.dylib")
  endforeach()
  if(_OpenCVLite_filename IN_LIST _OpenCVLite_expected_filenames)
    return()
  endif()

  set(_OpenCVLite_soname_versions
    "${_OpenCVLite_VERSION_MAJOR}.${_OpenCVLite_VERSION_MINOR}"
    "${OpenCVLite_VERSION}"
    "${_OpenCVLite_VERSION_MAJOR}${_OpenCVLite_VERSION_MINOR}"
    "${_OpenCVLite_version_suffix}")
  set(_OpenCVLite_soname_stems
    "libopencv_${component}"
    "libopencv_${component}d")
  foreach(_OpenCVLite_stem IN LISTS _OpenCVLite_soname_stems)
    foreach(_OpenCVLite_soname_version IN LISTS _OpenCVLite_soname_versions)
      if(_OpenCVLite_filename STREQUAL
           "${_OpenCVLite_stem}.so.${_OpenCVLite_soname_version}" OR
         _OpenCVLite_filename STREQUAL
           "${_OpenCVLite_stem}.${_OpenCVLite_soname_version}.dylib")
        return()
      endif()
    endforeach()
  endforeach()
  message(FATAL_ERROR
    "${target} imported location does not match OpenCV Lite header version "
    "${OpenCVLite_VERSION}: ${_OpenCVLite_library_real}")
endfunction()

function(_opencv_lite_validate_existing_target component)
  set(_OpenCVLite_target "OpenCVLite::${component}")
  get_target_property(_OpenCVLite_imported "${_OpenCVLite_target}" IMPORTED)
  if(NOT _OpenCVLite_imported)
    message(FATAL_ERROR "${_OpenCVLite_target} is not a verifiable imported target")
  endif()

  get_target_property(_OpenCVLite_includes "${_OpenCVLite_target}"
    INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _OpenCVLite_includes)
    message(FATAL_ERROR "${_OpenCVLite_target} has no verifiable include directory")
  endif()
  set(_OpenCVLite_has_expected_include FALSE)
  foreach(_OpenCVLite_include IN LISTS _OpenCVLite_includes)
    if(_OpenCVLite_include MATCHES "\\$<")
      message(FATAL_ERROR
        "${_OpenCVLite_target} has an unverified generator-expression include")
    endif()
    if(NOT IS_DIRECTORY "${_OpenCVLite_include}")
      message(FATAL_ERROR
        "${_OpenCVLite_target} include directory does not exist: ${_OpenCVLite_include}")
    endif()
    _opencv_lite_require_contained(
      "${_OpenCVLite_include}" "${_OpenCVLite_target} include directory")
    if(EXISTS "${_OpenCVLite_include}/opencv2/core.hpp" AND
       EXISTS "${_OpenCVLite_include}/opencv2/core/version.hpp")
      set(_OpenCVLite_has_expected_include TRUE)
    endif()
  endforeach()
  if(NOT _OpenCVLite_has_expected_include)
    message(FATAL_ERROR
      "${_OpenCVLite_target} does not expose the validated OpenCV Lite headers")
  endif()

  set(_OpenCVLite_location_properties IMPORTED_LOCATION IMPORTED_IMPLIB)
  get_target_property(_OpenCVLite_configurations "${_OpenCVLite_target}"
    IMPORTED_CONFIGURATIONS)
  list(APPEND _OpenCVLite_configurations DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  list(REMOVE_DUPLICATES _OpenCVLite_configurations)
  foreach(_OpenCVLite_configuration IN LISTS _OpenCVLite_configurations)
    string(TOUPPER "${_OpenCVLite_configuration}" _OpenCVLite_configuration_upper)
    list(APPEND _OpenCVLite_location_properties
      "IMPORTED_LOCATION_${_OpenCVLite_configuration_upper}"
      "IMPORTED_IMPLIB_${_OpenCVLite_configuration_upper}")
  endforeach()
  list(REMOVE_DUPLICATES _OpenCVLite_location_properties)

  set(_OpenCVLite_has_location FALSE)
  foreach(_OpenCVLite_property IN LISTS _OpenCVLite_location_properties)
    get_target_property(_OpenCVLite_locations "${_OpenCVLite_target}"
      "${_OpenCVLite_property}")
    if(NOT _OpenCVLite_locations)
      continue()
    endif()
    foreach(_OpenCVLite_location IN LISTS _OpenCVLite_locations)
      if(_OpenCVLite_location MATCHES "\\$<")
        message(FATAL_ERROR
          "${_OpenCVLite_target} has an unverified generator-expression location")
      endif()
      if(NOT EXISTS "${_OpenCVLite_location}")
        message(FATAL_ERROR
          "${_OpenCVLite_target} imported location does not exist: ${_OpenCVLite_location}")
      endif()
      _opencv_lite_require_contained(
        "${_OpenCVLite_location}" "${_OpenCVLite_target} imported location")
      _opencv_lite_validate_library_abi(
        "${_OpenCVLite_target}" "${component}" "${_OpenCVLite_location}")
      set(_OpenCVLite_has_location TRUE)
    endforeach()
  endforeach()
  if(NOT _OpenCVLite_has_location)
    message(FATAL_ERROR
      "${_OpenCVLite_target} has no verifiable imported library location")
  endif()
endfunction()

function(_opencv_lite_find_component component)
  string(TOUPPER "${component}" _OpenCVLite_upper)
  if(TARGET OpenCVLite::${component})
    _opencv_lite_validate_existing_target("${component}")
    set(OpenCVLite_${component}_FOUND TRUE PARENT_SCOPE)
    return()
  endif()
  set(_OpenCVLite_release_names "opencv_${component}${_OpenCVLite_version_suffix}" "opencv_${component}")
  set(_OpenCVLite_debug_names "opencv_${component}${_OpenCVLite_version_suffix}d" "opencv_${component}d")
  set(_OpenCVLite_library_paths
    "${_OpenCVLite_root}/lib"
    "${_OpenCVLite_root}/lib64"
    "${_OpenCVLite_root}/lib/x64")

  unset(_OpenCVLite_discovered_release_library)
  unset(_OpenCVLite_discovered_release_library CACHE)
  set(_OpenCVLite_discovered_release_library
    "_OpenCVLite_discovered_release_library-NOTFOUND")
  find_library(_OpenCVLite_discovered_release_library
    NAMES ${_OpenCVLite_release_names}
    PATHS ${_OpenCVLite_library_paths}
    NO_DEFAULT_PATH)
  set(OpenCVLite_${_OpenCVLite_upper}_RELEASE_LIBRARY
    "${_OpenCVLite_discovered_release_library}")
  unset(_OpenCVLite_discovered_release_library CACHE)
  unset(_OpenCVLite_discovered_debug_library)
  unset(_OpenCVLite_discovered_debug_library CACHE)
  set(_OpenCVLite_discovered_debug_library
    "_OpenCVLite_discovered_debug_library-NOTFOUND")
  find_library(_OpenCVLite_discovered_debug_library
    NAMES ${_OpenCVLite_debug_names}
    PATHS ${_OpenCVLite_library_paths}
    NO_DEFAULT_PATH)
  set(OpenCVLite_${_OpenCVLite_upper}_DEBUG_LIBRARY
    "${_OpenCVLite_discovered_debug_library}")
  unset(_OpenCVLite_discovered_debug_library CACHE)

  if(NOT OpenCVLite_${_OpenCVLite_upper}_RELEASE_LIBRARY AND
     NOT OpenCVLite_${_OpenCVLite_upper}_DEBUG_LIBRARY)
    message(FATAL_ERROR
      "OPENCV_LITE_ROOT does not contain an opencv_${component} library: ${_OpenCVLite_root}")
  endif()

  foreach(_OpenCVLite_library IN ITEMS
      "${OpenCVLite_${_OpenCVLite_upper}_RELEASE_LIBRARY}"
      "${OpenCVLite_${_OpenCVLite_upper}_DEBUG_LIBRARY}")
    if(_OpenCVLite_library)
      _opencv_lite_require_contained("${_OpenCVLite_library}" "opencv_${component} library")
    endif()
  endforeach()

  if(WIN32)
    add_library(OpenCVLite::${component} SHARED IMPORTED)
  else()
    add_library(OpenCVLite::${component} UNKNOWN IMPORTED)
  endif()
  set_target_properties(OpenCVLite::${component} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OpenCVLite_INCLUDE_DIR}")

  if(WIN32)
    unset(_OpenCVLite_discovered_release_runtime)
    unset(_OpenCVLite_discovered_release_runtime CACHE)
    set(_OpenCVLite_discovered_release_runtime
      "_OpenCVLite_discovered_release_runtime-NOTFOUND")
    find_file(_OpenCVLite_discovered_release_runtime
      NAMES "opencv_${component}${_OpenCVLite_version_suffix}.dll" "opencv_${component}.dll"
      PATHS "${_OpenCVLite_root}/bin"
      NO_DEFAULT_PATH)
    set(_OpenCVLite_${_OpenCVLite_upper}_RELEASE_RUNTIME
      "${_OpenCVLite_discovered_release_runtime}")
    unset(_OpenCVLite_discovered_release_runtime CACHE)
    unset(_OpenCVLite_discovered_debug_runtime)
    unset(_OpenCVLite_discovered_debug_runtime CACHE)
    set(_OpenCVLite_discovered_debug_runtime
      "_OpenCVLite_discovered_debug_runtime-NOTFOUND")
    find_file(_OpenCVLite_discovered_debug_runtime
      NAMES "opencv_${component}${_OpenCVLite_version_suffix}d.dll" "opencv_${component}d.dll"
      PATHS "${_OpenCVLite_root}/bin"
      NO_DEFAULT_PATH)
    set(_OpenCVLite_${_OpenCVLite_upper}_DEBUG_RUNTIME
      "${_OpenCVLite_discovered_debug_runtime}")
    unset(_OpenCVLite_discovered_debug_runtime CACHE)

    set(_OpenCVLite_imported_configurations)
    if(OpenCVLite_${_OpenCVLite_upper}_RELEASE_LIBRARY)
      if(NOT _OpenCVLite_${_OpenCVLite_upper}_RELEASE_RUNTIME)
        message(FATAL_ERROR "OPENCV_LITE_ROOT does not contain opencv_${component} release runtime")
      endif()
      _opencv_lite_require_contained("${_OpenCVLite_${_OpenCVLite_upper}_RELEASE_RUNTIME}"
        "opencv_${component} release runtime")
      set_property(TARGET OpenCVLite::${component} PROPERTY
        IMPORTED_IMPLIB_RELEASE "${OpenCVLite_${_OpenCVLite_upper}_RELEASE_LIBRARY}")
      set_property(TARGET OpenCVLite::${component} PROPERTY
        IMPORTED_LOCATION_RELEASE "${_OpenCVLite_${_OpenCVLite_upper}_RELEASE_RUNTIME}")
      list(APPEND _OpenCVLite_imported_configurations RELEASE)
    endif()
    if(OpenCVLite_${_OpenCVLite_upper}_DEBUG_LIBRARY)
      if(NOT _OpenCVLite_${_OpenCVLite_upper}_DEBUG_RUNTIME)
        message(FATAL_ERROR "OPENCV_LITE_ROOT does not contain opencv_${component} debug runtime")
      endif()
      _opencv_lite_require_contained("${_OpenCVLite_${_OpenCVLite_upper}_DEBUG_RUNTIME}"
        "opencv_${component} debug runtime")
      set_property(TARGET OpenCVLite::${component} PROPERTY
        IMPORTED_IMPLIB_DEBUG "${OpenCVLite_${_OpenCVLite_upper}_DEBUG_LIBRARY}")
      set_property(TARGET OpenCVLite::${component} PROPERTY
        IMPORTED_LOCATION_DEBUG "${_OpenCVLite_${_OpenCVLite_upper}_DEBUG_RUNTIME}")
      list(APPEND _OpenCVLite_imported_configurations DEBUG)
    endif()
    if(OpenCVLite_${_OpenCVLite_upper}_RELEASE_LIBRARY AND
       NOT OpenCVLite_${_OpenCVLite_upper}_DEBUG_LIBRARY)
      set_property(TARGET OpenCVLite::${component} PROPERTY MAP_IMPORTED_CONFIG_DEBUG Release)
      set_property(TARGET OpenCVLite::${component} PROPERTY MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
      set_property(TARGET OpenCVLite::${component} PROPERTY MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    elseif(OpenCVLite_${_OpenCVLite_upper}_DEBUG_LIBRARY AND
           NOT OpenCVLite_${_OpenCVLite_upper}_RELEASE_LIBRARY)
      set_property(TARGET OpenCVLite::${component} PROPERTY MAP_IMPORTED_CONFIG_RELEASE Debug)
      set_property(TARGET OpenCVLite::${component} PROPERTY MAP_IMPORTED_CONFIG_RELWITHDEBINFO Debug)
      set_property(TARGET OpenCVLite::${component} PROPERTY MAP_IMPORTED_CONFIG_MINSIZEREL Debug)
    else()
      set_property(TARGET OpenCVLite::${component} PROPERTY MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
      set_property(TARGET OpenCVLite::${component} PROPERTY MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    endif()
    set_property(TARGET OpenCVLite::${component} PROPERTY
      IMPORTED_CONFIGURATIONS "${_OpenCVLite_imported_configurations}")
  else()
    set(_OpenCVLite_selected_library "${OpenCVLite_${_OpenCVLite_upper}_RELEASE_LIBRARY}")
    if(NOT _OpenCVLite_selected_library)
      set(_OpenCVLite_selected_library "${OpenCVLite_${_OpenCVLite_upper}_DEBUG_LIBRARY}")
    endif()
    set_target_properties(OpenCVLite::${component} PROPERTIES
      IMPORTED_LOCATION "${_OpenCVLite_selected_library}")
  endif()
  set(OpenCVLite_${component}_FOUND TRUE PARENT_SCOPE)
endfunction()

set(_OpenCVLite_existing_target_count 0)
foreach(_OpenCVLite_component IN LISTS _OpenCVLite_requested_components)
  if(TARGET OpenCVLite::${_OpenCVLite_component})
    math(EXPR _OpenCVLite_existing_target_count "${_OpenCVLite_existing_target_count} + 1")
  endif()
endforeach()
if(_OpenCVLite_existing_target_count GREATER 0 AND
   _OpenCVLite_existing_target_count LESS _OpenCVLite_requested_count)
  message(FATAL_ERROR
    "FindOpenCVLite rejects partial imported targets for the requested components")
endif()

foreach(_OpenCVLite_component IN LISTS _OpenCVLite_requested_components)
  _opencv_lite_find_component("${_OpenCVLite_component}")
endforeach()

find_package_handle_standard_args(OpenCVLite
  REQUIRED_VARS OpenCVLite_INCLUDE_DIR
  VERSION_VAR OpenCVLite_VERSION
  HANDLE_COMPONENTS)

mark_as_advanced(OpenCVLite_INCLUDE_DIR)
