include(FindPackageHandleStandardArgs)

if(TARGET OpenCVLite::core OR TARGET OpenCVLite::imgproc OR TARGET OpenCVLite::imgcodecs)
  message(FATAL_ERROR
    "FindOpenCVLite requires ownership of OpenCVLite imported targets; a target already exists")
endif()

foreach(_OpenCVLite_predefined IN ITEMS
    OpenCVLite_INCLUDE_DIR
    OpenCVLite_CORE_RELEASE_LIBRARY
    OpenCVLite_CORE_DEBUG_LIBRARY
    OpenCVLite_IMGPROC_RELEASE_LIBRARY
    OpenCVLite_IMGPROC_DEBUG_LIBRARY
    OpenCVLite_IMGCODECS_RELEASE_LIBRARY
    OpenCVLite_IMGCODECS_DEBUG_LIBRARY)
  if(DEFINED ${_OpenCVLite_predefined} OR DEFINED CACHE{${_OpenCVLite_predefined}})
    message(FATAL_ERROR
      "FindOpenCVLite refuses a predefined result: ${_OpenCVLite_predefined}")
  endif()
endforeach()

if(NOT DEFINED ENV{OPENCV_LITE_ROOT} OR "$ENV{OPENCV_LITE_ROOT}" STREQUAL "")
  message(FATAL_ERROR "OPENCV_LITE_ROOT is required when KFCORE_BUILD_YOLO_OPENCV=ON")
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

function(_opencv_lite_find_component component)
  string(TOUPPER "${component}" _OpenCVLite_upper)
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
endfunction()

_opencv_lite_find_component(core)
_opencv_lite_find_component(imgproc)
_opencv_lite_find_component(imgcodecs)

find_package_handle_standard_args(OpenCVLite
  REQUIRED_VARS OpenCVLite_INCLUDE_DIR
  HANDLE_COMPONENTS)

mark_as_advanced(OpenCVLite_INCLUDE_DIR)
