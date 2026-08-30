function(kfcore_reset_test_directory allowed_root target_directory)
  if("${allowed_root}" STREQUAL "" OR "${target_directory}" STREQUAL "")
    message(FATAL_ERROR "Allowed test root and target directory must not be empty")
  endif()

  set(_allowed_root "${allowed_root}")
  set(_target_directory "${target_directory}")
  cmake_path(ABSOLUTE_PATH _allowed_root NORMALIZE
    OUTPUT_VARIABLE _allowed_root_absolute)
  cmake_path(ABSOLUTE_PATH _target_directory NORMALIZE
    OUTPUT_VARIABLE _target_absolute)
  if(NOT IS_DIRECTORY "${_allowed_root_absolute}")
    message(FATAL_ERROR
      "Allowed test root must be an existing directory: ${_allowed_root_absolute}")
  endif()

  cmake_path(IS_PREFIX _allowed_root_absolute "${_target_absolute}"
    NORMALIZE _target_is_descendant)
  if(NOT _target_is_descendant OR
     "${_target_absolute}" STREQUAL "${_allowed_root_absolute}")
    message(FATAL_ERROR
      "Refusing to reset a directory outside the allowed test root: "
      "${_target_absolute}")
  endif()

  file(REAL_PATH "${_allowed_root_absolute}" _allowed_root_real)
  if(EXISTS "${_target_absolute}")
    file(REAL_PATH "${_target_absolute}" _target_real)
    cmake_path(IS_PREFIX _allowed_root_real "${_target_real}"
      NORMALIZE _real_target_is_descendant)
    if(NOT _real_target_is_descendant OR
       "${_target_real}" STREQUAL "${_allowed_root_real}")
      message(FATAL_ERROR
        "Refusing to reset a directory that resolves outside the allowed test root: "
        "${_target_real}")
    endif()
  endif()

  file(REMOVE_RECURSE "${_target_absolute}")
  file(MAKE_DIRECTORY "${_target_absolute}")
endfunction()

function(kfcore_get_parent_build_config binary_directory output_variable)
  load_cache("${binary_directory}" READ_WITH_PREFIX _kfcore_parent_ CMAKE_BUILD_TYPE)
  if(NOT DEFINED _kfcore_parent_CMAKE_BUILD_TYPE OR
     NOT "${_kfcore_parent_CMAKE_BUILD_TYPE}" MATCHES "^[A-Za-z0-9_.+-]+$")
    message(FATAL_ERROR
      "Parent build tree must define a valid CMAKE_BUILD_TYPE: ${binary_directory}")
  endif()
  set(${output_variable} "${_kfcore_parent_CMAKE_BUILD_TYPE}" PARENT_SCOPE)
endfunction()

function(kfcore_require_environment_directory variable_name)
  if(NOT DEFINED ENV{${variable_name}} OR
     NOT IS_DIRECTORY "$ENV{${variable_name}}")
    message(FATAL_ERROR "${variable_name} must name an existing directory")
  endif()
endfunction()
