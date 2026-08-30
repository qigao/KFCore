function(kfcore_resolve_model_path model_root relative_path output_variable)
  if("${model_root}" STREQUAL "" OR NOT IS_DIRECTORY "${model_root}")
    message(FATAL_ERROR "KFCORE_MODEL_ROOT must name an existing directory")
  endif()
  if("${relative_path}" STREQUAL "")
    message(FATAL_ERROR "relative model path must not be empty")
  endif()

  set(_relative_path "${relative_path}")
  cmake_path(IS_ABSOLUTE _relative_path _relative_is_absolute)
  if(_relative_is_absolute)
    message(FATAL_ERROR "relative model path must not be absolute: ${relative_path}")
  endif()

  file(REAL_PATH "${model_root}" _model_root_real)
  set(_model_candidate "${_model_root_real}")
  cmake_path(APPEND _model_candidate "${relative_path}"
    OUTPUT_VARIABLE _model_candidate_absolute)
  cmake_path(NORMAL_PATH _model_candidate_absolute)
  cmake_path(IS_PREFIX _model_root_real "${_model_candidate_absolute}"
    NORMALIZE _model_is_within_root)
  if(NOT _model_is_within_root OR
     "${_model_candidate_absolute}" STREQUAL "${_model_root_real}")
    message(FATAL_ERROR
      "relative model path escapes KFCORE_MODEL_ROOT: ${relative_path}")
  endif()
  if(NOT EXISTS "${_model_candidate_absolute}" OR
     IS_DIRECTORY "${_model_candidate_absolute}")
    message(FATAL_ERROR
      "resolved model path is not an existing file: ${_model_candidate_absolute}")
  endif()

  file(REAL_PATH "${_model_candidate_absolute}" _model_real)
  cmake_path(IS_PREFIX _model_root_real "${_model_real}"
    NORMALIZE _real_model_is_within_root)
  if(NOT _real_model_is_within_root)
    message(FATAL_ERROR
      "resolved model path escapes KFCORE_MODEL_ROOT: ${relative_path}")
  endif()
  set(${output_variable} "${_model_real}" PARENT_SCOPE)
endfunction()

function(kfcore_resolve_optional_model_path model_root relative_path output_variable)
  if("${relative_path}" STREQUAL "")
    set(${output_variable} "" PARENT_SCOPE)
    return()
  endif()

  kfcore_resolve_model_path("${model_root}" "${relative_path}" _resolved_model)
  set(${output_variable} "${_resolved_model}" PARENT_SCOPE)
endfunction()
