if(NOT DEFINED KFCORE_MODEL_PATH_MODULE OR
   NOT DEFINED KFCORE_MODEL_ROOT OR
   NOT DEFINED KFCORE_MODEL_RELATIVE OR
   NOT DEFINED KFCORE_MODEL_RESULT)
  message(FATAL_ERROR "Model path fixture inputs are required")
endif()

include("${KFCORE_MODEL_PATH_MODULE}")
if(KFCORE_MODEL_OPTIONAL)
  kfcore_resolve_optional_model_path(
    "${KFCORE_MODEL_ROOT}" "${KFCORE_MODEL_RELATIVE}" _resolved_model)
else()
  kfcore_resolve_model_path(
    "${KFCORE_MODEL_ROOT}" "${KFCORE_MODEL_RELATIVE}" _resolved_model)
endif()
file(WRITE "${KFCORE_MODEL_RESULT}" "${_resolved_model}")
