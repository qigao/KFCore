if(NOT DEFINED KFCORE_SAFETY_MODULE OR
   NOT DEFINED KFCORE_ALLOWED_ROOT OR
   NOT DEFINED KFCORE_TARGET_DIR)
  message(FATAL_ERROR "Safety module, allowed root, and target directory are required")
endif()

include("${KFCORE_SAFETY_MODULE}")
kfcore_reset_test_directory("${KFCORE_ALLOWED_ROOT}" "${KFCORE_TARGET_DIR}")
