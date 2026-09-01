cmake_add_test(
  NAME test_hand_models_cuda_validation
  SOURCES test_tensorrt_backend_validation.cpp
  LIBS KFCore::hand_models_cuda TurboUtils::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "hand_models_cuda/tests")

set(_hand_engine_root
  "hand_gesture_model/tensorrt/${KFCORE_TENSORRT_ENGINE_PROFILE}")
set(_hand_cuda_palm
  "./yolo-models/${_hand_engine_root}/palm_detection.engine")
set(_hand_cuda_landmark
  "./yolo-models/${_hand_engine_root}/hand_landmark.engine")
set(_hand_cuda_classifier
  "./yolo-models/${_hand_engine_root}/keypoint_classifier.engine")

cmake_add_test(
  NAME test_hand_models_cuda_integration
  SOURCES test_tensorrt_integration.cpp
  LIBS KFCore::hand_models_cuda TurboUtils::TinyTest
  DEFS
    "KFCORE_HAND_TRT_TEST_PALM=\"${_hand_cuda_palm}\""
    "KFCORE_HAND_TRT_TEST_LANDMARK=\"${_hand_cuda_landmark}\""
    "KFCORE_HAND_TRT_TEST_CLASSIFIER=\"${_hand_cuda_classifier}\""
  COMPILE_FEATURES cxx_std_17
  TEST_PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  FOLDER "hand_models_cuda/tests")

if(WIN32)
  set_tests_properties(
    test_hand_models_cuda_validation
    test_hand_models_cuda_integration
    PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
endif()
