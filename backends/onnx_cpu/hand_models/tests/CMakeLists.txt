cmake_add_test(
  NAME test_hand_models_cpu_validation
  SOURCES test_cpu_backend_validation.cpp
  LIBS KFCore::hand_models_cpu Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "backends/onnx_cpu/hand_models/tests")

set(_hand_cpu_palm
  "./yolo-models/hand_gesture_model/palm_detection/palm_detection_full_inf_post_192x192.onnx")
set(_hand_cpu_landmark
  "./yolo-models/hand_gesture_model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx")
set(_hand_cpu_classifier
  "./yolo-models/hand_gesture_model/keypoint_classifier/keypoint_classifier.onnx")

cmake_add_test(
  NAME test_hand_models_cpu_integration
  SOURCES test_cpu_integration.cpp
  LIBS KFCore::hand_models_cpu Salts::TinyTest
  DEFS
    KFCORE_HAND_CPU_TEST_PALM="${_hand_cpu_palm}"
    KFCORE_HAND_CPU_TEST_LANDMARK="${_hand_cpu_landmark}"
    KFCORE_HAND_CPU_TEST_CLASSIFIER="${_hand_cpu_classifier}"
  COMPILE_FEATURES cxx_std_17
  TEST_PROPERTIES WORKING_DIRECTORY "${KFCORE_MODEL_TEST_WORKING_DIRECTORY}"
  FOLDER "backends/onnx_cpu/hand_models/tests")

if(WIN32)
  set_tests_properties(
    test_hand_models_cpu_validation
    test_hand_models_cpu_integration
    PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$ENV{ONNXRUNTIME_ROOT}/lib")
endif()
