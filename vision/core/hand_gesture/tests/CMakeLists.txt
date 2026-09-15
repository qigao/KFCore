set(_hand_gesture_test_folder "vision/core/hand_gesture/tests")

cmake_add_test(
  NAME test_hand_gesture_feature_encoder
  SOURCES test_feature_encoder.cpp
  LIBS KFCore::hand_gesture Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_gesture_test_folder}")
