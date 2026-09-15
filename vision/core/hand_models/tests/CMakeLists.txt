set(_hand_model_core_test_folder "vision/core/hand_models/tests")

cmake_add_test(
  NAME test_hand_model_geometry
  SOURCES test_geometry.cpp
  LIBS KFCore::hand_model_core Salts::TinyTest
  INCLUDES ../include ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_model_core_test_folder}")

cmake_add_test(
  NAME test_hand_model_decode
  SOURCES test_decode.cpp
  LIBS KFCore::hand_model_core Salts::TinyTest
  INCLUDES ../include ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_model_core_test_folder}")

cmake_add_test(
  NAME test_hand_tracking
  SOURCES test_hand_tracking.cpp
  LIBS KFCore::hand_model_core KFCore::image_processor_cpu Salts::TinyTest
  INCLUDES ../include
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_model_core_test_folder}")

cmake_add_test(
  NAME test_hand_model_api
  SOURCES test_hand_model_api.cpp
  LIBS KFCore::hand_model_runtime Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_model_core_test_folder}")
