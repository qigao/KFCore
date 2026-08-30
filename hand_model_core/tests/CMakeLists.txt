set(_hand_model_core_test_folder "hand_model_core/tests")

cmake_add_test(
  NAME test_hand_model_geometry
  SOURCES test_geometry.cpp
  LIBS KFCore::hand_model_core TurboUtils::TinyTest
  INCLUDES ../include ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_model_core_test_folder}")
cmake_add_test(
  NAME test_hand_model_decode
  SOURCES test_decode.cpp
  LIBS KFCore::hand_model_core TurboUtils::TinyTest
  INCLUDES ../include ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_model_core_test_folder}")
cmake_add_test(
  NAME test_hand_model_hand_pipeline
  SOURCES test_hand_pipeline.cpp
  LIBS KFCore::hand_model_core KFCore::image_processor_cpu TurboUtils::TinyTest
  INCLUDES ../include
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_model_core_test_folder}")

cmake_add_test(
  NAME test_hand_model_api
  SOURCES test_hand_model_api.cpp
  LIBS KFCore::hand_model_core TurboUtils::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_model_core_test_folder}")
