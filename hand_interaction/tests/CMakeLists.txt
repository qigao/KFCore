set(_hand_test_folder "vision/interaction/hand/tests")

cmake_add_test(
  NAME test_hand_primitive_extractor
  SOURCES test_primitive_extractor.cpp
  LIBS KFCore::hand_interaction Salts::TinyTest
  INCLUDES ../include
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_test_folder}")

cmake_add_test(
  NAME test_hand_interaction
  SOURCES test_hand_interaction.cpp
  LIBS KFCore::hand_interaction Salts::TinyTest
  INCLUDES ../include
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_test_folder}")
