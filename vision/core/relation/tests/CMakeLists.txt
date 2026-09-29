set(_relation_test_folder "vision/core/relation/tests")

cmake_add_test(
  NAME test_relation_decode
  SOURCES test_decode.cpp
  LIBS KFCore::relation_core Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_relation_test_folder}")


cmake_add_test(
  NAME test_relation_vocabulary_score
  SOURCES test_vocabulary_score.cpp
  LIBS KFCore::relation_core Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_relation_test_folder}")
