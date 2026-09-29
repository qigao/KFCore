set(_scene_interaction_test_folder "scene_interaction/tests")

cmake_add_test(
  NAME test_scene_interaction
  SOURCES test_scene_interaction.cpp
  LIBS KFCore::scene_interaction Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_scene_interaction_test_folder}")
