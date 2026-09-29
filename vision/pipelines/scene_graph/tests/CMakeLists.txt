set(_scene_graph_test_folder "vision/pipelines/scene_graph/tests")

cmake_add_test(
  NAME test_scene_graph_pipeline
  SOURCES test_scene_graph.cpp
  LIBS KFCore::scene_graph_pipeline Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_scene_graph_test_folder}")
