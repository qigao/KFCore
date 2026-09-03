cmake_add_test(
  NAME test_thig_temporal_graph
  SOURCES test_temporal_graph.cpp
  LIBS KFCore::thig Salts::TinyTest
  INCLUDES ../include
  COMPILE_FEATURES cxx_std_17
  FOLDER "thig/tests")
