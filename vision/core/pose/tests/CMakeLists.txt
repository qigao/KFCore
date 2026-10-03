set(_pose_test_folder "vision/core/pose/tests")

if(BUILD_TESTS)
cmake_add_test(
  NAME test_pose_schema
  SOURCES test_schema.cpp
  LIBS KFCore::pose_core Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_pose_test_folder}")
endif()
