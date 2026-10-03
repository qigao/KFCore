set(_yolo_core_test_folder "vision/core/yolo/tests")

cmake_add_test(
  NAME test_raw_yolo
  SOURCES test_raw_yolo.cpp
  LIBS KFCore::yolo_core Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_core_test_folder}")

cmake_add_test(
  NAME test_yolo_tracking
  SOURCES
    test_yolo_tracking.cpp
    ../src/tracking.cpp
    ${CMAKE_SOURCE_DIR}/trackers/src/tracker.c
    ${CMAKE_SOURCE_DIR}/trackers/tests/tracker_test_alloc.c
  LIBS KFCore::kalman miniblas Salts::TinyTest
  DEFS KFCORE_TRACKERS_TEST_ALLOCATOR
  INCLUDES
    ../include
    ../src
    ${CMAKE_SOURCE_DIR}/trackers/include
    ${CMAKE_SOURCE_DIR}/trackers/tests
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_core_test_folder}")
