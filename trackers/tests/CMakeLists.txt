cmake_add_test(
  NAME test_tracker
  SOURCES test_trackers.c
  LIBS KFCore::trackers Salts::TinyTest
  FOLDER "trackers/tests"
  TARGET_PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF)

cmake_add_test(
  NAME test_bytetrack_status
  SOURCES
    test_bytetrack_status.c
    tracker_test_alloc.c
    ../src/tracker.c
  LIBS KFCore::kfcore miniblas Salts::TinyTest
  DEFS KFCORE_TRACKERS_TEST_ALLOCATOR
  INCLUDES ../include ${CMAKE_CURRENT_SOURCE_DIR}
  FOLDER "trackers/tests"
  TARGET_PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF)
