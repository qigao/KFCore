set(_yolo_core_test_folder "yolo_core/tests")

cmake_add_test(
  NAME test_raw_yolo
  SOURCES test_raw_yolo.cpp
  LIBS KFCore::yolo_core Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_core_test_folder}")

cmake_add_test(
  NAME test_yolo_domain_profile
  SOURCES test_yolo_domain_profile.cpp ../examples/yolo_domain_profile.cpp
  LIBS KFCore::yolo_core Salts::TinyTest
  INCLUDES ../examples
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_core_test_folder}")
cmake_add_test(
  NAME test_yolo_domain_cli
  SOURCES
    test_yolo_domain_cli.cpp
    ../examples/yolo_domain_cli.cpp
    ../examples/yolo_domain_profile.cpp
  LIBS KFCore::yolo_core TurboParser::Capture Salts::TinyTest
  INCLUDES ../examples
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_core_test_folder}")
cmake_add_test(
  NAME test_yolo_domain_capture
  SOURCES test_yolo_domain_capture.cpp ../examples/yolo_domain_capture.cpp
  LIBS TurboParser::Capture Salts::TinyTest
  INCLUDES ../examples
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_core_test_folder}")
cmake_add_test(
  NAME test_yolo_domain_ui
  SOURCES
    test_yolo_domain_ui.cpp
    ../examples/yolo_domain_capture.cpp
    ../examples/yolo_domain_frame.cpp
    ../examples/yolo_domain_profile.cpp
    ../examples/yolo_domain_ui.cpp
  LIBS
    KFCore::yolo_core
    KFCore::image_processor_cpu
    TurboParser::Capture
    Salts::TinyTest
  INCLUDES ../examples ${Stb_INCLUDE_DIR}
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_core_test_folder}")
cmake_add_test(
  NAME test_yolo_tracking
  SOURCES
    test_yolo_tracking.cpp
    ../src/tracking.cpp
    ../../trackers/src/tracker.c
    ../../trackers/tests/tracker_test_alloc.c
  LIBS KFCore::kfcore miniblas Salts::TinyTest
  DEFS KFCORE_TRACKERS_TEST_ALLOCATOR
  INCLUDES ../include ../src ../../trackers/include ../../trackers/tests
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_core_test_folder}")
