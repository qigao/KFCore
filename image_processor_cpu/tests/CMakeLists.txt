cmake_add_test(
  NAME test_image_processor_cpu
  SOURCES test_cpu_image_processor.cpp
  LIBS KFCore::image_processor_cpu TurboUtils::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "image_processor_cpu/tests")
