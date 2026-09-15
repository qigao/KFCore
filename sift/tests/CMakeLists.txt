cmake_add_test(
  NAME test_sift
  SOURCES test_sift.cpp
  LIBS KFCore::sift Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "sift/tests")

cmake_add_test(
  NAME test_popsift_options
  SOURCES test_popsift_options.cpp
  LIBS KFCore::sift Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "sift/tests")

cmake_add_test(
  NAME test_popsift_result_validation
  SOURCES test_popsift_result_validation.cpp
  LIBS KFCore::sift Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "sift/tests")

if(KFCORE_ENABLE_CUDA)
  cmake_add_test(
    NAME test_popsift_internal
    SOURCES test_popsift_internal.cpp
    LIBS kfcore_popsift_internal CUDA::cudart_static Threads::Threads Salts::TinyTest
    DEFS KFCORE_POPSIFT_TESTING=1
    INCLUDES ../vendor/popsift/src
    COMPILE_FEATURES cxx_std_17
    FOLDER "sift/tests"
    TEST_PROPERTIES TIMEOUT 30)

  cmake_add_test(
    NAME test_popsift_adapter
    SOURCES test_popsift_adapter.cpp
    LIBS KFCore::sift_popsift CUDA::cudart_static Salts::TinyTest
    COMPILE_FEATURES cxx_std_17
    FOLDER "sift/tests")

  cmake_add_benchmark(
    NAME benchmark_popsift
    SOURCES benchmark_popsift.cpp
    LIBS KFCore::sift_popsift Salts::TinyTest
    COMPILE_FEATURES cxx_std_17
    FOLDER "sift/benchmarks")
endif()
