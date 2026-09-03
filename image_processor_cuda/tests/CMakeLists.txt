cmake_add_test(
  NAME test_image_processor_cuda_api
  SOURCES test_image_processor.cpp
  LIBS KFCore::image_processor_cuda Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "image_processor_cuda/tests")

cmake_add_test(
  NAME test_image_processor_cuda
  SOURCES test_image_processor_cuda.cu
  LIBS
    KFCore::image_processor_cuda
    KFCore::image_processor_cpu
    CUDA::cudart
    Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "image_processor_cuda/tests"
  TARGET_PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

cmake_add_benchmark(
  NAME benchmark_yuv_preprocess
  SOURCES benchmark_yuv_preprocess.cpp
  LIBS
    KFCore::image_processor_cuda
    KFCore::image_processor_cpu
    CUDA::cudart
    Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "image_processor_cuda/benchmarks")
