add_library(kfcore_image_core SHARED
  src/cpu.cpp
  src/error.cpp)
target_include_directories(kfcore_image_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_image_core PUBLIC cxx_std_17)
set_target_properties(kfcore_image_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "image_processor"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_image_core
  ALIAS KFCore::image_core
  EXPORT_NAME image_core
  EXPORT_SET KFCoreTargets
  FOLDER "image_processor")

if(KFCORE_IMAGE_PROCESSOR_CUDA_ENABLED)
  add_library(kfcore_image_processor SHARED
    src/image_processor.cpp
    src/image_processor_cuda.cu)
  target_include_directories(kfcore_image_processor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_features(kfcore_image_processor PUBLIC cxx_std_17)
  target_link_libraries(kfcore_image_processor
    PUBLIC KFCore::image_core
    PRIVATE CUDA::cudart)
  set_target_properties(kfcore_image_processor PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "image_processor"
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_image_processor
  ALIAS KFCore::image_processor
  EXPORT_NAME image_processor
  EXPORT_SET KFCoreTargets
  FOLDER "image_processor")
endif()

if(BUILD_TESTS)
  add_executable(test_cpu_image_processor tests/test_cpu_image_processor.cpp)
  target_link_libraries(test_cpu_image_processor PRIVATE
    KFCore::image_core
    TurboUtils::TinyTest)
  set_target_properties(test_cpu_image_processor PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "image_processor/tests")
  add_test(NAME test_cpu_image_processor COMMAND test_cpu_image_processor)

  if(KFCORE_IMAGE_PROCESSOR_CUDA_ENABLED)
    add_executable(test_image_processor tests/test_image_processor.cpp)
  target_link_libraries(test_image_processor PRIVATE
    KFCore::image_processor
    TurboUtils::TinyTest)
  set_target_properties(test_image_processor PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "image_processor/tests")
  add_test(NAME test_image_processor COMMAND test_image_processor)

  add_executable(test_image_processor_cuda tests/test_image_processor_cuda.cu)
  target_link_libraries(test_image_processor_cuda PRIVATE
    KFCore::image_processor
    CUDA::cudart
    TurboUtils::TinyTest)
  set_target_properties(test_image_processor_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "image_processor/tests")
  add_test(NAME test_image_processor_cuda COMMAND test_image_processor_cuda)

  add_executable(benchmark_yuv_preprocess tests/benchmark_yuv_preprocess.cpp)
  target_link_libraries(benchmark_yuv_preprocess PRIVATE
    KFCore::image_processor
    CUDA::cudart
    TurboUtils::TinyTest)
  set_target_properties(benchmark_yuv_preprocess PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "image_processor/benchmarks")
  endif()

  if(KFCORE_INSTALL AND KFCORE_IMAGE_PROCESSOR_CUDA_ENABLED)
    set(_image_processor_consumer_root
      "${CMAKE_BINARY_DIR}/tests/image_processor_installed_consumer")
    add_test(NAME test_image_processor_installed_consumer
      COMMAND "${CMAKE_COMMAND}"
        "-DKFCORE_BINARY_DIR=${CMAKE_BINARY_DIR}"
        "-DKFCORE_CONSUMER_SOURCE_DIR=${CMAKE_SOURCE_DIR}/cmake/tests/ImageProcessorInstalledConsumer"
        "-DKFCORE_CONSUMER_BINARY_DIR=${_image_processor_consumer_root}/build"
        "-DKFCORE_CONSUMER_INSTALL_PREFIX=${_image_processor_consumer_root}/install"
        "-DKFCORE_CONSUMER_KFCORE_DIR=${_image_processor_consumer_root}/install/${CMAKE_INSTALL_LIBDIR}/cmake/KFCore"
        "-DKFCORE_CONSUMER_BUILD_CONFIG=$<CONFIG>"
        "-DKFCORE_CONSUMER_CUDATOOLKIT_ROOT=${CUDAToolkit_ROOT}"
        -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_image_processor_installed_consumer.cmake")
  endif()
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
