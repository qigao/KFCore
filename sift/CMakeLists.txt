add_library(kfcore_sift SHARED src/sift_extractor.cpp)
target_include_directories(kfcore_sift PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_sift PUBLIC cxx_std_17)
target_link_libraries(kfcore_sift PUBLIC KFCore::image_processor)
set_target_properties(kfcore_sift PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "sift"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_sift
  ALIAS KFCore::sift
  EXPORT_NAME sift
  EXPORT_SET KFCoreTargets
  FOLDER "sift")

if(KFCORE_BUILD_SIFT_POPSIFT)
  add_subdirectory(vendor/popsift)

  add_library(kfcore_sift_popsift SHARED src/popsift_extractor.cpp)
  target_include_directories(kfcore_sift_popsift PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_features(kfcore_sift_popsift PUBLIC cxx_std_17)
  target_link_libraries(kfcore_sift_popsift
    PUBLIC KFCore::sift
    PRIVATE kfcore_popsift_internal CUDA::cudart)
  set_target_properties(kfcore_sift_popsift PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "sift"
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
  cmake_config_target(kfcore_sift_popsift
    ALIAS KFCore::sift_popsift
    EXPORT_NAME sift_popsift
    EXPORT_SET KFCoreTargets
    FOLDER "sift")
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/popsift/COPYING.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/popsift/UPSTREAM.md"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/KFCore/popsift")
endif()

if(BUILD_TESTS)
  add_executable(test_sift tests/test_sift.cpp)
  target_link_libraries(test_sift PRIVATE KFCore::sift TurboUtils::TinyTest)
  set_target_properties(test_sift PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "sift/tests")
  add_test(NAME test_sift COMMAND test_sift)

  add_executable(test_popsift_options tests/test_popsift_options.cpp)
  target_link_libraries(test_popsift_options PRIVATE KFCore::sift TurboUtils::TinyTest)
  set_target_properties(test_popsift_options PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "sift/tests")
  add_test(NAME test_popsift_options COMMAND test_popsift_options)

  if(KFCORE_BUILD_SIFT_POPSIFT)
    add_executable(test_popsift_internal tests/test_popsift_internal.cpp)
    target_include_directories(test_popsift_internal PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/vendor/popsift/src")
    target_link_libraries(test_popsift_internal PRIVATE
      kfcore_popsift_internal
      Threads::Threads
      TurboUtils::TinyTest)
    set_target_properties(test_popsift_internal PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "sift/tests")
    add_test(NAME test_popsift_internal COMMAND test_popsift_internal)

    add_executable(test_popsift_result_validation tests/test_popsift_result_validation.cpp)
    target_include_directories(test_popsift_result_validation PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_popsift_result_validation PRIVATE
      KFCore::sift
      TurboUtils::TinyTest)
    set_target_properties(test_popsift_result_validation PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "sift/tests")
    add_test(NAME test_popsift_result_validation COMMAND test_popsift_result_validation)

    add_executable(test_popsift_adapter tests/test_popsift_adapter.cpp)
    target_link_libraries(test_popsift_adapter PRIVATE
      KFCore::sift_popsift
      CUDA::cudart
      TurboUtils::TinyTest)
    set_target_properties(test_popsift_adapter PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "sift/tests")
    add_test(NAME test_popsift_adapter COMMAND test_popsift_adapter)

    add_executable(benchmark_popsift tests/benchmark_popsift.cpp)
    target_link_libraries(benchmark_popsift PRIVATE
      KFCore::sift_popsift
      TurboUtils::TinyTest)
    set_target_properties(benchmark_popsift PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "sift/benchmarks")
  endif()

  if(KFCORE_INSTALL)
    set(_sift_consumer_root "${CMAKE_BINARY_DIR}/tests/sift_installed_consumer")
    add_test(NAME test_sift_installed_consumer
      COMMAND "${CMAKE_COMMAND}"
        "-DKFCORE_BINARY_DIR=${CMAKE_BINARY_DIR}"
        "-DKFCORE_CONSUMER_SOURCE_DIR=${CMAKE_SOURCE_DIR}/cmake/tests/SiftInstalledConsumer"
        "-DKFCORE_CONSUMER_BINARY_DIR=${_sift_consumer_root}/build"
        "-DKFCORE_CONSUMER_INSTALL_PREFIX=${_sift_consumer_root}/install"
        "-DKFCORE_CONSUMER_KFCORE_DIR=${_sift_consumer_root}/install/${CMAKE_INSTALL_LIBDIR}/cmake/KFCore"
        "-DKFCORE_CONSUMER_BUILD_CONFIG=$<CONFIG>"
        "-DKFCORE_CONSUMER_CUDATOOLKIT_ROOT=${CUDAToolkit_ROOT}"
        "-DKFCORE_EXPECT_SIFT_POPSIFT=${KFCORE_BUILD_SIFT_POPSIFT}"
        -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_sift_installed_consumer.cmake")
  endif()
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
