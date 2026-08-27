find_package(OpenCVLite REQUIRED COMPONENTS core imgproc)

add_library(kfcore_face_applications SHARED
  src/composer.cpp
  src/error.cpp
  src/geometry.cpp
  src/preprocess.cpp)
target_sources(kfcore_face_applications PRIVATE src/tensorrt.cpp)
target_include_directories(kfcore_face_applications
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_face_applications PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_applications
  PRIVATE
    KFCore::image_processor
  PUBLIC
    KFCore::face_models
    KFCore::tensorrt_yolo
    KFCore::yolo_opencv
    OpenCVLite::core
    OpenCVLite::imgproc)
set_target_properties(kfcore_face_applications PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "face_applications"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_face_applications
  ALIAS KFCore::face_applications
  EXPORT_NAME face_applications
  EXPORT_SET KFCoreTargets
  FOLDER "face_applications")

if(KFCORE_BUILD_FACE_APPLICATION_EXAMPLES)
  find_package(OpenCVLite REQUIRED COMPONENTS imgcodecs)
  add_executable(face_swap_image
    examples/face_swap_cli.cpp
    examples/face_swap_image.cpp)
  target_include_directories(face_swap_image PRIVATE examples)
  target_link_libraries(face_swap_image PRIVATE
    KFCore::face_applications
    OpenCVLite::imgcodecs
    TurboUtils::Core)
  set_target_properties(face_swap_image PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/examples")
endif()

if(BUILD_TESTS)
  add_executable(test_face_swap_cli
    tests/test_face_swap_cli.cpp
    examples/face_swap_cli.cpp)
  target_include_directories(test_face_swap_cli PRIVATE examples)
  target_link_libraries(test_face_swap_cli PRIVATE
    TurboUtils::Core
    TurboUtils::TinyTest)
  set_target_properties(test_face_swap_cli PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/tests")
  add_test(NAME test_face_swap_cli COMMAND test_face_swap_cli)

  add_executable(test_face_geometry tests/test_geometry.cpp)
  target_link_libraries(test_face_geometry PRIVATE
    KFCore::face_applications
    TurboUtils::TinyTest)
  set_target_properties(test_face_geometry PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/tests")
  add_test(NAME test_face_geometry COMMAND test_face_geometry)

  add_executable(test_face_preprocess tests/test_preprocess.cpp)
  target_link_libraries(test_face_preprocess PRIVATE
    KFCore::face_applications
    TurboUtils::TinyTest)
  set_target_properties(test_face_preprocess PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/tests")
  add_test(NAME test_face_preprocess COMMAND test_face_preprocess)

  add_executable(benchmark_face_preprocess tests/benchmark_preprocess.cpp)
  target_link_libraries(benchmark_face_preprocess PRIVATE
    KFCore::face_applications
    KFCore::image_processor
    CUDA::cudart
    TurboUtils::TinyTest)
  set_target_properties(benchmark_face_preprocess PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/benchmarks")

  add_executable(test_face_composer tests/test_composer.cpp)
  target_link_libraries(test_face_composer PRIVATE
    KFCore::face_applications
    TurboUtils::TinyTest)
  set_target_properties(test_face_composer PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/tests")
  add_test(NAME test_face_composer COMMAND test_face_composer)

  add_executable(test_face_application_validation tests/test_application_validation.cpp)
  target_link_libraries(test_face_application_validation PRIVATE
    KFCore::face_applications
    TurboUtils::TinyTest)
  set_target_properties(test_face_application_validation PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/tests")
  add_test(NAME test_face_application_validation COMMAND test_face_application_validation)

  if(KFCORE_BUILD_FACE_APPLICATION_INTEGRATION_TESTS)
    find_package(OpenCVLite REQUIRED COMPONENTS imgcodecs)
    set(_kfcore_face_application_required_files
      KFCORE_FACE_APPLICATION_TEST_ENGINE_12FACE
      KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68
      KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE
      KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER
      KFCORE_FACE_APPLICATION_TEST_MATRIX
      KFCORE_FACE_APPLICATION_TEST_SOURCE_IMAGE
      KFCORE_FACE_APPLICATION_TEST_TARGET_IMAGE)
    foreach(_kfcore_file_variable IN LISTS _kfcore_face_application_required_files)
      if(NOT IS_ABSOLUTE "${${_kfcore_file_variable}}" OR
         NOT EXISTS "${${_kfcore_file_variable}}" OR
         IS_DIRECTORY "${${_kfcore_file_variable}}")
        message(FATAL_ERROR
          "${_kfcore_file_variable} must name an existing absolute file when "
          "KFCORE_BUILD_FACE_APPLICATION_INTEGRATION_TESTS=ON")
      endif()
    endforeach()

    add_executable(test_face_swap_integration tests/test_face_swap_integration.cpp)
    target_link_libraries(test_face_swap_integration PRIVATE
      KFCore::face_applications
      OpenCVLite::imgcodecs
      TurboUtils::TinyTest)
    target_compile_definitions(test_face_swap_integration PRIVATE
      "KFCORE_TEST_12FACE_ENGINE=\"${KFCORE_FACE_APPLICATION_TEST_ENGINE_12FACE}\""
      "KFCORE_TEST_FACE68_ENGINE=\"${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68}\""
      "KFCORE_TEST_ARCFACE_ENGINE=\"${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE}\""
      "KFCORE_TEST_INSWAPPER_ENGINE=\"${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER}\""
      "KFCORE_TEST_GFPGAN_ENGINE=\"${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN}\""
      "KFCORE_TEST_INSWAPPER_MATRIX=\"${KFCORE_FACE_APPLICATION_TEST_MATRIX}\""
      "KFCORE_TEST_SOURCE_IMAGE=\"${KFCORE_FACE_APPLICATION_TEST_SOURCE_IMAGE}\""
      "KFCORE_TEST_TARGET_IMAGE=\"${KFCORE_FACE_APPLICATION_TEST_TARGET_IMAGE}\"")
    set_target_properties(test_face_swap_integration PROPERTIES
      CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
      FOLDER "face_applications/tests")
    add_test(NAME test_face_swap_integration COMMAND test_face_swap_integration)
    if(WIN32)
      set_tests_properties(test_face_swap_integration PROPERTIES
        ENVIRONMENT_MODIFICATION
          "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:$ENV{OPENCV_LITE_ROOT}/bin")
    endif()
  endif()

  if(WIN32)
    set_tests_properties(test_face_swap_cli test_face_geometry test_face_preprocess test_face_composer
      test_face_application_validation PROPERTIES
      ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:$ENV{OPENCV_LITE_ROOT}/bin")
  endif()

  if(KFCORE_INSTALL)
    include("${CMAKE_SOURCE_DIR}/cmake/RequireNinja.cmake")
    kfcore_require_ninja(_face_application_consumer_ninja)
    set(_face_application_consumer_root
      "${CMAKE_BINARY_DIR}/tests/face_application_installed_consumer")
    add_test(NAME test_face_application_installed_consumer
      COMMAND "${CMAKE_COMMAND}"
        "-DKFCORE_BINARY_DIR=${CMAKE_BINARY_DIR}"
        "-DKFCORE_CONSUMER_SOURCE_DIR=${CMAKE_SOURCE_DIR}/cmake/tests/FaceApplicationInstalledConsumer"
        "-DKFCORE_CONSUMER_BINARY_DIR=${_face_application_consumer_root}/build"
        "-DKFCORE_CONSUMER_INSTALL_PREFIX=${_face_application_consumer_root}/install"
        "-DKFCORE_CONSUMER_KFCORE_DIR=${_face_application_consumer_root}/install/${CMAKE_INSTALL_LIBDIR}/cmake/KFCore"
        "-DKFCORE_CONSUMER_BUILD_CONFIG=$<CONFIG>"
        "-DKFCORE_CONSUMER_NINJA_EXECUTABLE=${_face_application_consumer_ninja}"
        "-DKFCORE_CONSUMER_CUDATOOLKIT_ROOT=${CUDAToolkit_ROOT}"
        "-DKFCORE_CONSUMER_TENSORRT_ROOT=$ENV{TENSORRT_ROOT}"
        "-DKFCORE_CONSUMER_OPENCV_LITE_ROOT=$ENV{OPENCV_LITE_ROOT}"
        "-DKFCORE_CONSUMER_TURBOUTILS_ROOT=$ENV{TURBOUTILS_ROOT}"
        -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_face_application_installed_consumer.cmake")
  endif()
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
