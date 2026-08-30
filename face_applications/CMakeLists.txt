add_library(kfcore_face_applications SHARED
  src/composer.cpp
  src/error.cpp
  src/geometry.cpp)
target_sources(kfcore_face_applications PRIVATE src/tensorrt.cpp)
target_include_directories(kfcore_face_applications
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_face_applications PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_applications
  PUBLIC
    KFCore::image_processor
    KFCore::face_models
    KFCore::tensorrt_yolo)
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

if(BUILD_EXAMPLES)
  find_package(OpenCVLite REQUIRED COMPONENTS core imgproc imgcodecs highgui)
  add_executable(face_swap_image
    examples/face_swap_cli.cpp
    examples/face_swap_image.cpp)
  target_include_directories(face_swap_image PRIVATE examples)
  target_link_libraries(face_swap_image PRIVATE
    KFCore::face_applications
    OpenCVLite::core
    OpenCVLite::imgcodecs
    TurboUtils::Core)
  set_target_properties(face_swap_image PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/examples")

  add_executable(face_swap_demo
    examples/face_swap_cli.cpp
    examples/face_swap_demo.cpp
    examples/face_swap_demo_metrics.cpp
    examples/face_swap_demo_ui.cpp)
  target_include_directories(face_swap_demo PRIVATE examples)
  target_link_libraries(face_swap_demo PRIVATE
    KFCore::face_applications
    OpenCVLite::core
    OpenCVLite::imgproc
    OpenCVLite::imgcodecs
    OpenCVLite::highgui
    TurboUtils::Core)
  set_target_properties(face_swap_demo PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/examples")
endif()

if(BUILD_TESTS)
  find_package(OpenCVLite REQUIRED COMPONENTS core imgproc)
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

  add_executable(test_face_swap_demo_ui
    tests/test_face_swap_demo_ui.cpp
    examples/face_swap_demo_metrics.cpp
    examples/face_swap_demo_ui.cpp)
  target_include_directories(test_face_swap_demo_ui PRIVATE examples)
  target_link_libraries(test_face_swap_demo_ui PRIVATE
    KFCore::face_applications
    OpenCVLite::core
    OpenCVLite::imgproc
    TurboUtils::TinyTest)
  set_target_properties(test_face_swap_demo_ui PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/tests")
  add_test(NAME test_face_swap_demo_ui COMMAND test_face_swap_demo_ui)

  add_executable(test_face_geometry tests/test_geometry.cpp)
  target_link_libraries(test_face_geometry PRIVATE
    KFCore::face_applications
    TurboUtils::TinyTest)
  set_target_properties(test_face_geometry PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/tests")
  add_test(NAME test_face_geometry COMMAND test_face_geometry)

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

  set(_kfcore_face_application_required_files
    KFCORE_FACE_APPLICATION_TEST_ENGINE_12FACE
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER
    KFCORE_FACE_APPLICATION_TEST_MATRIX
    KFCORE_FACE_APPLICATION_TEST_SOURCE_IMAGE
    KFCORE_FACE_APPLICATION_TEST_TARGET_IMAGE)
  set(_kfcore_face_application_has_assets OFF)
  foreach(_kfcore_file_variable IN LISTS _kfcore_face_application_required_files)
    if(DEFINED ${_kfcore_file_variable} AND
       NOT "${${_kfcore_file_variable}}" STREQUAL "")
      set(_kfcore_face_application_has_assets ON)
      break()
    endif()
  endforeach()

  if(_kfcore_face_application_has_assets)
    find_package(OpenCVLite REQUIRED COMPONENTS imgcodecs)
    foreach(_kfcore_file_variable IN LISTS _kfcore_face_application_required_files)
      if(NOT IS_ABSOLUTE "${${_kfcore_file_variable}}" OR
         NOT EXISTS "${${_kfcore_file_variable}}" OR
         IS_DIRECTORY "${${_kfcore_file_variable}}")
        message(FATAL_ERROR
          "${_kfcore_file_variable} must name an existing absolute file when "
          "face-application integration assets are configured")
      endif()
    endforeach()

    function(_kfcore_configure_face_application_test_assets target)
      target_compile_definitions(${target} PRIVATE
        "KFCORE_TEST_12FACE_ENGINE=\"${KFCORE_FACE_APPLICATION_TEST_ENGINE_12FACE}\""
        "KFCORE_TEST_FACE68_ENGINE=\"${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68}\""
        "KFCORE_TEST_ARCFACE_ENGINE=\"${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE}\""
        "KFCORE_TEST_INSWAPPER_ENGINE=\"${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER}\""
        "KFCORE_TEST_GFPGAN_ENGINE=\"${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN}\""
        "KFCORE_TEST_AGE_GENDER_ENGINE=\"${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER}\""
        "KFCORE_TEST_INSWAPPER_MATRIX=\"${KFCORE_FACE_APPLICATION_TEST_MATRIX}\""
        "KFCORE_TEST_SOURCE_IMAGE=\"${KFCORE_FACE_APPLICATION_TEST_SOURCE_IMAGE}\""
        "KFCORE_TEST_TARGET_IMAGE=\"${KFCORE_FACE_APPLICATION_TEST_TARGET_IMAGE}\"")
    endfunction()

    add_executable(test_face_swap_integration tests/test_face_swap_integration.cpp)
    target_include_directories(test_face_swap_integration PRIVATE examples)
    target_link_libraries(test_face_swap_integration PRIVATE
      KFCore::face_applications
      OpenCVLite::core
      OpenCVLite::imgcodecs
      OpenCVLite::imgproc
      TurboUtils::TinyTest)
    _kfcore_configure_face_application_test_assets(test_face_swap_integration)
    set_target_properties(test_face_swap_integration PROPERTIES
      CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
      FOLDER "face_applications/tests")
    add_test(NAME test_face_swap_integration COMMAND test_face_swap_integration)
    if(WIN32)
      set_tests_properties(test_face_swap_integration PROPERTIES
        ENVIRONMENT_MODIFICATION
          "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR};PATH=path_list_prepend:$ENV{OPENCV_LITE_ROOT}/bin;PATH=path_list_prepend:$ENV{TURBOUTILS_ROOT}/bin")
    endif()

    add_executable(benchmark_face_swap_pipeline
      tests/benchmark_face_swap_pipeline.cpp
      examples/face_swap_demo_metrics.cpp)
    target_include_directories(benchmark_face_swap_pipeline PRIVATE examples)
    target_link_libraries(benchmark_face_swap_pipeline PRIVATE
      KFCore::face_applications
      OpenCVLite::core
      OpenCVLite::imgcodecs
      TurboUtils::TinyTest)
    _kfcore_configure_face_application_test_assets(benchmark_face_swap_pipeline)
    set_target_properties(benchmark_face_swap_pipeline PROPERTIES
      CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
      FOLDER "face_applications/benchmarks")
  endif()

  if(WIN32)
    set_tests_properties(test_face_swap_cli test_face_swap_demo_ui test_face_geometry test_face_composer
      test_face_application_validation PROPERTIES
      ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR};PATH=path_list_prepend:$ENV{OPENCV_LITE_ROOT}/bin;PATH=path_list_prepend:$ENV{TURBOUTILS_ROOT}/bin")
  endif()

  set(_face_application_consumer_root
    "${CMAKE_BINARY_DIR}/tests/face_application_installed_consumer")
  add_test(NAME test_face_application_installed_consumer
    COMMAND "${CMAKE_COMMAND}"
      "-DKFCORE_BINARY_DIR=${CMAKE_BINARY_DIR}"
      "-DKFCORE_CONSUMER_SOURCE_DIR=${CMAKE_SOURCE_DIR}/cmake/tests/FaceApplicationInstalledConsumer"
      "-DKFCORE_CONSUMER_BINARY_DIR=${_face_application_consumer_root}/build"
      "-DKFCORE_CONSUMER_INSTALL_PREFIX=${_face_application_consumer_root}/install"
      "-DKFCORE_CONSUMER_KFCORE_DIR=${_face_application_consumer_root}/install/${CMAKE_INSTALL_LIBDIR}/cmake/KFCore"
      -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_face_application_installed_consumer.cmake")
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
