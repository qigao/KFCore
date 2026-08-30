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
    KFCORE_FACE_APPLICATION_TEST_MATRIX_RELATIVE
    KFCORE_FACE_APPLICATION_TEST_SOURCE_IMAGE_RELATIVE
    KFCORE_FACE_APPLICATION_TEST_TARGET_IMAGE_RELATIVE)
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
      if(NOT DEFINED ${_kfcore_file_variable} OR
         "${${_kfcore_file_variable}}" STREQUAL "")
        message(FATAL_ERROR
          "${_kfcore_file_variable} is required when face-application "
          "integration assets are configured")
      endif()
    endforeach()

    set(_kfcore_face_engine_root
      "tensorrt/$ENV{KFCORE_TENSORRT_ENGINE_PROFILE}")
    kfcore_resolve_model_path(
      "$ENV{KFCORE_MODEL_ROOT}"
      "${_kfcore_face_engine_root}/yolov12n-face.engine"
      _kfcore_face_detector_engine)
    kfcore_resolve_model_path(
      "$ENV{KFCORE_MODEL_ROOT}"
      "${_kfcore_face_engine_root}/2dfan4.engine"
      _kfcore_face68_engine)
    kfcore_resolve_model_path(
      "$ENV{KFCORE_MODEL_ROOT}"
      "${_kfcore_face_engine_root}/arcface_w600k_r50.engine"
      _kfcore_arcface_engine)
    kfcore_resolve_model_path(
      "$ENV{KFCORE_MODEL_ROOT}"
      "${_kfcore_face_engine_root}/inswapper_128.engine"
      _kfcore_inswapper_engine)
    kfcore_resolve_model_path(
      "$ENV{KFCORE_MODEL_ROOT}"
      "${KFCORE_FACE_APPLICATION_TEST_MATRIX_RELATIVE}"
      _kfcore_inswapper_matrix)
    kfcore_resolve_model_path(
      "$ENV{KFCORE_MODEL_ROOT}"
      "${KFCORE_FACE_APPLICATION_TEST_SOURCE_IMAGE_RELATIVE}"
      _kfcore_face_source_image)
    kfcore_resolve_model_path(
      "$ENV{KFCORE_MODEL_ROOT}"
      "${KFCORE_FACE_APPLICATION_TEST_TARGET_IMAGE_RELATIVE}"
      _kfcore_face_target_image)
    kfcore_resolve_model_path(
      "$ENV{KFCORE_MODEL_ROOT}"
      "${_kfcore_face_engine_root}/gfpgan_1.4.engine"
      _kfcore_gfpgan_engine)
    kfcore_resolve_model_path(
      "$ENV{KFCORE_MODEL_ROOT}"
      "${_kfcore_face_engine_root}/age-gender.engine"
      _kfcore_age_gender_engine)

    function(_kfcore_configure_face_application_test_assets target)
      target_compile_definitions(${target} PRIVATE
        "KFCORE_TEST_12FACE_ENGINE=\"${_kfcore_face_detector_engine}\""
        "KFCORE_TEST_FACE68_ENGINE=\"${_kfcore_face68_engine}\""
        "KFCORE_TEST_ARCFACE_ENGINE=\"${_kfcore_arcface_engine}\""
        "KFCORE_TEST_INSWAPPER_ENGINE=\"${_kfcore_inswapper_engine}\""
        "KFCORE_TEST_GFPGAN_ENGINE=\"${_kfcore_gfpgan_engine}\""
        "KFCORE_TEST_AGE_GENDER_ENGINE=\"${_kfcore_age_gender_engine}\""
        "KFCORE_TEST_INSWAPPER_MATRIX=\"${_kfcore_inswapper_matrix}\""
        "KFCORE_TEST_SOURCE_IMAGE=\"${_kfcore_face_source_image}\""
        "KFCORE_TEST_TARGET_IMAGE=\"${_kfcore_face_target_image}\"")
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
