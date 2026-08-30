add_library(kfcore_face_applications_cpu SHARED
  src/composer.cpp
  src/cpu.cpp
  src/geometry.cpp
  src/onnx_session.cpp)
target_sources(kfcore_face_applications_cpu PRIVATE src/preprocess.cpp)
target_include_directories(kfcore_face_applications_cpu
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/face_models/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_face_applications_cpu PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_applications_cpu PUBLIC
  KFCore::image_core
  KFCore::face_model_core
  ONNXRuntime::ONNXRuntime)
set_target_properties(kfcore_face_applications_cpu PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
  FOLDER "face_applications_cpu"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_face_applications_cpu
  ALIAS KFCore::face_applications_cpu
  EXPORT_NAME face_applications_cpu
  EXPORT_SET KFCoreTargets
  FOLDER "face_applications_cpu")

if(BUILD_TESTS)
  foreach(_cpu_face_test IN ITEMS geometry preprocess composer)
    add_executable(test_cpu_face_${_cpu_face_test}
      tests/test_cpu_face_${_cpu_face_test}.cpp)
    target_include_directories(test_cpu_face_${_cpu_face_test} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_cpu_face_${_cpu_face_test} PRIVATE
      KFCore::face_applications_cpu
      TurboUtils::TinyTest)
    set_target_properties(test_cpu_face_${_cpu_face_test} PROPERTIES
      CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
      FOLDER "face_applications_cpu/tests")
    add_test(NAME test_cpu_face_${_cpu_face_test}
      COMMAND test_cpu_face_${_cpu_face_test})
  endforeach()

  add_executable(test_cpu_face_application_validation
    tests/test_cpu_face_application_validation.cpp)
  target_link_libraries(test_cpu_face_application_validation PRIVATE
    KFCore::face_applications_cpu
    TurboUtils::TinyTest)
  set_target_properties(test_cpu_face_application_validation PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications_cpu/tests")
  add_test(NAME test_cpu_face_application_validation
    COMMAND test_cpu_face_application_validation)

  if(WIN32)
    set_tests_properties(
      test_cpu_face_geometry
      test_cpu_face_preprocess
      test_cpu_face_composer
      test_cpu_face_application_validation
      PROPERTIES ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$ENV{ONNXRUNTIME_ROOT}/lib")
  endif()

  if(KFCORE_ENABLE_FACE_APPLICATION_CPU_INTEGRATION_TESTS)
    set(_cpu_face_assets
      "${KFCORE_FACE_CPU_TEST_MODEL_DIR}/yolov12n-face.onnx"
      "${KFCORE_FACE_CPU_TEST_MODEL_DIR}/2dfan4.onnx"
      "${KFCORE_FACE_CPU_TEST_MODEL_DIR}/arcface_w600k_r50.onnx"
      "${KFCORE_FACE_CPU_TEST_MODEL_DIR}/inswapper_128.onnx"
      "${KFCORE_FACE_CPU_TEST_MODEL_DIR}/gfpgan_1.4.onnx"
      "${KFCORE_FACE_CPU_TEST_MODEL_DIR}/age-gender.onnx"
      "${KFCORE_FACE_CPU_TEST_MATRIX}"
      "${KFCORE_FACE_CPU_TEST_SOURCE_IMAGE}"
      "${KFCORE_FACE_CPU_TEST_TARGET_IMAGE}")
    foreach(_cpu_face_asset IN LISTS _cpu_face_assets)
      if(NOT IS_ABSOLUTE "${_cpu_face_asset}" OR
         NOT EXISTS "${_cpu_face_asset}" OR IS_DIRECTORY "${_cpu_face_asset}")
        message(FATAL_ERROR
          "CPU face integration assets must be existing absolute files: ${_cpu_face_asset}")
      endif()
    endforeach()
    find_path(KFCORE_CPU_FACE_STB_INCLUDE_DIR
      NAMES stb_image.h
      HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
    if(NOT KFCORE_CPU_FACE_STB_INCLUDE_DIR)
      message(FATAL_ERROR
        "CPU face integration tests require the stb_image.h test dependency")
    endif()

    function(kfcore_configure_cpu_face_asset_target target_name)
      target_include_directories(${target_name} PRIVATE
        ${KFCORE_CPU_FACE_STB_INCLUDE_DIR})
      target_link_libraries(${target_name} PRIVATE
        KFCore::face_applications_cpu
        TurboUtils::TinyTest)
      target_compile_definitions(${target_name} PRIVATE
        "KFCORE_FACE_CPU_TEST_DETECTOR=\"${KFCORE_FACE_CPU_TEST_MODEL_DIR}/yolov12n-face.onnx\""
        "KFCORE_FACE_CPU_TEST_FACE68=\"${KFCORE_FACE_CPU_TEST_MODEL_DIR}/2dfan4.onnx\""
        "KFCORE_FACE_CPU_TEST_ARCFACE=\"${KFCORE_FACE_CPU_TEST_MODEL_DIR}/arcface_w600k_r50.onnx\""
        "KFCORE_FACE_CPU_TEST_INSWAPPER=\"${KFCORE_FACE_CPU_TEST_MODEL_DIR}/inswapper_128.onnx\""
        "KFCORE_FACE_CPU_TEST_GFPGAN=\"${KFCORE_FACE_CPU_TEST_MODEL_DIR}/gfpgan_1.4.onnx\""
        "KFCORE_FACE_CPU_TEST_AGE_GENDER=\"${KFCORE_FACE_CPU_TEST_MODEL_DIR}/age-gender.onnx\""
        "KFCORE_FACE_CPU_TEST_MATRIX=\"${KFCORE_FACE_CPU_TEST_MATRIX}\""
        "KFCORE_FACE_CPU_TEST_SOURCE_IMAGE=\"${KFCORE_FACE_CPU_TEST_SOURCE_IMAGE}\""
        "KFCORE_FACE_CPU_TEST_TARGET_IMAGE=\"${KFCORE_FACE_CPU_TEST_TARGET_IMAGE}\"")
      set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
        FOLDER "face_applications_cpu/tests")
      if(WIN32)
        add_custom_command(
          TARGET ${target_name}
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  $<TARGET_FILE:ONNXRuntime::ONNXRuntime>
                  $<TARGET_FILE_DIR:${target_name}>)
      endif()
    endfunction()

    add_executable(test_cpu_face_pipeline_integration
      tests/test_cpu_face_pipeline_integration.cpp)
    kfcore_configure_cpu_face_asset_target(test_cpu_face_pipeline_integration)
    add_test(NAME test_cpu_face_pipeline_integration
      COMMAND test_cpu_face_pipeline_integration)
    if(WIN32)
      set_tests_properties(test_cpu_face_pipeline_integration PROPERTIES
        ENVIRONMENT_MODIFICATION
          "PATH=path_list_prepend:$ENV{ONNXRUNTIME_ROOT}/lib")
    endif()

    add_executable(benchmark_cpu_face_pipeline
      tests/benchmark_cpu_pipeline.cpp)
    kfcore_configure_cpu_face_asset_target(benchmark_cpu_face_pipeline)
  endif()
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
