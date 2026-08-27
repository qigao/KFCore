add_library(kfcore_face_models SHARED
  src/age_gender.cpp
  src/arcface.cpp
  src/contracts.cpp
  src/error.cpp
  src/face68.cpp
  src/gfpgan.cpp
  src/inswapper.cpp
  src/inswapper_embedding.cpp)
target_include_directories(kfcore_face_models
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_face_models PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_models
  PUBLIC KFCore::tensorrt_runtime
  PRIVATE TurboUtils::Core)
set_target_properties(kfcore_face_models PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "face_models"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_face_models
  ALIAS KFCore::face_models
  EXPORT_NAME face_models
  EXPORT_SET KFCoreTargets
  FOLDER "face_models")

if(BUILD_TESTS)
  add_executable(test_face_model_contracts tests/test_contracts.cpp)
  target_include_directories(test_face_model_contracts PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_face_model_contracts PRIVATE
    KFCore::face_models
    TurboUtils::TinyTest)
  set_target_properties(test_face_model_contracts PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "face_models/tests")
  add_test(NAME test_face_model_contracts COMMAND test_face_model_contracts)

  add_executable(test_face_model_results tests/test_results.cpp)
  target_include_directories(test_face_model_results PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_face_model_results PRIVATE
    KFCore::face_models
    TurboUtils::TinyTest)
  set_target_properties(test_face_model_results PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "face_models/tests")
  add_test(NAME test_face_model_results COMMAND test_face_model_results)

  add_executable(test_face_swap_adapter_api tests/test_swap_adapter_api.cpp)
  target_link_libraries(test_face_swap_adapter_api PRIVATE
    KFCore::face_models
    TurboUtils::TinyTest)
  set_target_properties(test_face_swap_adapter_api PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "face_models/tests")
  add_test(NAME test_face_swap_adapter_api COMMAND test_face_swap_adapter_api)

  add_executable(test_inswapper_embedding tests/test_inswapper_embedding.cpp)
  target_link_libraries(test_inswapper_embedding PRIVATE
    KFCore::face_models
    TurboUtils::TinyTest)
  set_target_properties(test_inswapper_embedding PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "face_models/tests")
  add_test(NAME test_inswapper_embedding COMMAND test_inswapper_embedding)

  if(WIN32)
    set_tests_properties(test_face_model_contracts test_face_model_results
      test_face_swap_adapter_api test_inswapper_embedding PROPERTIES
      ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin")
  endif()

  if(KFCORE_INSTALL)
    include("${CMAKE_SOURCE_DIR}/cmake/RequireNinja.cmake")
    kfcore_require_ninja(_tensorrt_runtime_consumer_ninja)

    add_test(NAME test_tensorrt_runtime_installed_consumer_missing_ninja
      COMMAND "${CMAKE_COMMAND}"
        -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_tensorrt_runtime_installed_consumer_missing_ninja.cmake")

    set(_tensorrt_runtime_consumer_root
      "${CMAKE_BINARY_DIR}/tests/tensorrt_runtime_installed_consumer")
    add_test(NAME test_tensorrt_runtime_installed_consumer
      COMMAND "${CMAKE_COMMAND}"
        "-DKFCORE_BINARY_DIR=${CMAKE_BINARY_DIR}"
        "-DKFCORE_CONSUMER_SOURCE_DIR=${CMAKE_SOURCE_DIR}/cmake/tests/TensorRtRuntimeInstalledConsumer"
        "-DKFCORE_CONSUMER_BINARY_DIR=${_tensorrt_runtime_consumer_root}/build"
        "-DKFCORE_CONSUMER_INSTALL_PREFIX=${_tensorrt_runtime_consumer_root}/install"
        "-DKFCORE_CONSUMER_KFCORE_DIR=${_tensorrt_runtime_consumer_root}/install/${CMAKE_INSTALL_LIBDIR}/cmake/KFCore"
        "-DKFCORE_CONSUMER_BUILD_CONFIG=$<CONFIG>"
        "-DKFCORE_CONSUMER_NINJA_EXECUTABLE=${_tensorrt_runtime_consumer_ninja}"
        "-DKFCORE_CONSUMER_CUDATOOLKIT_ROOT=${CUDAToolkit_ROOT}"
        "-DKFCORE_CONSUMER_TENSORRT_ROOT=$ENV{TENSORRT_ROOT}"
        -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_tensorrt_runtime_installed_consumer.cmake")
  endif()

  if(KFCORE_BUILD_TENSORRT_RUNTIME_INTEGRATION_TESTS AND
     (NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE_VALIDATED}" STREQUAL "" OR
      NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER_VALIDATED}" STREQUAL "" OR
      NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68_VALIDATED}" STREQUAL "" OR
      NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER_VALIDATED}" STREQUAL "" OR
      NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN_VALIDATED}" STREQUAL ""))
    add_executable(test_face_models_integration
      tests/test_face_models_integration.cpp)
    target_link_libraries(test_face_models_integration PRIVATE
      KFCore::face_models
      TurboUtils::TinyTest)
    set_target_properties(test_face_models_integration PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "face_models/tests")

    set(_kfcore_face_integration_tests)
    if(NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE_VALIDATED}" STREQUAL "")
      add_test(NAME test_face_models_integration_arcface
        COMMAND test_face_models_integration)
      set_property(TEST test_face_models_integration_arcface PROPERTY ENVIRONMENT
        "KFCORE_FACE_MODEL_TEST_KIND=arcface"
        "KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE=${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE_VALIDATED}")
      list(APPEND _kfcore_face_integration_tests test_face_models_integration_arcface)
    endif()
    if(NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER_VALIDATED}" STREQUAL "")
      add_test(NAME test_face_models_integration_age_gender
        COMMAND test_face_models_integration)
      set_property(TEST test_face_models_integration_age_gender PROPERTY ENVIRONMENT
        "KFCORE_FACE_MODEL_TEST_KIND=age_gender"
        "KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER=${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER_VALIDATED}")
      list(APPEND _kfcore_face_integration_tests test_face_models_integration_age_gender)
    endif()
    if(NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68_VALIDATED}" STREQUAL "")
      add_test(NAME test_face_models_integration_face68
        COMMAND test_face_models_integration)
      set_property(TEST test_face_models_integration_face68 PROPERTY ENVIRONMENT
        "KFCORE_FACE_MODEL_TEST_KIND=face68"
        "KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68=${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68_VALIDATED}")
      list(APPEND _kfcore_face_integration_tests test_face_models_integration_face68)
    endif()
    if(NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER_VALIDATED}" STREQUAL "")
      add_test(NAME test_face_models_integration_inswapper
        COMMAND test_face_models_integration)
      set_property(TEST test_face_models_integration_inswapper PROPERTY ENVIRONMENT
        "KFCORE_FACE_MODEL_TEST_KIND=inswapper"
        "KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER=${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER_VALIDATED}")
      list(APPEND _kfcore_face_integration_tests test_face_models_integration_inswapper)
    endif()
    if(NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN_VALIDATED}" STREQUAL "")
      add_test(NAME test_face_models_integration_gfpgan
        COMMAND test_face_models_integration)
      set_property(TEST test_face_models_integration_gfpgan PROPERTY ENVIRONMENT
        "KFCORE_FACE_MODEL_TEST_KIND=gfpgan"
        "KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN=${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN_VALIDATED}")
      list(APPEND _kfcore_face_integration_tests test_face_models_integration_gfpgan)
    endif()

    if(WIN32)
      set_tests_properties(${_kfcore_face_integration_tests} PROPERTIES
        ENVIRONMENT_MODIFICATION
          "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
    endif()
  endif()
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
