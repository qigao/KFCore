add_library(kfcore_face_models SHARED
  src/age_gender.cpp
  src/arcface.cpp
  src/contracts.cpp
  src/error.cpp
  src/face68.cpp)
target_include_directories(kfcore_face_models
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_face_models PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_models PUBLIC KFCore::tensorrt_runtime)
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

  if(WIN32)
    set_tests_properties(test_face_model_contracts test_face_model_results PROPERTIES
      ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin")
  endif()

  if(KFCORE_BUILD_TENSORRT_RUNTIME_INTEGRATION_TESTS AND
     (NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE_VALIDATED}" STREQUAL "" OR
      NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER_VALIDATED}" STREQUAL "" OR
      NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68_VALIDATED}" STREQUAL ""))
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
