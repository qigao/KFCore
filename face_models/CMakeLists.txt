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
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
