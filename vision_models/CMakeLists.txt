add_library(kfcore_vision_model_core SHARED
  src/decode.cpp
  src/error.cpp
  src/geometry.cpp
  src/pipeline.cpp)
target_include_directories(kfcore_vision_model_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_vision_model_core PUBLIC cxx_std_17)
target_link_libraries(kfcore_vision_model_core
  PUBLIC KFCore::image_core
  PRIVATE KFCore::trackers)
set_target_properties(kfcore_vision_model_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision_models"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_vision_model_core
  ALIAS KFCore::vision_model_core
  EXPORT_NAME vision_model_core
  EXPORT_SET KFCoreTargets
  FOLDER "vision_models")

if(KFCORE_BUILD_VISION_MODELS_CPU)
  add_library(kfcore_vision_models_cpu SHARED
    src/cpu.cpp
    src/onnx_session.cpp)
  target_include_directories(kfcore_vision_models_cpu
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_features(kfcore_vision_models_cpu PUBLIC cxx_std_17)
  target_link_libraries(kfcore_vision_models_cpu
    PUBLIC KFCore::vision_model_core
    PRIVATE ONNXRuntime::ONNXRuntime)
  set_target_properties(kfcore_vision_models_cpu PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "vision_models"
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
  cmake_config_target(kfcore_vision_models_cpu
    ALIAS KFCore::vision_models_cpu
    EXPORT_NAME vision_models_cpu
    EXPORT_SET KFCoreTargets
    FOLDER "vision_models")
  if(WIN32)
    add_custom_command(
      TARGET kfcore_vision_models_cpu
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              $<TARGET_FILE:ONNXRuntime::ONNXRuntime>
              $<TARGET_FILE_DIR:kfcore_vision_models_cpu>)
  endif()
endif()

if(BUILD_TESTS)
  add_executable(test_vision_model_geometry tests/test_geometry.cpp)
  target_include_directories(test_vision_model_geometry PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_vision_model_geometry PRIVATE
    KFCore::vision_model_core
    TurboUtils::TinyTest)
  target_compile_features(test_vision_model_geometry PRIVATE cxx_std_17)
  add_test(NAME test_vision_model_geometry COMMAND test_vision_model_geometry)

  add_executable(test_vision_model_decode tests/test_decode.cpp)
  target_include_directories(test_vision_model_decode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_vision_model_decode PRIVATE
    KFCore::vision_model_core
    TurboUtils::TinyTest)
  target_compile_features(test_vision_model_decode PRIVATE cxx_std_17)
  add_test(NAME test_vision_model_decode COMMAND test_vision_model_decode)

  add_executable(test_vision_model_hand_pipeline tests/test_hand_pipeline.cpp)
  target_include_directories(test_vision_model_hand_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_vision_model_hand_pipeline PRIVATE
    KFCore::vision_model_core
    TurboUtils::TinyTest)
  target_compile_features(test_vision_model_hand_pipeline PRIVATE cxx_std_17)
  add_test(NAME test_vision_model_hand_pipeline COMMAND test_vision_model_hand_pipeline)

  if(KFCORE_BUILD_VISION_MODELS_CPU)
    add_executable(test_vision_models_cpu_validation
      tests/test_cpu_backend_validation.cpp)
    target_link_libraries(test_vision_models_cpu_validation PRIVATE
      KFCore::vision_models_cpu
      TurboUtils::TinyTest)
    target_compile_features(test_vision_models_cpu_validation PRIVATE cxx_std_17)
    add_test(NAME test_vision_models_cpu_validation
      COMMAND test_vision_models_cpu_validation)
    if(WIN32)
      set_tests_properties(test_vision_models_cpu_validation PROPERTIES
        ENVIRONMENT_MODIFICATION
          "PATH=path_list_prepend:$ENV{ONNXRUNTIME_ROOT}/lib")
    endif()

    if(KFCORE_BUILD_VISION_MODELS_CPU_INTEGRATION_TESTS)
      set(_vision_cpu_palm
        "${KFCORE_VISION_MODEL_TEST_MODEL_DIR}/hand_gesture_model/palm_detection/palm_detection_full_inf_post_192x192.onnx")
      set(_vision_cpu_hand
        "${KFCORE_VISION_MODEL_TEST_MODEL_DIR}/hand_gesture_model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx")
      set(_vision_cpu_classifier
        "${KFCORE_VISION_MODEL_TEST_MODEL_DIR}/hand_gesture_model/keypoint_classifier/keypoint_classifier.onnx")
      set(_vision_cpu_face
        "${KFCORE_VISION_MODEL_TEST_MODEL_DIR}/MediaPipeFaceLandmarkDetector.onnx")
      foreach(_vision_cpu_asset IN ITEMS
          "${_vision_cpu_palm}" "${_vision_cpu_hand}"
          "${_vision_cpu_classifier}" "${_vision_cpu_face}"
          "${KFCORE_VISION_MODEL_TEST_HAND_IMAGE}"
          "${KFCORE_VISION_MODEL_TEST_FACE_IMAGE}")
        if(NOT IS_ABSOLUTE "${_vision_cpu_asset}" OR
           NOT EXISTS "${_vision_cpu_asset}" OR IS_DIRECTORY "${_vision_cpu_asset}")
          message(FATAL_ERROR
            "CPU vision-model integration assets must be existing absolute files: ${_vision_cpu_asset}")
        endif()
      endforeach()
      find_path(KFCORE_VISION_CPU_STB_INCLUDE_DIR
        NAMES stb_image.h
        HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
      if(NOT KFCORE_VISION_CPU_STB_INCLUDE_DIR)
        message(FATAL_ERROR
          "CPU vision-model integration tests require stb_image.h")
      endif()

      add_executable(test_vision_models_cpu_integration
        tests/test_cpu_integration.cpp)
      target_include_directories(test_vision_models_cpu_integration PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${KFCORE_VISION_CPU_STB_INCLUDE_DIR})
      target_compile_definitions(test_vision_models_cpu_integration PRIVATE
        "KFCORE_VISION_CPU_TEST_PALM=\"${_vision_cpu_palm}\""
        "KFCORE_VISION_CPU_TEST_HAND=\"${_vision_cpu_hand}\""
        "KFCORE_VISION_CPU_TEST_CLASSIFIER=\"${_vision_cpu_classifier}\""
        "KFCORE_VISION_CPU_TEST_FACE=\"${_vision_cpu_face}\""
        "KFCORE_VISION_CPU_TEST_HAND_IMAGE=\"${KFCORE_VISION_MODEL_TEST_HAND_IMAGE}\""
        "KFCORE_VISION_CPU_TEST_FACE_IMAGE=\"${KFCORE_VISION_MODEL_TEST_FACE_IMAGE}\"")
      target_link_libraries(test_vision_models_cpu_integration PRIVATE
        KFCore::vision_models_cpu
        ONNXRuntime::ONNXRuntime
        TurboUtils::TinyTest)
      target_compile_features(test_vision_models_cpu_integration PRIVATE cxx_std_17)
      add_test(NAME test_vision_models_cpu_integration
        COMMAND test_vision_models_cpu_integration)
      if(WIN32)
        set_tests_properties(test_vision_models_cpu_integration PROPERTIES
          ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$ENV{ONNXRUNTIME_ROOT}/lib")
      endif()
    endif()
  endif()
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
