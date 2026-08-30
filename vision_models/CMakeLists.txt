add_library(kfcore_vision_model_core SHARED
  src/decode.cpp
  src/error.cpp
  src/geometry.cpp
  src/hand_appearance.cpp
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

add_library(kfcore_vision_models_tensorrt SHARED
    src/tensorrt.cpp
    src/tensorrt_models.cpp)
  target_include_directories(kfcore_vision_models_tensorrt
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_features(kfcore_vision_models_tensorrt PUBLIC cxx_std_17)
  target_link_libraries(kfcore_vision_models_tensorrt
    PUBLIC KFCore::vision_model_core
    PRIVATE
      KFCore::image_processor
      KFCore::tensorrt_runtime
      KFCore::tensorrt_yolo)
  set_target_properties(kfcore_vision_models_tensorrt PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "vision_models"
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
  cmake_config_target(kfcore_vision_models_tensorrt
    ALIAS KFCore::vision_models_tensorrt
    EXPORT_NAME vision_models_tensorrt
  EXPORT_SET KFCoreTargets
  FOLDER "vision_models")

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

    if(KFCORE_ENABLE_VISION_MODELS_CPU_INTEGRATION_TESTS)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}"
        "hand_gesture_model/palm_detection/palm_detection_full_inf_post_192x192.onnx"
        _vision_cpu_palm)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}"
        "hand_gesture_model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx"
        _vision_cpu_hand)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}"
        "hand_gesture_model/keypoint_classifier/keypoint_classifier.onnx"
        _vision_cpu_classifier)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}" "MediaPipeFaceLandmarkDetector.onnx"
        _vision_cpu_face)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}" "yolov12n-face.onnx"
        _vision_cpu_face_detector)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}" "${KFCORE_VISION_MODEL_TEST_HAND_IMAGE_RELATIVE}"
        _vision_cpu_hand_image)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}" "${KFCORE_VISION_MODEL_TEST_FACE_IMAGE_RELATIVE}"
        _vision_cpu_face_image)
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
        "KFCORE_VISION_CPU_TEST_FACE_DETECTOR=\"${_vision_cpu_face_detector}\""
        "KFCORE_VISION_CPU_TEST_HAND_IMAGE=\"${_vision_cpu_hand_image}\""
        "KFCORE_VISION_CPU_TEST_FACE_IMAGE=\"${_vision_cpu_face_image}\"")
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

  add_executable(test_vision_models_tensorrt_validation
      tests/test_tensorrt_backend_validation.cpp)
    target_include_directories(test_vision_models_tensorrt_validation PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_vision_models_tensorrt_validation PRIVATE
      KFCore::vision_models_tensorrt
      TurboUtils::TinyTest)
    target_compile_features(test_vision_models_tensorrt_validation PRIVATE cxx_std_17)
    add_test(NAME test_vision_models_tensorrt_validation
      COMMAND test_vision_models_tensorrt_validation)
    if(WIN32)
      set_tests_properties(test_vision_models_tensorrt_validation PROPERTIES
        ENVIRONMENT_MODIFICATION
          "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
    endif()

    if(KFCORE_ENABLE_VISION_MODELS_TENSORRT_INTEGRATION_TESTS)
      set(_vision_trt_profile "$ENV{KFCORE_TENSORRT_ENGINE_PROFILE}")
      set(_vision_trt_hand_root
        "hand_gesture_model/tensorrt/${_vision_trt_profile}")
      set(_vision_trt_face_root "tensorrt/${_vision_trt_profile}")
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}" "${_vision_trt_hand_root}/palm_detection.engine"
        _vision_trt_palm)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}" "${_vision_trt_hand_root}/hand_landmark.engine"
        _vision_trt_hand)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}" "${_vision_trt_hand_root}/keypoint_classifier.engine"
        _vision_trt_classifier)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}" "${_vision_trt_face_root}/face_landmark.engine"
        _vision_trt_face)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}" "${_vision_trt_face_root}/yolov12n-face.engine"
        _vision_trt_face_detector)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}" "${KFCORE_VISION_MODEL_TEST_HAND_IMAGE_RELATIVE}"
        _vision_trt_hand_image)
      kfcore_resolve_model_path(
        "$ENV{KFCORE_MODEL_ROOT}" "${KFCORE_VISION_MODEL_TEST_FACE_IMAGE_RELATIVE}"
        _vision_trt_face_image)
      find_path(KFCORE_VISION_TRT_STB_INCLUDE_DIR
        NAMES stb_image.h
        HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
      if(NOT KFCORE_VISION_TRT_STB_INCLUDE_DIR)
        message(FATAL_ERROR
          "TensorRT vision-model integration tests require stb_image.h")
      endif()

      add_executable(test_vision_models_tensorrt_integration
        tests/test_tensorrt_integration.cpp)
      target_include_directories(test_vision_models_tensorrt_integration PRIVATE
        ${KFCORE_VISION_TRT_STB_INCLUDE_DIR})
      target_compile_definitions(test_vision_models_tensorrt_integration PRIVATE
        "KFCORE_VISION_TRT_TEST_PALM=\"${_vision_trt_palm}\""
        "KFCORE_VISION_TRT_TEST_HAND=\"${_vision_trt_hand}\""
        "KFCORE_VISION_TRT_TEST_CLASSIFIER=\"${_vision_trt_classifier}\""
        "KFCORE_VISION_TRT_TEST_FACE=\"${_vision_trt_face}\""
        "KFCORE_VISION_TRT_TEST_FACE_DETECTOR=\"${_vision_trt_face_detector}\""
        "KFCORE_VISION_TRT_TEST_HAND_IMAGE=\"${_vision_trt_hand_image}\""
        "KFCORE_VISION_TRT_TEST_FACE_IMAGE=\"${_vision_trt_face_image}\"")
      target_link_libraries(test_vision_models_tensorrt_integration PRIVATE
        KFCore::vision_models_tensorrt
        CUDA::cudart
        TurboUtils::TinyTest)
      target_compile_features(test_vision_models_tensorrt_integration PRIVATE cxx_std_17)
      add_test(NAME test_vision_models_tensorrt_integration
        COMMAND test_vision_models_tensorrt_integration)
      if(WIN32)
        set_tests_properties(test_vision_models_tensorrt_integration PROPERTIES
          ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
      endif()
    endif()

  set(_vision_models_consumer_kind combined)
  set(_vision_models_consumer_root
    "${CMAKE_BINARY_DIR}/tests/vision_models_${_vision_models_consumer_kind}_installed_consumer")
  add_test(NAME test_vision_models_${_vision_models_consumer_kind}_installed_consumer
    COMMAND "${CMAKE_COMMAND}"
      "-DKFCORE_BINARY_DIR=${CMAKE_BINARY_DIR}"
      "-DKFCORE_CONSUMER_SOURCE_DIR=${CMAKE_SOURCE_DIR}/cmake/tests/VisionModelsInstalledConsumer"
      "-DKFCORE_CONSUMER_BINARY_DIR=${_vision_models_consumer_root}/build"
      "-DKFCORE_CONSUMER_INSTALL_PREFIX=${_vision_models_consumer_root}/install"
      "-DKFCORE_CONSUMER_KFCORE_DIR=${_vision_models_consumer_root}/install/${CMAKE_INSTALL_LIBDIR}/cmake/KFCore"
      -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_vision_models_installed_consumer.cmake")
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
