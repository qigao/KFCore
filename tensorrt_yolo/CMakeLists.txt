add_library(kfcore_yolo_tracking SHARED
  src/engine_contract.cpp
  src/tracking.cpp)

target_link_libraries(kfcore_yolo_tracking PRIVATE KFCore::trackers)
target_include_directories(kfcore_yolo_tracking PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(kfcore_yolo_tracking PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "tensorrt_yolo"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_yolo_tracking
  ALIAS KFCore::yolo_tracking
  EXPORT_NAME yolo_tracking
  EXPORT_SET KFCoreTargets
  FOLDER "tensorrt_yolo")

if(BUILD_TESTS AND KFCORE_INSTALL)
  if(ENABLE_SANITIZER_ADDRESS)
    set(_yolo_consumer_enable_sanitizer_address ON)
  else()
    set(_yolo_consumer_enable_sanitizer_address OFF)
  endif()
  set(_yolo_tracking_consumer_source_dir
    "${CMAKE_BINARY_DIR}/tests/yolo_tracking_installed_consumer/source")
  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/tests/YoloTrackingInstalledConsumer/CMakeLists.txt"
    "${_yolo_tracking_consumer_source_dir}/CMakeLists.txt"
    COPYONLY)
  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/tests/YoloTrackingInstalledConsumer/main.cpp"
    "${_yolo_tracking_consumer_source_dir}/main.cpp"
    COPYONLY)
  set(_trackers_consumer_source_dir
    "${CMAKE_BINARY_DIR}/tests/trackers_installed_consumer/source")
  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/tests/TrackersInstalledConsumer/CMakeLists.txt"
    "${_trackers_consumer_source_dir}/CMakeLists.txt"
    COPYONLY)
  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/tests/TrackersInstalledConsumer/main.c"
    "${_trackers_consumer_source_dir}/main.c"
    COPYONLY)
  function(_kfcore_add_installed_consumer_test test_name source_dir work_name mode)
    set(_consumer_root "${CMAKE_BINARY_DIR}/tests/${work_name}")
    add_test(NAME "${test_name}"
      COMMAND "${CMAKE_COMMAND}"
        "-DKFCORE_BINARY_DIR=${CMAKE_BINARY_DIR}"
        "-DKFCORE_CONSUMER_SOURCE_DIR=${source_dir}"
        "-DKFCORE_CONSUMER_BINARY_DIR=${_consumer_root}/build"
        "-DKFCORE_CONSUMER_INSTALL_PREFIX=${_consumer_root}/install"
        "-DKFCORE_CONSUMER_KFCORE_DIR=${_consumer_root}/install/${CMAKE_INSTALL_LIBDIR}/cmake/KFCore"
        "-DKFCORE_CONSUMER_TRACKERS_DIR=${_consumer_root}/install/${CMAKE_INSTALL_LIBDIR}/cmake/trackers"
        "-DKFCORE_CONSUMER_ENABLE_SANITIZER_ADDRESS=${_yolo_consumer_enable_sanitizer_address}"
        "-DKFCORE_CONSUMER_BUILD_CONFIG=$<CONFIG>"
        "-DKFCORE_CONSUMER_MODE=${mode}"
        "-DKFCORE_CONSUMER_HAS_TENSORRT_YOLO=${KFCORE_BUILD_TENSORRT_YOLO}"
        "-DKFCORE_CONSUMER_HAS_YOLO_OPENCV=${KFCORE_BUILD_YOLO_OPENCV}"
        -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_yolo_tracking_installed_consumer.cmake")
  endfunction()

  _kfcore_add_installed_consumer_test(
    test_yolo_tracking_installed_consumer
    "${_yolo_tracking_consumer_source_dir}"
    yolo_tracking_installed_consumer
    kfcore_first)
  _kfcore_add_installed_consumer_test(
    test_trackers_installed_consumer
    "${_trackers_consumer_source_dir}"
    trackers_installed_consumer
    trackers_first)
  _kfcore_add_installed_consumer_test(
    test_yolo_tracking_dependency_first_installed_consumer
    "${_yolo_tracking_consumer_source_dir}"
    yolo_dep_first_consumer
    dependency_first)
  _kfcore_add_installed_consumer_test(
    test_yolo_tracking_repeated_kfcore_installed_consumer
    "${_yolo_tracking_consumer_source_dir}"
    yolo_repeat_consumer
    repeated_kfcore)
endif()

if(KFCORE_BUILD_YOLO_OPENCV)
  find_package(OpenCVLite REQUIRED COMPONENTS core imgproc)

  add_library(kfcore_yolo_opencv SHARED src/opencv.cpp)
  target_link_libraries(kfcore_yolo_opencv
    PUBLIC
      KFCore::yolo_tracking
      OpenCVLite::core
      OpenCVLite::imgproc)
  target_include_directories(kfcore_yolo_opencv PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  set_target_properties(kfcore_yolo_opencv PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_yolo"
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
  cmake_config_target(kfcore_yolo_opencv
    ALIAS KFCore::yolo_opencv
    EXPORT_NAME yolo_opencv
    EXPORT_SET KFCoreTargets
    FOLDER "tensorrt_yolo")
endif()

if(BUILD_TESTS)
  add_executable(test_tensorrt_contract tests/test_tensorrt_contract.cpp)
  target_include_directories(test_tensorrt_contract PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_tensorrt_contract PRIVATE
    KFCore::yolo_tracking
    TurboUtils::TinyTest)
  set_target_properties(test_tensorrt_contract PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_yolo/tests")
  add_test(NAME test_tensorrt_contract COMMAND test_tensorrt_contract)

  add_executable(test_tensorrt_detection_helpers
    tests/test_tensorrt_detection_helpers.cpp
    src/detector_helpers.cpp)
  target_include_directories(test_tensorrt_detection_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_tensorrt_detection_helpers PRIVATE
    KFCore::yolo_tracking
    TurboUtils::TinyTest)
  set_target_properties(test_tensorrt_detection_helpers PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_yolo/tests")
  add_test(NAME test_tensorrt_detection_helpers COMMAND test_tensorrt_detection_helpers)

  add_executable(test_yolo_tracking
    tests/test_yolo_tracking.cpp
    src/tracking.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../trackers/src/tracker.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../trackers/tests/tracker_test_alloc.c)
  target_compile_definitions(test_yolo_tracking PRIVATE KFCORE_TRACKERS_TEST_ALLOCATOR)
  target_include_directories(test_yolo_tracking PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../trackers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../trackers/tests)
  target_link_libraries(test_yolo_tracking PRIVATE
    KFCore::kfcore
    miniblas
    TurboUtils::TinyTest)
  set_target_properties(test_yolo_tracking PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_yolo/tests")
  add_test(NAME test_yolo_tracking COMMAND test_yolo_tracking)
endif()

if(KFCORE_BUILD_YOLO_OPENCV AND BUILD_TESTS)
  add_executable(test_yolo_opencv tests/test_yolo_opencv.cpp)
  target_link_libraries(test_yolo_opencv PRIVATE
    KFCore::yolo_opencv
    TurboUtils::TinyTest)
  set_target_properties(test_yolo_opencv PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_yolo/tests")
  add_test(NAME test_yolo_opencv COMMAND test_yolo_opencv)
endif()

if(KFCORE_BUILD_TENSORRT_YOLO)
  add_library(tensorrt_yolo SHARED
    src/cuda_buffer.cpp
    src/detector_helpers.cpp
    src/detector.cpp
    src/engine_file.cpp
    src/engine.cpp
    src/letterbox.cu)

  target_include_directories(tensorrt_yolo
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(tensorrt_yolo
    PUBLIC KFCore::yolo_tracking
    PRIVATE
      TensorRT::nvinfer
      TensorRT::nvinfer_plugin
      CUDA::cudart)
  set_target_properties(tensorrt_yolo PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_yolo"
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
  cmake_config_target(tensorrt_yolo
    ALIAS KFCore::tensorrt_yolo
    EXPORT_NAME tensorrt_yolo
    EXPORT_SET KFCoreTargets
    FOLDER "tensorrt_yolo")

  if(KFCORE_BUILD_YOLO_OPENCV)
    find_package(OpenCVLite REQUIRED COMPONENTS imgcodecs)
    add_executable(track_image_sequence examples/track_image_sequence.cpp)
    target_link_libraries(track_image_sequence PRIVATE
      KFCore::tensorrt_yolo
      KFCore::yolo_opencv
      OpenCVLite::imgcodecs)
    set_target_properties(track_image_sequence PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "tensorrt_yolo/examples")
  endif()

  if(KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS)
    add_executable(test_cuda_buffer
      tests/test_cuda_buffer.cpp
      src/cuda_buffer.cpp)
    target_include_directories(test_cuda_buffer PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(test_cuda_buffer PRIVATE
      KFCORE_YOLO_CUDA_BUFFER_TESTING)
    target_link_libraries(test_cuda_buffer PRIVATE
      KFCore::yolo_tracking
      CUDA::cudart
      TurboUtils::TinyTest)
    set_target_properties(test_cuda_buffer PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "tensorrt_yolo/tests")
    add_test(NAME test_cuda_buffer COMMAND test_cuda_buffer)

    add_executable(test_engine_file
      tests/test_engine_file.cpp
      src/engine_file.cpp)
    target_include_directories(test_engine_file PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_engine_file PRIVATE
      KFCore::yolo_tracking
      TurboUtils::TinyTest)
    set_target_properties(test_engine_file PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "tensorrt_yolo/tests")
    add_test(NAME test_engine_file COMMAND test_engine_file)

    add_executable(test_tensorrt_integration
      tests/test_tensorrt_integration.cpp)
    target_link_libraries(test_tensorrt_integration PRIVATE
      KFCore::tensorrt_yolo
      TensorRT::nvinfer
      TensorRT::nvinfer_plugin
      CUDA::cudart
      TurboUtils::TinyTest)
    set_target_properties(test_tensorrt_integration PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "tensorrt_yolo/tests")
    kfcore_add_tensorrt_integration_test(
      test_tensorrt_integration
      "${KFCORE_TENSORRT_TEST_ENGINE_VALIDATED}"
      test_tensorrt_integration)
    kfcore_add_optional_tensorrt_integration_test(
      test_tensorrt_integration_yolo11_face
      "${KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE_VALIDATED}"
      test_tensorrt_integration)

    if(DEFINED ENV{KFCORE_TENSORRT_TEST_ENGINE_FP16}
       AND NOT "$ENV{KFCORE_TENSORRT_TEST_ENGINE_FP16}" STREQUAL "")
      add_executable(test_tensorrt_integration_fp16
        tests/test_tensorrt_integration.cpp)
      target_compile_definitions(test_tensorrt_integration_fp16 PRIVATE
        KFCORE_TENSORRT_TEST_FP16)
      target_link_libraries(test_tensorrt_integration_fp16 PRIVATE
        KFCore::tensorrt_yolo
        TensorRT::nvinfer
        TensorRT::nvinfer_plugin
        CUDA::cudart
        TurboUtils::TinyTest)
      set_target_properties(test_tensorrt_integration_fp16 PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tensorrt_yolo/tests")
      add_test(NAME test_tensorrt_integration_fp16
        COMMAND test_tensorrt_integration_fp16)
      set_tests_properties(test_tensorrt_integration_fp16 PROPERTIES
        ENVIRONMENT
          "KFCORE_TENSORRT_TEST_ENGINE_FP16=$ENV{KFCORE_TENSORRT_TEST_ENGINE_FP16}")
    endif()

    add_executable(test_letterbox_cuda
      tests/test_letterbox_cuda.cu)
    target_include_directories(test_letterbox_cuda PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_letterbox_cuda PRIVATE
      KFCore::tensorrt_yolo
      CUDA::cudart
      TurboUtils::TinyTest)
    set_target_properties(test_letterbox_cuda PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "tensorrt_yolo/tests")
    add_test(NAME test_letterbox_cuda COMMAND test_letterbox_cuda)
  endif()

endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
