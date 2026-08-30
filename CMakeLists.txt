cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

project(
  KFCore
  VERSION 0.1.0
  DESCRIPTION "Kalman filter implementations in C"
  HOMEPAGE_URL "https://github.com/qigao/KFCore"
  LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")
include(CTest)
include(CMakeOptions.cmake)
include(GNUInstallDirs)
include(Sanitizers)
include(CmakeUtils)
include(ValidateTensorRtIntegrationEngine)
include(ModelPaths)

if(KFCORE_ENABLE_TENSORRT_INTEGRATION_TESTS)
  if(NOT DEFINED KFCORE_TENSORRT_TEST_ENGINE_RELATIVE OR
     "${KFCORE_TENSORRT_TEST_ENGINE_RELATIVE}" STREQUAL "")
    message(FATAL_ERROR
      "KFCORE_TENSORRT_TEST_ENGINE_RELATIVE is required when "
      "KFCORE_ENABLE_TENSORRT_INTEGRATION_TESTS is enabled")
  endif()
  kfcore_resolve_model_path(
    "$ENV{KFCORE_MODEL_ROOT}"
    "${KFCORE_TENSORRT_TEST_ENGINE_RELATIVE}"
    KFCORE_TENSORRT_TEST_ENGINE)
  kfcore_resolve_optional_model_path(
    "$ENV{KFCORE_MODEL_ROOT}"
    "${KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE_RELATIVE}"
    KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE)
  kfcore_resolve_optional_model_path(
    "$ENV{KFCORE_MODEL_ROOT}"
    "${KFCORE_TENSORRT_TEST_ENGINE_FP16_RELATIVE}"
    KFCORE_TENSORRT_TEST_ENGINE_FP16)
  kfcore_validate_tensorrt_integration_engine(
    "${KFCORE_TENSORRT_TEST_ENGINE}"
    KFCORE_TENSORRT_TEST_ENGINE_VALIDATED)
  kfcore_validate_optional_tensorrt_integration_engine(
    "${KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE}"
    KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE_VALIDATED
    KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE)
endif()

if(KFCORE_ENABLE_TENSORRT_RUNTIME_INTEGRATION_TESTS)
  set(_kfcore_tensorrt_engine_root
    "tensorrt/$ENV{KFCORE_TENSORRT_ENGINE_PROFILE}")
  set(_kfcore_hand_engine_root
    "hand_gesture_model/tensorrt/$ENV{KFCORE_TENSORRT_ENGINE_PROFILE}")
  kfcore_resolve_model_path(
    "$ENV{KFCORE_MODEL_ROOT}"
    "${_kfcore_hand_engine_root}/palm_detection.engine"
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_PALM)
  kfcore_resolve_model_path(
    "$ENV{KFCORE_MODEL_ROOT}"
    "${_kfcore_tensorrt_engine_root}/2dfan4.engine"
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68)
  kfcore_resolve_model_path(
    "$ENV{KFCORE_MODEL_ROOT}"
    "${_kfcore_tensorrt_engine_root}/arcface_w600k_r50.engine"
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE)
  kfcore_resolve_model_path(
    "$ENV{KFCORE_MODEL_ROOT}"
    "${_kfcore_tensorrt_engine_root}/age-gender.engine"
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER)
  kfcore_resolve_model_path(
    "$ENV{KFCORE_MODEL_ROOT}"
    "${_kfcore_tensorrt_engine_root}/inswapper_128.engine"
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER)
  kfcore_resolve_model_path(
    "$ENV{KFCORE_MODEL_ROOT}"
    "${_kfcore_tensorrt_engine_root}/gfpgan_1.4.engine"
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN)
  kfcore_validate_optional_tensorrt_runtime_engine(
    "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE}"
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE
    ArcFace
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE_VALIDATED)
  kfcore_validate_optional_tensorrt_runtime_engine(
    "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER}"
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER
    AgeGender
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER_VALIDATED)
  kfcore_validate_optional_tensorrt_runtime_engine(
    "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68}"
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68
    Face68
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68_VALIDATED)
  kfcore_validate_optional_tensorrt_runtime_engine(
    "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER}"
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER
    InSwapper
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER_VALIDATED)
  kfcore_validate_optional_tensorrt_runtime_engine(
    "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN}"
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN
    GFPGAN
    KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN_VALIDATED)
endif()

find_package(Threads REQUIRED)

find_package(TurboUtils CONFIG REQUIRED
             PATHS "$ENV{TURBOUTILS_ROOT}" NO_DEFAULT_PATH)
find_package(TurboParser CONFIG REQUIRED
             PATHS "$ENV{TURBOPARSER_ROOT}" NO_DEFAULT_PATH)
find_package(TurboNet CONFIG REQUIRED
             PATHS "$ENV{TURBONET_ROOT}" NO_DEFAULT_PATH)

if(BUILD_TESTS)
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  add_test(NAME test_find_tensorrt
    COMMAND "${CMAKE_COMMAND}"
      "-DKFCORE_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
      "-DKFCORE_TEST_BINARY_DIR=${CMAKE_BINARY_DIR}/tests"
      -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_find_tensorrt.cmake")
  add_test(NAME test_find_opencv_lite
    COMMAND "${CMAKE_COMMAND}"
      "-DKFCORE_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
      "-DKFCORE_TEST_BINARY_DIR=${CMAKE_BINARY_DIR}/tests"
      -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_find_opencv_lite.cmake")
  add_test(NAME test_find_onnxruntime
    COMMAND "${CMAKE_COMMAND}"
      "-DKFCORE_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
      "-DKFCORE_TEST_BINARY_DIR=${CMAKE_BINARY_DIR}/tests"
      -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_find_onnxruntime.cmake")
  add_test(NAME test_tensorrt_integration_engine_config
    COMMAND "${CMAKE_COMMAND}"
      "-DKFCORE_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
      "-DKFCORE_TEST_BINARY_DIR=${CMAKE_BINARY_DIR}/tests"
      -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_tensorrt_integration_engine_config.cmake")
  add_test(NAME test_safe_test_directory
    COMMAND "${CMAKE_COMMAND}"
      "-DKFCORE_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
      "-DKFCORE_TEST_BINARY_DIR=${CMAKE_BINARY_DIR}/tests"
      -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_safe_test_directory.cmake")
  add_test(NAME test_model_paths
    COMMAND "${CMAKE_COMMAND}"
      "-DKFCORE_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
      "-DKFCORE_TEST_BINARY_DIR=${CMAKE_BINARY_DIR}/tests"
      -P "${CMAKE_SOURCE_DIR}/cmake/tests/test_model_paths.cmake")
endif()

add_subdirectory(vendor)
add_subdirectory(trackers)

set(KFCORE_IMAGE_PROCESSOR_CUDA_ENABLED ON)
set(KFCORE_IMAGE_CORE_ENABLED ON)
enable_language(CXX)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED MODULE)
find_package(ONNXRuntime REQUIRED MODULE)

add_subdirectory(image_processor)
add_subdirectory(thig)
add_subdirectory(sift)
add_subdirectory(tensorrt_yolo)
add_subdirectory(tensorrt_runtime)
add_subdirectory(face_models)
add_subdirectory(vision_models)
add_subdirectory(hand_interaction)
add_subdirectory(face_applications)
add_subdirectory(face_applications_cpu)

set(APRILTAG_BUILD_EXAMPLES "${BUILD_EXAMPLES}" CACHE BOOL
    "Build AprilTag example executables" FORCE)
set(APRILTAG_BUILD_PYTHON_WRAPPER OFF CACHE BOOL
    "Build AprilTag Python wrapper" FORCE)
set(APRILTAG_BUILD_TESTS OFF CACHE BOOL
    "Build AprilTag upstream tests" FORCE)
set(APRILTAG_INSTALL OFF CACHE BOOL "Install AprilTag from the vendored subproject" FORCE)
add_subdirectory(apriltag)
add_subdirectory(kalman) 

include(CMakePackageConfigHelpers)

  install(
    TARGETS kfcore  
    EXPORT KFCoreTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

  install(FILES kalman/kalman_takasu.h kalman/kalman_udu.h kalman/kalman_ekf.h
                kalman/kalman_ukf.h kalman/signal_filters.h
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kfcore")

  install(FILES kalman/navigation_tools/frame_transform.h
                kalman/navigation_tools/navtoolbox.h
                kalman/navigation_tools/nav_fusion2d.h
                kalman/navigation_tools/nav_fusion3d.h
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kfcore/navigation_tools")

  install(FILES cmake/FindTensorRT.cmake
                cmake/FindOpenCVLite.cmake
                cmake/FindONNXRuntime.cmake
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KFCore")

  install(
    EXPORT KFCoreTargets
    NAMESPACE KFCore::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KFCore")

  configure_package_config_file(
    cmake/KFCoreConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/KFCoreConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KFCore")

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/KFCoreConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/KFCoreConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/KFCoreConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KFCore")
