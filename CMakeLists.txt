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
include(CMakeOptions.cmake)
include(GNUInstallDirs)
include(Sanitizers)
include(CmakeUtils)
include(ValidateTensorRtIntegrationEngine)

if(KFCORE_BUILD_TENSORRT_YOLO AND NOT KFCORE_BUILD_YOLO_TRACKING)
  message(FATAL_ERROR
    "KFCORE_BUILD_TENSORRT_YOLO requires KFCORE_BUILD_YOLO_TRACKING=ON")
endif()

if(KFCORE_BUILD_YOLO_OPENCV AND NOT KFCORE_BUILD_YOLO_TRACKING)
  message(FATAL_ERROR
    "KFCORE_BUILD_YOLO_OPENCV requires KFCORE_BUILD_YOLO_TRACKING=ON")
endif()

if(KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS AND NOT KFCORE_BUILD_TENSORRT_YOLO)
  message(FATAL_ERROR
    "KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS requires KFCORE_BUILD_TENSORRT_YOLO=ON")
endif()

if(KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS)
  kfcore_validate_tensorrt_integration_engine(
    "${KFCORE_TENSORRT_TEST_ENGINE}"
    KFCORE_TENSORRT_TEST_ENGINE_VALIDATED)
  kfcore_validate_optional_tensorrt_integration_engine(
    "${KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE}"
    KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE_VALIDATED
    KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE)
endif()

if(KFCORE_BUILD_TENSORRT_RUNTIME_INTEGRATION_TESTS AND
   NOT KFCORE_BUILD_TENSORRT_RUNTIME)
  message(FATAL_ERROR
    "KFCORE_BUILD_TENSORRT_RUNTIME_INTEGRATION_TESTS requires "
    "KFCORE_BUILD_TENSORRT_RUNTIME=ON")
endif()

if(KFCORE_BUILD_TENSORRT_RUNTIME_INTEGRATION_TESTS)
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

include(CTest)

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
endif()

add_subdirectory(vendor)
add_subdirectory(trackers)

set(KFCORE_IMAGE_PROCESSOR_CUDA_ENABLED OFF)
if(KFCORE_BUILD_IMAGE_PROCESSOR OR KFCORE_BUILD_TENSORRT_YOLO OR KFCORE_BUILD_SIFT)
  set(KFCORE_IMAGE_PROCESSOR_CUDA_ENABLED ON)
endif()

set(KFCORE_IMAGE_CORE_ENABLED ${KFCORE_IMAGE_PROCESSOR_CUDA_ENABLED})
if(KFCORE_BUILD_FACE_APPLICATIONS_CPU)
  set(KFCORE_IMAGE_CORE_ENABLED ON)
endif()
if(KFCORE_IMAGE_CORE_ENABLED)
  enable_language(CXX)
endif()
if(KFCORE_IMAGE_PROCESSOR_CUDA_ENABLED)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
if(KFCORE_IMAGE_CORE_ENABLED)
  add_subdirectory(image_processor)
endif()

if(KFCORE_BUILD_SIFT_POPSIFT AND NOT KFCORE_BUILD_SIFT)
  message(FATAL_ERROR "KFCORE_BUILD_SIFT_POPSIFT requires KFCORE_BUILD_SIFT=ON")
endif()

if(KFCORE_BUILD_SIFT)
  add_subdirectory(sift)
endif()

if(KFCORE_BUILD_YOLO_TRACKING)
  enable_language(CXX)
  if(KFCORE_BUILD_TENSORRT_YOLO)
    find_package(TensorRT REQUIRED)
  endif()
  add_subdirectory(tensorrt_yolo)
endif()

if(KFCORE_BUILD_TENSORRT_RUNTIME)
  enable_language(CXX)
  add_subdirectory(tensorrt_runtime)
endif()

if(KFCORE_BUILD_FACE_MODELS AND NOT KFCORE_BUILD_TENSORRT_RUNTIME)
  message(FATAL_ERROR
    "KFCORE_BUILD_FACE_MODELS requires KFCORE_BUILD_TENSORRT_RUNTIME=ON")
endif()

if(KFCORE_BUILD_FACE_MODELS OR KFCORE_BUILD_FACE_APPLICATIONS_CPU)
  add_subdirectory(face_models)
endif()

if(KFCORE_BUILD_FACE_APPLICATIONS AND
   (NOT KFCORE_BUILD_FACE_MODELS OR NOT KFCORE_BUILD_TENSORRT_YOLO OR
    NOT KFCORE_BUILD_YOLO_OPENCV))
  message(FATAL_ERROR
    "KFCORE_BUILD_FACE_APPLICATIONS requires KFCORE_BUILD_FACE_MODELS, "
    "KFCORE_BUILD_TENSORRT_YOLO, and KFCORE_BUILD_YOLO_OPENCV")
endif()

if(KFCORE_BUILD_FACE_APPLICATION_INTEGRATION_TESTS AND
   NOT KFCORE_BUILD_FACE_APPLICATIONS)
  message(FATAL_ERROR
    "KFCORE_BUILD_FACE_APPLICATION_INTEGRATION_TESTS requires "
    "KFCORE_BUILD_FACE_APPLICATIONS=ON")
endif()

if(KFCORE_BUILD_FACE_APPLICATIONS)
  add_subdirectory(face_applications)
endif()

if(KFCORE_BUILD_FACE_APPLICATIONS_CPU)
  enable_language(CXX)
  find_package(ONNXRuntime REQUIRED MODULE)
  add_subdirectory(face_applications_cpu)
endif()

if(KFCORE_BUILD_APRILTAG)
  set(APRILTAG_BUILD_EXAMPLES "${KFCORE_BUILD_APRILTAG_EXAMPLES}" CACHE BOOL
      "Build AprilTag example executables" FORCE)
  set(APRILTAG_BUILD_PYTHON_WRAPPER "${KFCORE_BUILD_APRILTAG_PYTHON}" CACHE BOOL
      "Build AprilTag Python wrapper" FORCE)
  set(APRILTAG_BUILD_TESTS "${KFCORE_BUILD_APRILTAG_TESTS}" CACHE BOOL
      "Build AprilTag upstream tests" FORCE)
  set(APRILTAG_INSTALL OFF CACHE BOOL "Install AprilTag from the vendored subproject" FORCE)
  add_subdirectory(apriltag)
endif()
add_subdirectory(kalman) 

if(KFCORE_INSTALL)
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

  if(KFCORE_BUILD_NAVIGATION_TOOLS)
    install(FILES kalman/navigation_tools/frame_transform.h
                  kalman/navigation_tools/navtoolbox.h
                  kalman/navigation_tools/nav_fusion2d.h
                  kalman/navigation_tools/nav_fusion3d.h
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kfcore/navigation_tools")
  endif()

  if(KFCORE_BUILD_TENSORRT_RUNTIME OR KFCORE_BUILD_TENSORRT_YOLO)
    install(FILES cmake/FindTensorRT.cmake
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KFCore")
  endif()

  if(KFCORE_BUILD_YOLO_OPENCV OR KFCORE_BUILD_FACE_APPLICATIONS)
    install(FILES cmake/FindOpenCVLite.cmake
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KFCore")
  endif()

  if(KFCORE_BUILD_FACE_APPLICATIONS_CPU)
    install(FILES cmake/FindONNXRuntime.cmake
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KFCore")
  endif()

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
endif()
