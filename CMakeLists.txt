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
include(CmakeUtils)

set(KFCORE_TENSORRT_ENGINE_PROFILE
    "rtx4060-sm89-trt11.2.1-default"
    CACHE STRING "TensorRT engine profile directory under yolo-models/tensorrt")
if(NOT KFCORE_TENSORRT_ENGINE_PROFILE MATCHES "^[A-Za-z0-9._-]+$")
  message(FATAL_ERROR
    "KFCORE_TENSORRT_ENGINE_PROFILE must be one path segment")
endif()

set(KFCORE_GENERATED_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
configure_file(
  cmake/model_configuration.hpp.in
  "${KFCORE_GENERATED_INCLUDE_DIR}/kfcore/model_configuration.hpp"
  @ONLY)

if(BUILD_TESTS)
  enable_testing()
endif()

find_package(Threads REQUIRED)

find_package(TurboUtils CONFIG REQUIRED
             PATHS "$ENV{TURBOUTILS_ROOT}" NO_DEFAULT_PATH)
find_package(TurboParser CONFIG REQUIRED
             PATHS "$ENV{TURBOPARSER_ROOT}" NO_DEFAULT_PATH)
find_package(TurboNet CONFIG REQUIRED
             PATHS "$ENV{TURBONET_ROOT}" NO_DEFAULT_PATH)

add_subdirectory(vendor)
add_subdirectory(trackers)

enable_language(CXX)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED MODULE)
find_package(ONNXRuntime REQUIRED MODULE)

add_subdirectory(image_processor_core)
add_subdirectory(image_processor_cpu)
add_subdirectory(image_processor_cuda)
add_subdirectory(thig)
add_subdirectory(sift)
add_subdirectory(yolo_core)
add_subdirectory(yolo_onnx)
add_subdirectory(yolo_tensorrt)
add_subdirectory(runtime_tensorrt)
add_subdirectory(runtime_onnx)
add_subdirectory(face_model_core)
add_subdirectory(face_models_cpu)
add_subdirectory(face_models_cuda)
add_subdirectory(hand_model_core)
add_subdirectory(hand_models_cpu)
add_subdirectory(hand_models_cuda)
add_subdirectory(hand_interaction)
add_subdirectory(face_applications_cuda)
add_subdirectory(face_applications_cpu)

add_subdirectory(apriltag)
add_subdirectory(kalman) 

include(CMakePackageConfigHelpers)

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

install(FILES
        "${KFCORE_GENERATED_INCLUDE_DIR}/kfcore/model_configuration.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kfcore")
