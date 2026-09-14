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

find_package(Salts CONFIG REQUIRED
             PATHS "$ENV{SALTS_ROOT}" NO_DEFAULT_PATH)  
find_package(SaltsUtils CONFIG REQUIRED PATHS "$ENV{SALTS_UTILS_ROOT}" NO_DEFAULT_PATH)

add_subdirectory(vendor)
add_subdirectory(trackers)

enable_language(CXX)
if(KFCORE_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(TensorRT REQUIRED MODULE)
endif()
if(KFCORE_ENABLE_ONNX_CPU OR KFCORE_ENABLE_ONNX_CUDA)
  find_package(ONNXRuntime REQUIRED MODULE)
endif()

add_subdirectory(vision/core/image_processor)
add_subdirectory(vision/image/cpu)
if(KFCORE_ENABLE_CUDA)
  add_subdirectory(vision/image/cuda)
endif()
add_subdirectory(thig)
add_subdirectory(sift)
add_subdirectory(vision/core/yolo)
add_subdirectory(vision/core/face_models)
add_subdirectory(vision/core/hand_models)
add_subdirectory(hand_interaction)

if(KFCORE_ENABLE_CUDA)
  add_subdirectory(backends/tensorrt_cuda/runtime)
  add_subdirectory(backends/tensorrt_cuda/yolo)
  add_subdirectory(backends/tensorrt_cuda/face_models)
  add_subdirectory(backends/tensorrt_cuda/hand_models)
  add_subdirectory(applications/tensorrt_cuda)
endif()

if(KFCORE_ENABLE_ONNX_CPU OR KFCORE_ENABLE_ONNX_CUDA)
  add_subdirectory(backends/onnx_cpu/runtime)
  add_subdirectory(backends/onnx_cpu/yolo)
  add_subdirectory(backends/onnx_cpu/face_models)
  add_subdirectory(backends/onnx_cpu/hand_models)
  add_subdirectory(applications/onnx_cpu)
endif()

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

  install(FILES cmake/FindOpenCVLite.cmake
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KFCore")
  if(KFCORE_ENABLE_CUDA)
    install(FILES cmake/FindTensorRT.cmake
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KFCore")
  endif()
  if(KFCORE_ENABLE_ONNX_CPU OR KFCORE_ENABLE_ONNX_CUDA)
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

install(FILES
        "${KFCORE_GENERATED_INCLUDE_DIR}/kfcore/model_configuration.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kfcore")
