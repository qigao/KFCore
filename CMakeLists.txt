cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

project(
  KFCore
  VERSION 0.1.0
  DESCRIPTION "KFCore model, vision, inference, and estimation runtime"
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

if(BUILD_TESTS)
  enable_testing()
endif()

find_package(Threads REQUIRED)
find_package(Salts CONFIG REQUIRED
             HINTS "$ENV{SALTS_ROOT}")
find_package(SaltsUtils CONFIG REQUIRED
             HINTS "$ENV{SALTS_UTILS_ROOT}")

add_subdirectory(vendor)
add_subdirectory(esn)
add_subdirectory(trackers)

enable_language(CXX)
add_subdirectory(runtime/abi)
add_subdirectory(runtime/core)
add_subdirectory(tools/kfmodel)

if(KFCORE_ENABLE_CUDA)
  enable_language(CUDA)
endif()
if(KFCORE_ENABLE_CUDA OR KFCORE_ONNXRUNTIME_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
if(KFCORE_ENABLE_TENSORRT)
  find_package(TensorRT REQUIRED MODULE)
endif()
if(KFCORE_ENABLE_ONNXRUNTIME)
  find_package(ONNXRuntime REQUIRED MODULE)
endif()

add_subdirectory(vision/core/image_processor)
add_subdirectory(vision/image/cpu)
if(KFCORE_ENABLE_CUDA)
  add_subdirectory(vision/image/cuda)
endif()
add_subdirectory(sift)
add_subdirectory(vision/core/yolo)
add_subdirectory(vision/core/pose)
add_subdirectory(vision/core/relation)
add_subdirectory(vision/pipelines/whole_body)
add_subdirectory(vision/pipelines/scene_graph)
add_subdirectory(vision/core/face_models)
add_subdirectory(applications/face_swap)
add_subdirectory(vision/core/hand_models)
add_subdirectory(vision/mediapipe)
add_subdirectory(gesture_interaction)
add_subdirectory(scene_interaction)
if(KFCORE_BUILD_HAND_PREVIEW)
  add_subdirectory(tools/hand_preview)
endif()
if(KFCORE_ENABLE_TEMPORAL_GESTURE_GRU)
  add_subdirectory(vision/core/hand_gesture)
  add_subdirectory(tools/kfgesture)
endif()

if(KFCORE_ENABLE_TENSORRT)
  add_subdirectory(backends/tensorrt/runtime)
  add_subdirectory(backends/tensorrt/plugin)
endif()

if(KFCORE_ENABLE_ONNXRUNTIME)
  add_subdirectory(backends/onnxruntime/plugin)
endif()

add_subdirectory(apriltag)
add_subdirectory(kalman)

include(CMakePackageConfigHelpers)

install(FILES kalman/kalman_status.h kalman/kalman_takasu.h kalman/kalman_udu.h kalman/kalman_ekf.h
              kalman/kalman_ukf.h kalman/signal_filters.h
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kfcore")

install(FILES kalman/navigation_tools/frame_transform.h
              kalman/navigation_tools/navtoolbox.h
              kalman/navigation_tools/nav_fusion2d.h
              kalman/navigation_tools/nav_fusion3d.h
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kfcore/navigation_tools")

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
