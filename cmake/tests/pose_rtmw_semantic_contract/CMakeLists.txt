cmake_minimum_required(VERSION 3.20)

project(
  KFCorePoseRtmwSemanticContract
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(_kfcore_root "${CMAKE_CURRENT_LIST_DIR}/../../..")
file(REAL_PATH "${_kfcore_root}" _kfcore_root)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

list(APPEND CMAKE_MODULE_PATH "${_kfcore_root}/cmake")
include(GNUInstallDirs)
include("${_kfcore_root}/cmake/CmakeUtils.cmake")

set(BUILD_TESTS ON)
set(KFCORE_BUILD_POSE_RTMW_ORT_CPU_QUALIFICATION OFF)
set(KFCORE_ENABLE_ONNXRUNTIME OFF)
set(KFCORE_ENABLE_CUDA OFF)
set(KFCORE_ENABLE_TENSORRT OFF)
set(KFCORE_ENABLE_HF_TOKENIZER_PROVIDER OFF)

find_package(Threads REQUIRED)
find_package(Salts CONFIG REQUIRED HINTS "$ENV{SALTS_ROOT}")
find_package(SaltsUtils CONFIG REQUIRED HINTS "$ENV{SALTS_UTILS_ROOT}")

add_subdirectory(
  "${_kfcore_root}/runtime/abi"
  "${CMAKE_BINARY_DIR}/runtime-abi")
add_subdirectory(
  "${_kfcore_root}/runtime/core"
  "${CMAKE_BINARY_DIR}/runtime-core")
add_subdirectory(
  "${_kfcore_root}/vision/core/image_processor"
  "${CMAKE_BINARY_DIR}/image-processor-core")
add_subdirectory(
  "${_kfcore_root}/vision/image/cpu"
  "${CMAKE_BINARY_DIR}/image-processor-cpu")
add_subdirectory(
  "${_kfcore_root}/vision/core/pose"
  "${CMAKE_BINARY_DIR}/pose-core")
