cmake_minimum_required(VERSION 3.20)

project(
  KFCoreRelationGpuQualification
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

set(BUILD_TESTS OFF)
set(KFCORE_BUILD_RELATION_ORT_CPU_QUALIFICATION ON)
set(KFCORE_ENABLE_ONNXRUNTIME ON)
set(KFCORE_ONNXRUNTIME_ENABLE_CUDA ON)
set(KFCORE_ONNXRUNTIME_ALLOW_CPU_NODES OFF)
set(KFCORE_ENABLE_CUDA ON)
set(KFCORE_ENABLE_TENSORRT ON)
set(KFCORE_ENABLE_HF_TOKENIZER_PROVIDER OFF)
set(KFCORE_INSTALL_PLUGINDIR "plugins")

find_package(Threads REQUIRED)
find_package(Salts CONFIG REQUIRED HINTS "$ENV{SALTS_ROOT}")
find_package(SaltsUtils CONFIG REQUIRED HINTS "$ENV{SALTS_UTILS_ROOT}")
find_package(CUDAToolkit REQUIRED)
find_package(ONNXRuntime REQUIRED MODULE)
find_package(TensorRT REQUIRED MODULE)

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
  "${_kfcore_root}/vision/core/relation"
  "${CMAKE_BINARY_DIR}/relation-core")
add_subdirectory(
  "${_kfcore_root}/backends/onnxruntime/plugin"
  "${CMAKE_BINARY_DIR}/onnxruntime-plugin")
add_subdirectory(
  "${_kfcore_root}/backends/tensorrt/runtime"
  "${CMAKE_BINARY_DIR}/tensorrt-runtime")
add_subdirectory(
  "${_kfcore_root}/backends/tensorrt/plugin"
  "${CMAKE_BINARY_DIR}/tensorrt-plugin")
