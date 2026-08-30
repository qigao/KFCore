cmake_minimum_required(VERSION 3.20)

project(TensorRtRuntimeInstalledConsumer LANGUAGES CXX)

set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)

if(NOT DEFINED KFCORE_EXPECTED_KFCORE_DIR)
  message(FATAL_ERROR "Expected KFCore directory is required")
endif()

find_package(KFCore CONFIG REQUIRED
  PATHS "${KFCORE_EXPECTED_KFCORE_DIR}"
  NO_DEFAULT_PATH)

file(REAL_PATH "${KFCORE_EXPECTED_KFCORE_DIR}" _expected_kfcore_dir)
file(REAL_PATH "${KFCore_DIR}" _resolved_kfcore_dir)
if(NOT _resolved_kfcore_dir STREQUAL _expected_kfcore_dir)
  message(FATAL_ERROR "Consumer resolved KFCore outside its install prefix")
endif()
if(NOT TARGET KFCore::tensorrt_runtime OR NOT TARGET KFCore::face_models)
  message(FATAL_ERROR
    "Installed KFCore package does not export the TensorRT runtime and face model targets")
endif()

add_executable(tensorrt_runtime_installed_consumer main.cpp)
target_link_libraries(tensorrt_runtime_installed_consumer PRIVATE
  KFCore::tensorrt_runtime
  KFCore::face_models)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/consumer-target-file-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:tensorrt_runtime_installed_consumer>")
