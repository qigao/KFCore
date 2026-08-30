cmake_minimum_required(VERSION 3.20)

project(VisionModelsInstalledConsumer LANGUAGES CXX)

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
if(NOT TARGET KFCore::vision_model_core)
  message(FATAL_ERROR
    "Installed KFCore package does not export the vision-model core")
endif()
if(NOT TARGET KFCore::vision_models_cpu)
  message(FATAL_ERROR
    "Installed KFCore package does not export the CPU vision backend")
endif()
if(NOT TARGET KFCore::vision_models_tensorrt OR
   NOT TARGET KFCore::tensorrt_yolo)
  message(FATAL_ERROR
    "Installed KFCore package does not export the TensorRT vision backend")
endif()
if(NOT TARGET KFCore::thig OR NOT TARGET KFCore::hand_interaction)
  message(FATAL_ERROR
    "Installed KFCore package does not export hand interaction")
endif()

add_executable(vision_models_installed_consumer main.cpp)
target_link_libraries(vision_models_installed_consumer PRIVATE
  KFCore::vision_model_core
  KFCore::hand_interaction
  KFCore::vision_models_cpu
  KFCore::vision_models_tensorrt)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/consumer-target-file-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:vision_models_installed_consumer>")
