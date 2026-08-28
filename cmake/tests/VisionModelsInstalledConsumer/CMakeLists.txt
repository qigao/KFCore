cmake_minimum_required(VERSION 3.20)

project(VisionModelsInstalledConsumer LANGUAGES CXX)

set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)

if(NOT DEFINED KFCORE_EXPECTED_KFCORE_DIR OR
   NOT DEFINED KFCORE_EXPECT_CPU OR NOT DEFINED KFCORE_EXPECT_TENSORRT OR
   NOT DEFINED KFCORE_EXPECT_HAND_INTERACTION)
  message(FATAL_ERROR "Expected KFCore directory and backend flags are required")
endif()
if(KFCORE_EXPECT_CPU)
  if(NOT DEFINED ONNXRUNTIME_ROOT OR NOT IS_DIRECTORY "${ONNXRUNTIME_ROOT}")
    message(FATAL_ERROR "CPU consumer requires ONNXRUNTIME_ROOT")
  endif()
  set(ENV{ONNXRUNTIME_ROOT} "${ONNXRUNTIME_ROOT}")
endif()
if(KFCORE_EXPECT_TENSORRT)
  if(NOT DEFINED TENSORRT_ROOT OR NOT IS_DIRECTORY "${TENSORRT_ROOT}" OR
     NOT DEFINED CUDAToolkit_ROOT OR NOT IS_DIRECTORY "${CUDAToolkit_ROOT}")
    message(FATAL_ERROR "TensorRT consumer requires TensorRT and CUDA roots")
  endif()
  set(ENV{TENSORRT_ROOT} "${TENSORRT_ROOT}")
endif()

find_package(KFCore CONFIG REQUIRED
  PATHS "${KFCORE_EXPECTED_KFCORE_DIR}"
  NO_DEFAULT_PATH)

file(REAL_PATH "${KFCORE_EXPECTED_KFCORE_DIR}" _expected_kfcore_dir)
file(REAL_PATH "${KFCore_DIR}" _resolved_kfcore_dir)
if(NOT _resolved_kfcore_dir STREQUAL _expected_kfcore_dir)
  message(FATAL_ERROR "Consumer resolved KFCore outside its install prefix")
endif()
if(NOT DEFINED KFCore_HAS_VISION_MODELS OR NOT KFCore_HAS_VISION_MODELS OR
   NOT TARGET KFCore::vision_model_core)
  message(FATAL_ERROR
    "Installed KFCore package does not report/export the vision-model core")
endif()
if(KFCORE_EXPECT_CPU AND
   (NOT DEFINED KFCore_HAS_VISION_MODELS_CPU OR
    NOT KFCore_HAS_VISION_MODELS_CPU OR
    NOT TARGET KFCore::vision_models_cpu))
  message(FATAL_ERROR
    "Installed KFCore package does not report/export the CPU vision backend")
endif()
if(KFCORE_EXPECT_TENSORRT AND
   (NOT DEFINED KFCore_HAS_VISION_MODELS_TENSORRT OR
    NOT KFCore_HAS_VISION_MODELS_TENSORRT OR
    NOT TARGET KFCore::vision_models_tensorrt))
  message(FATAL_ERROR
    "Installed KFCore package does not report/export the TensorRT vision backend")
endif()
if(KFCORE_EXPECT_HAND_INTERACTION AND
   (NOT DEFINED KFCore_HAS_THIG OR NOT KFCore_HAS_THIG OR
    NOT DEFINED KFCore_HAS_HAND_INTERACTION OR
    NOT KFCore_HAS_HAND_INTERACTION OR
    NOT TARGET KFCore::thig OR NOT TARGET KFCore::hand_interaction))
  message(FATAL_ERROR
    "Installed KFCore package does not report/export hand interaction")
endif()

add_executable(vision_models_installed_consumer main.cpp)
target_link_libraries(vision_models_installed_consumer PRIVATE
  KFCore::vision_model_core)
if(KFCORE_EXPECT_HAND_INTERACTION)
  target_link_libraries(vision_models_installed_consumer PRIVATE
    KFCore::hand_interaction)
  target_compile_definitions(vision_models_installed_consumer PRIVATE
    KFCORE_CONSUMER_HAS_HAND_INTERACTION=1)
endif()
if(KFCORE_EXPECT_CPU)
  target_link_libraries(vision_models_installed_consumer PRIVATE
    KFCore::vision_models_cpu)
  target_compile_definitions(vision_models_installed_consumer PRIVATE
    KFCORE_CONSUMER_HAS_CPU=1)
endif()
if(KFCORE_EXPECT_TENSORRT)
  target_link_libraries(vision_models_installed_consumer PRIVATE
    KFCore::vision_models_tensorrt)
  target_compile_definitions(vision_models_installed_consumer PRIVATE
    KFCORE_CONSUMER_HAS_TENSORRT=1)
endif()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/consumer-target-file-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:vision_models_installed_consumer>")
