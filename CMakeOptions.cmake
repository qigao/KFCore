set(CMAKE_COLOR_DIAGNOSTICS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_TESTS "Build test cases" OFF)
option(KFCORE_ENABLE_CUDA "Build CUDA and TensorRT backends" ON)
option(KFCORE_ENABLE_ONNX_CPU "Build ONNX Runtime CPU backends" OFF)
option(KFCORE_ENABLE_ONNX_CUDA "Build ONNX Runtime CUDA backends" OFF)
if(KFCORE_ENABLE_ONNX_CUDA AND KFCORE_ENABLE_ONNX_CPU)
  message(FATAL_ERROR "Select one ONNX provider per build: ONNX_CPU or ONNX_CUDA")
endif()
if(KFCORE_ENABLE_ONNX_CUDA AND KFCORE_ENABLE_CUDA)
  message(FATAL_ERROR "ONNX_CUDA builds require KFCORE_ENABLE_CUDA=OFF to exclude TensorRT")
endif()

set(KFCORE_MODEL_TEST_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" CACHE PATH
    "Working directory containing yolo-models for real-model integration tests")

option(KFCORE_ONNX_CUDA_ALLOW_CPU_NODES "Allow ORT CPU node placement alongside the required CUDA provider" OFF)
