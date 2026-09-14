set(CMAKE_COLOR_DIAGNOSTICS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_TESTS "Build test cases" OFF)

option(KFCORE_ENABLE_CUDA "Build CUDA image/compute support" ON)
option(KFCORE_ENABLE_TENSORRT "Build TensorRT execution backend" ON)
option(KFCORE_ENABLE_ONNX_CPU "Build ONNX Runtime CPU execution backend" OFF)
option(KFCORE_ENABLE_ONNX_CUDA "Build ONNX Runtime CUDA execution backend" OFF)

if(KFCORE_ENABLE_TENSORRT AND NOT KFCORE_ENABLE_CUDA)
  message(FATAL_ERROR "TensorRT requires KFCORE_ENABLE_CUDA=ON")
endif()
if(KFCORE_ENABLE_ONNX_CUDA AND KFCORE_ENABLE_ONNX_CPU)
  message(FATAL_ERROR "Select one ONNX provider personality per plugin build: ONNX_CPU or ONNX_CUDA")
endif()

set(KFCORE_MODEL_TEST_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" CACHE PATH
    "Working directory containing yolo-models for real-model integration tests")

option(KFCORE_ONNX_CUDA_ALLOW_CPU_NODES "Allow ORT CPU node placement alongside the required CUDA provider" OFF)
