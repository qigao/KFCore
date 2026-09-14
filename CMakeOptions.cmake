set(CMAKE_COLOR_DIAGNOSTICS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_TESTS "Build test cases" OFF)

option(KFCORE_ENABLE_CUDA "Build CUDA image/compute support" ON)
option(KFCORE_ENABLE_TENSORRT "Build TensorRT execution backend" ON)
option(KFCORE_ENABLE_ONNXRUNTIME "Build ONNX Runtime execution backend" OFF)
option(KFCORE_ONNXRUNTIME_ENABLE_CUDA
       "Expose CUDA devices from the ONNX Runtime backend when the installed ORT SDK provides CUDA EP"
       OFF)

if(KFCORE_ENABLE_TENSORRT AND NOT KFCORE_ENABLE_CUDA)
  message(FATAL_ERROR "TensorRT requires KFCORE_ENABLE_CUDA=ON")
endif()
if(KFCORE_ONNXRUNTIME_ENABLE_CUDA AND NOT KFCORE_ENABLE_ONNXRUNTIME)
  message(FATAL_ERROR
    "KFCORE_ONNXRUNTIME_ENABLE_CUDA requires KFCORE_ENABLE_ONNXRUNTIME=ON")
endif()

set(KFCORE_MODEL_TEST_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" CACHE PATH
    "Working directory containing yolo-models for real-model integration tests")

option(KFCORE_ONNXRUNTIME_ALLOW_CPU_NODES
       "Allow ORT CPU node placement inside an explicitly selected CUDA session"
       OFF)
