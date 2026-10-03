set(CMAKE_COLOR_DIAGNOSTICS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_TESTS "Build test cases" OFF)
option(KFCORE_BUILD_HAND_PREVIEW "Build Windows OpenCV MediaPipe preview" OFF)

set(KFCORE_INSTALL_PLUGINDIR "plugins" CACHE STRING
    "Install directory, relative to CMAKE_INSTALL_PREFIX, for KFCore execution plugins")
if(KFCORE_INSTALL_PLUGINDIR STREQUAL "" OR
   IS_ABSOLUTE "${KFCORE_INSTALL_PLUGINDIR}")
  message(FATAL_ERROR
    "KFCORE_INSTALL_PLUGINDIR must be a non-empty path relative to CMAKE_INSTALL_PREFIX")
endif()
string(REPLACE "\\" "/" _kfcore_install_plugindir_normalized
       "${KFCORE_INSTALL_PLUGINDIR}")
if(_kfcore_install_plugindir_normalized MATCHES "(^|/)\\.\\.(/|$)")
  message(FATAL_ERROR
    "KFCORE_INSTALL_PLUGINDIR must stay within CMAKE_INSTALL_PREFIX")
endif()
unset(_kfcore_install_plugindir_normalized)

option(KFCORE_ENABLE_CUDA "Build CUDA image/compute support" OFF)
option(KFCORE_ENABLE_TENSORRT "Build TensorRT execution backend" OFF)
option(KFCORE_ENABLE_ONNXRUNTIME "Build ONNX Runtime execution backend" OFF)
option(KFCORE_ENABLE_TEMPORAL_GESTURE_GRU
       "Build the experimental temporal gesture GRU library and smoke tool"
       OFF)
option(KFCORE_ENABLE_HF_TOKENIZER_PROVIDER
       "Build the optional HuggingFace tokenizer.json provider for relation predicates"
       OFF)
option(KFCORE_BUILD_RELATION_ORT_CPU_QUALIFICATION
       "Build the focused C++ ORT CPU open-vocabulary qualification executable"
       OFF)
option(KFCORE_BUILD_POSE_RTMW_ORT_CPU_QUALIFICATION
       "Build the focused C++ RTMW ORT CPU golden qualification executable"
       OFF)
option(KFCORE_BUILD_RELEASED_SCENE_BEHAVIOR_QUALIFICATION
       "Build the released detector/relation SceneBehavior latency qualification"
       OFF)
set(KFCORE_TOKENIZERS_CPP_SOURCE_DIR "" CACHE PATH
    "Path to a pinned mlc-ai/tokenizers-cpp source checkout when KFCORE_ENABLE_HF_TOKENIZER_PROVIDER=ON")
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
    "Working directory containing model assets for real-model integration tests")

option(KFCORE_ONNXRUNTIME_ALLOW_CPU_NODES
       "Allow ORT CPU node placement inside an explicitly selected CUDA session"
       OFF)
