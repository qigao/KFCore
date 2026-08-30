set(_yolo_tensorrt_test_folder "yolo_tensorrt/tests")

cmake_add_test(
  NAME test_yolo_tensorrt_contract
  SOURCES test_tensorrt_contract.cpp ../src/engine_contract.cpp
  LIBS KFCore::yolo_core TurboUtils::TinyTest
  INCLUDES ../src ../../yolo_core/src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_tensorrt_test_folder}")
cmake_add_test(
  NAME test_yolo_tensorrt_detection_helpers
  SOURCES test_tensorrt_detection_helpers.cpp ../src/detector_helpers.cpp
  LIBS
    KFCore::yolo_core
    KFCore::image_processor_cuda
    TurboUtils::TinyTest
  INCLUDES ../src ../../yolo_core/src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_tensorrt_test_folder}")
cmake_add_test(
  NAME test_yolo_tensorrt_cuda_buffer
  SOURCES test_cuda_buffer.cpp ../src/cuda_buffer.cpp
  LIBS KFCore::yolo_core CUDA::cudart TurboUtils::TinyTest
  DEFS KFCORE_YOLO_CUDA_BUFFER_TESTING
  INCLUDES ../src ../../yolo_core/src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_tensorrt_test_folder}")
cmake_add_test(
  NAME test_yolo_tensorrt_engine_file
  SOURCES test_engine_file.cpp ../src/engine_file.cpp
  LIBS KFCore::yolo_core TurboUtils::TinyTest
  INCLUDES ../src ../../yolo_core/src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_tensorrt_test_folder}")

kfcore_resolve_model_path(
  "$ENV{KFCORE_MODEL_ROOT}"
  "tensorrt/$ENV{KFCORE_TENSORRT_ENGINE_PROFILE}/yolov11n-face.engine"
  _yolo_test_engine)
cmake_add_test(
  NAME test_yolo_tensorrt_integration
  SOURCES test_tensorrt_integration.cpp
  LIBS
    KFCore::yolo_tensorrt
    TensorRT::nvinfer
    TensorRT::nvinfer_plugin
    CUDA::cudart
    TurboUtils::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_tensorrt_test_folder}"
  TEST_PROPERTIES ENVIRONMENT "KFCORE_TENSORRT_TEST_ENGINE=${_yolo_test_engine}")

if(WIN32)
  set_tests_properties(test_yolo_tensorrt_integration PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
endif()
