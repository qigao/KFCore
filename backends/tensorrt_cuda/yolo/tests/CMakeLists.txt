set(_yolo_tensorrt_test_folder "backends/tensorrt_cuda/yolo/tests")

cmake_add_test(
  NAME test_yolo_tensorrt_contract
  SOURCES test_tensorrt_contract.cpp ../src/engine_contract.cpp
  LIBS KFCore::yolo_core Salts::TinyTest
  INCLUDES ../include ../src ${CMAKE_SOURCE_DIR}/vision/core/yolo/src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_tensorrt_test_folder}")
cmake_add_test(
  NAME test_yolo_tensorrt_detection_helpers
  SOURCES test_tensorrt_detection_helpers.cpp ../src/detector_helpers.cpp
  LIBS
    KFCore::yolo_core
    KFCore::image_processor_cuda
    Salts::TinyTest
  INCLUDES ../src ${CMAKE_SOURCE_DIR}/vision/core/yolo/src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_tensorrt_test_folder}")
cmake_add_test(
  NAME test_yolo_tensorrt_cuda_buffer
  SOURCES test_cuda_buffer.cpp ../src/cuda_buffer.cpp
  LIBS KFCore::yolo_core CUDA::cudart Salts::TinyTest
  DEFS KFCORE_YOLO_CUDA_BUFFER_TESTING
  INCLUDES ../src ${CMAKE_SOURCE_DIR}/vision/core/yolo/src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_tensorrt_test_folder}")
cmake_add_test(
  NAME test_yolo_tensorrt_engine_file
  SOURCES test_engine_file.cpp ../src/engine_file.cpp
  LIBS KFCore::yolo_core Salts::TinyTest
  INCLUDES ../src ${CMAKE_SOURCE_DIR}/vision/core/yolo/src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_tensorrt_test_folder}")

set(_yolo_test_engine
  "./yolo-models/tensorrt/${KFCORE_TENSORRT_ENGINE_PROFILE}/yolov11n-face.engine")
cmake_add_test(
  NAME test_yolo_tensorrt_integration
  SOURCES test_tensorrt_integration.cpp
  LIBS
    KFCore::yolo_tensorrt
    TensorRT::nvinfer
    TensorRT::nvinfer_plugin
    CUDA::cudart
    Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_tensorrt_test_folder}"
  TEST_PROPERTIES
    ENVIRONMENT "KFCORE_TENSORRT_TEST_ENGINE=${_yolo_test_engine}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

if(WIN32)
  set_tests_properties(test_yolo_tensorrt_integration PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
endif()

set(_yolo_test_person_engine
  "./yolo-models/tensorrt/${KFCORE_TENSORRT_ENGINE_PROFILE}/yolov8s.engine")
cmake_add_test(
  NAME test_yolov8s_person_tensorrt_integration
  SOURCES test_yolov8s_person_integration.cpp
  LIBS
    KFCore::yolo_tensorrt
    TensorRT::nvinfer
    TensorRT::nvinfer_plugin
    CUDA::cudart
    Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_yolo_tensorrt_test_folder}"
  TEST_PROPERTIES
    ENVIRONMENT "KFCORE_TENSORRT_TEST_PERSON_ENGINE=${_yolo_test_person_engine}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

if(WIN32)
  set_tests_properties(test_yolov8s_person_tensorrt_integration PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
endif()
