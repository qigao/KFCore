set(_runtime_tensorrt_test_folder "runtime_tensorrt/tests")
set(_runtime_tensorrt_test_includes ../src ../include)

cmake_add_test(
  NAME test_runtime_tensorrt_contract
  SOURCES test_tensor_validation.cpp
  LIBS KFCore::runtime_tensorrt Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_runtime_tensorrt_test_folder}")

cmake_add_test(
  NAME test_runtime_tensorrt_engine_file
  SOURCES test_engine_file.cpp ../src/engine_file.cpp ../src/error.cpp
  LIBS Salts::TinyTest
  INCLUDES ${_runtime_tensorrt_test_includes}
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_runtime_tensorrt_test_folder}")

cmake_add_test(
  NAME test_runtime_tensorrt_cuda_buffer
  SOURCES test_cuda_buffer.cpp ../src/cuda_buffer.cpp ../src/error.cpp
  LIBS CUDA::cudart Salts::TinyTest
  DEFS KFCORE_RUNTIME_TENSORRT_CUDA_BUFFER_TESTING=1
  INCLUDES ${_runtime_tensorrt_test_includes}
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_runtime_tensorrt_test_folder}")

cmake_add_test(
  NAME test_runtime_tensorrt_control
  SOURCES
    test_runtime_control.cpp
    ../src/cuda_buffer.cpp
    ../src/cuda_device.cpp
    ../src/error.cpp
  LIBS CUDA::cudart Salts::TinyTest
  INCLUDES ${_runtime_tensorrt_test_includes}
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_runtime_tensorrt_test_folder}")

set(_runtime_tensorrt_engine_root
  "tensorrt/${KFCORE_TENSORRT_ENGINE_PROFILE}")
set(_runtime_tensorrt_arcface_engine
  "./yolo-models/${_runtime_tensorrt_engine_root}/arcface_w600k_r50.engine")
set(_runtime_tensorrt_palm_engine
  "./yolo-models/hand_gesture_model/tensorrt/${KFCORE_TENSORRT_ENGINE_PROFILE}/palm_detection.engine")

cmake_add_test(
  NAME test_runtime_tensorrt_integration
  TEST_NAME test_runtime_tensorrt_integration_arcface
  SOURCES test_runtime_tensorrt_integration.cpp
  LIBS KFCore::runtime_tensorrt CUDA::cudart Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_runtime_tensorrt_test_folder}"
  TEST_PROPERTIES
    ENVIRONMENT
      "KFCORE_RUNTIME_TENSORRT_TEST_ENGINE_ARCFACE=${_runtime_tensorrt_arcface_engine}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

cmake_add_test(
  NAME test_runtime_tensorrt_dynamic_output_integration
  TEST_NAME test_runtime_tensorrt_dynamic_output
  SOURCES test_runtime_tensorrt_dynamic_output_integration.cpp
  LIBS KFCore::runtime_tensorrt Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_runtime_tensorrt_test_folder}"
  TEST_PROPERTIES
    ENVIRONMENT
      "KFCORE_RUNTIME_TENSORRT_TEST_ENGINE_PALM=${_runtime_tensorrt_palm_engine}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

if(WIN32)
  set_tests_properties(
    test_runtime_tensorrt_contract
    test_runtime_tensorrt_integration_arcface
    test_runtime_tensorrt_dynamic_output
    PROPERTIES ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
endif()
