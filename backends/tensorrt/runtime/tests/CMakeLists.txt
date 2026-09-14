set(_runtime_tensorrt_test_folder "backends/tensorrt/runtime/tests")
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

set(KFCORE_TENSORRT_TEST_ENGINE_ARCFACE "" CACHE FILEPATH
    "Optional TensorRT ArcFace engine used by the runtime integration test")
set(KFCORE_TENSORRT_TEST_ENGINE_PALM "" CACHE FILEPATH
    "Optional TensorRT palm-detection engine used by the dynamic-output integration test")

if(KFCORE_TENSORRT_TEST_ENGINE_ARCFACE)
  cmake_add_test(
    NAME test_runtime_tensorrt_integration
    TEST_NAME test_runtime_tensorrt_integration_arcface
    SOURCES test_runtime_tensorrt_integration.cpp
    LIBS KFCore::runtime_tensorrt CUDA::cudart Salts::TinyTest
    COMPILE_FEATURES cxx_std_17
    FOLDER "${_runtime_tensorrt_test_folder}"
    TEST_PROPERTIES
      ENVIRONMENT
        "KFCORE_RUNTIME_TENSORRT_TEST_ENGINE_ARCFACE=${KFCORE_TENSORRT_TEST_ENGINE_ARCFACE}"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

if(KFCORE_TENSORRT_TEST_ENGINE_PALM)
  cmake_add_test(
    NAME test_runtime_tensorrt_dynamic_output_integration
    TEST_NAME test_runtime_tensorrt_dynamic_output
    SOURCES test_runtime_tensorrt_dynamic_output_integration.cpp
    LIBS KFCore::runtime_tensorrt Salts::TinyTest
    COMPILE_FEATURES cxx_std_17
    FOLDER "${_runtime_tensorrt_test_folder}"
    TEST_PROPERTIES
      ENVIRONMENT
        "KFCORE_RUNTIME_TENSORRT_TEST_ENGINE_PALM=${KFCORE_TENSORRT_TEST_ENGINE_PALM}"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

if(WIN32)
  set(_runtime_tensorrt_windows_tests test_runtime_tensorrt_contract)
  if(KFCORE_TENSORRT_TEST_ENGINE_ARCFACE)
    list(APPEND _runtime_tensorrt_windows_tests test_runtime_tensorrt_integration_arcface)
  endif()
  if(KFCORE_TENSORRT_TEST_ENGINE_PALM)
    list(APPEND _runtime_tensorrt_windows_tests test_runtime_tensorrt_dynamic_output)
  endif()
  set_tests_properties(
    ${_runtime_tensorrt_windows_tests}
    PROPERTIES ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
endif()
