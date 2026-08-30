find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)

add_library(kfcore_tensorrt_runtime SHARED
  src/cuda_buffer.cpp
  src/cuda_device.cpp
  src/engine.cpp
  src/engine_file.cpp
  src/error.cpp
  src/executor.cpp
  src/tensor_validation.cpp)
target_include_directories(kfcore_tensorrt_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_tensorrt_runtime PUBLIC cxx_std_17)
target_link_libraries(kfcore_tensorrt_runtime PRIVATE
  CUDA::cudart
  TensorRT::nvinfer
  TensorRT::nvinfer_plugin)
set_target_properties(kfcore_tensorrt_runtime PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "tensorrt_runtime"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_tensorrt_runtime
  ALIAS KFCore::tensorrt_runtime
  EXPORT_NAME tensorrt_runtime
  EXPORT_SET KFCoreTargets
  FOLDER "tensorrt_runtime")

if(BUILD_TESTS)
  add_executable(test_tensorrt_runtime_contract tests/test_tensor_validation.cpp)
  target_include_directories(test_tensorrt_runtime_contract PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_tensorrt_runtime_contract PRIVATE
    KFCore::tensorrt_runtime
    TurboUtils::TinyTest)
  set_target_properties(test_tensorrt_runtime_contract PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_runtime/tests")
  add_test(NAME test_tensorrt_runtime_contract COMMAND test_tensorrt_runtime_contract)
  if(WIN32)
    set_tests_properties(test_tensorrt_runtime_contract PROPERTIES
      ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin")
  endif()

  add_executable(test_tensorrt_runtime_engine_file
    tests/test_engine_file.cpp
    src/engine_file.cpp
    src/error.cpp)
  target_include_directories(test_tensorrt_runtime_engine_file PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_tensorrt_runtime_engine_file PRIVATE TurboUtils::TinyTest)
  set_target_properties(test_tensorrt_runtime_engine_file PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_runtime/tests")
  add_test(NAME test_tensorrt_runtime_engine_file COMMAND test_tensorrt_runtime_engine_file)

  add_executable(test_tensorrt_runtime_cuda_buffer
    tests/test_cuda_buffer.cpp
    src/cuda_buffer.cpp
    src/error.cpp)
  target_include_directories(test_tensorrt_runtime_cuda_buffer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(test_tensorrt_runtime_cuda_buffer PRIVATE
    KFCORE_TENSORRT_RUNTIME_CUDA_BUFFER_TESTING=1)
  target_link_libraries(test_tensorrt_runtime_cuda_buffer PRIVATE
    CUDA::cudart
    TurboUtils::TinyTest)
  set_target_properties(test_tensorrt_runtime_cuda_buffer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_runtime/tests")
  add_test(NAME test_tensorrt_runtime_cuda_buffer COMMAND test_tensorrt_runtime_cuda_buffer)

  add_executable(test_tensorrt_runtime_control
    tests/test_runtime_control.cpp
    src/cuda_buffer.cpp
    src/cuda_device.cpp
    src/error.cpp)
  target_include_directories(test_tensorrt_runtime_control PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_tensorrt_runtime_control PRIVATE
    CUDA::cudart
    TurboUtils::TinyTest)
  set_target_properties(test_tensorrt_runtime_control PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_runtime/tests")
  add_test(NAME test_tensorrt_runtime_control COMMAND test_tensorrt_runtime_control)

  if(KFCORE_ENABLE_TENSORRT_RUNTIME_INTEGRATION_TESTS AND
     NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE_VALIDATED}" STREQUAL "")
    add_executable(test_tensorrt_runtime_integration
      tests/test_tensorrt_runtime_integration.cpp)
    target_link_libraries(test_tensorrt_runtime_integration PRIVATE
      KFCore::tensorrt_runtime
      CUDA::cudart
      TurboUtils::TinyTest)
    set_target_properties(test_tensorrt_runtime_integration PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "tensorrt_runtime/tests")
    add_test(NAME test_tensorrt_runtime_integration_arcface
      COMMAND test_tensorrt_runtime_integration)
    set_tests_properties(test_tensorrt_runtime_integration_arcface PROPERTIES
      ENVIRONMENT
        "KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE=${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE_VALIDATED}")
    if(WIN32)
      set_tests_properties(test_tensorrt_runtime_integration_arcface PROPERTIES
        ENVIRONMENT_MODIFICATION
          "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
    endif()
  endif()

  if(KFCORE_ENABLE_TENSORRT_RUNTIME_INTEGRATION_TESTS AND
     NOT "${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_PALM}" STREQUAL "")
    add_executable(test_tensorrt_runtime_dynamic_output_integration
      tests/test_tensorrt_runtime_dynamic_output_integration.cpp)
    target_link_libraries(test_tensorrt_runtime_dynamic_output_integration PRIVATE
      KFCore::tensorrt_runtime
      TurboUtils::TinyTest)
    set_target_properties(test_tensorrt_runtime_dynamic_output_integration PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "tensorrt_runtime/tests")
    add_test(NAME test_tensorrt_runtime_dynamic_output
      COMMAND test_tensorrt_runtime_dynamic_output_integration)
    set_tests_properties(test_tensorrt_runtime_dynamic_output PROPERTIES
      ENVIRONMENT
        "KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_PALM=${KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_PALM}")
    if(WIN32)
      set_tests_properties(test_tensorrt_runtime_dynamic_output PROPERTIES
        ENVIRONMENT_MODIFICATION
          "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
    endif()
  endif()
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
