cmake_add_test(
  NAME test_runtime_onnx_api
  SOURCES test_runtime_onnx_api.cpp
  LIBS KFCore::runtime_onnx Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "backends/onnx_cpu/runtime/tests")

add_executable(test_runtime_onnx_execution test_runtime_onnx_execution.cpp)
target_link_libraries(test_runtime_onnx_execution PRIVATE KFCore::runtime_onnx Salts::TinyTest)
target_compile_features(test_runtime_onnx_execution PRIVATE cxx_std_17)
target_compile_definitions(test_runtime_onnx_execution PRIVATE
  KFCORE_TEST_ONNX_CUDA=$<BOOL:${KFCORE_ONNXRUNTIME_ENABLE_CUDA}>
  KFCORE_TEST_ALLOW_CPU_NODES=$<BOOL:${KFCORE_ONNXRUNTIME_ALLOW_CPU_NODES}>
  "KFCORE_ONNX_FIXTURES=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures\"")
add_test(NAME test_runtime_onnx_execution COMMAND test_runtime_onnx_execution)
if(WIN32)
  set_tests_properties(test_runtime_onnx_api test_runtime_onnx_execution PROPERTIES
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$ENV{ONNXRUNTIME_ROOT}/lib")
  add_executable(test_runtime_onnx_load_failure test_runtime_onnx_load_failure.cpp)
  target_link_libraries(test_runtime_onnx_load_failure PRIVATE KFCore::runtime_onnx Salts::TinyTest)
  target_compile_features(test_runtime_onnx_load_failure PRIVATE cxx_std_17)
  add_test(NAME test_runtime_onnx_load_failure COMMAND test_runtime_onnx_load_failure)
  set_tests_properties(test_runtime_onnx_load_failure PROPERTIES
    ENVIRONMENT "ONNXRUNTIME_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
endif()
