cmake_add_test(
  NAME test_runtime_onnx_api
  SOURCES test_runtime_onnx_api.cpp
  LIBS KFCore::runtime_onnx Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "backends/onnx_cpu/runtime/tests")
