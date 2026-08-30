cmake_add_test(
  NAME test_runtime_onnx_api
  SOURCES test_runtime_onnx_api.cpp
  LIBS KFCore::runtime_onnx TurboUtils::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "runtime_onnx/tests")
