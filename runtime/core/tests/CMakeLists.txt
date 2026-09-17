add_executable(test_runtime_model_package test_model_package.cpp)
target_compile_features(test_runtime_model_package PRIVATE cxx_std_17)
target_link_libraries(test_runtime_model_package PRIVATE
  KFCore::runtime_core
  Salts::TinyTest)
set_target_properties(test_runtime_model_package PROPERTIES
  FOLDER "runtime/core/tests")
add_test(NAME test_runtime_model_package COMMAND test_runtime_model_package)
