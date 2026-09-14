add_executable(kfmodel main.cpp)
target_compile_features(kfmodel PRIVATE cxx_std_17)
target_link_libraries(kfmodel PRIVATE KFCore::runtime_core)
set_target_properties(kfmodel PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "tools/kfmodel")
cmake_config_target(kfmodel FOLDER "tools/kfmodel")
