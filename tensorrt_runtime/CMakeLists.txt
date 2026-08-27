add_library(kfcore_tensorrt_runtime SHARED
  src/error.cpp
  src/tensor_validation.cpp)
target_include_directories(kfcore_tensorrt_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_tensorrt_runtime PUBLIC cxx_std_17)
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
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
