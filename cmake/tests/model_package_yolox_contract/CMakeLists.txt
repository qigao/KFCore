cmake_minimum_required(VERSION 3.20)
project(KFCoreModelPackageYoloXContract LANGUAGES C CXX)

if(NOT DEFINED KFCORE_SOURCE_DIR OR KFCORE_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "KFCORE_SOURCE_DIR is required")
endif()

find_package(Salts CONFIG REQUIRED)
find_package(SaltsUtils CONFIG REQUIRED)

add_executable(kfcore_model_package_yolox_contract
  "${KFCORE_SOURCE_DIR}/runtime/core/src/model_package.cpp"
  "${KFCORE_SOURCE_DIR}/runtime/core/tests/test_model_package.cpp")

target_compile_features(
  kfcore_model_package_yolox_contract PRIVATE cxx_std_17)

target_include_directories(
  kfcore_model_package_yolox_contract PRIVATE
  "${KFCORE_SOURCE_DIR}/runtime/core/include")

target_link_libraries(
  kfcore_model_package_yolox_contract PRIVATE
  Salts::DataBind
  Salts::Crypto
  Salts::TinyTest)
