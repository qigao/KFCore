cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED KFCORE_SOURCE_DIR OR KFCORE_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "KFCORE_SOURCE_DIR is required")
endif()
if(NOT DEFINED KFCORE_CONTRACT_ROOT OR KFCORE_CONTRACT_ROOT STREQUAL "")
  message(FATAL_ERROR "KFCORE_CONTRACT_ROOT is required")
endif()

file(REMOVE_RECURSE "${KFCORE_CONTRACT_ROOT}")
file(MAKE_DIRECTORY "${KFCORE_CONTRACT_ROOT}")

# The repository defaults must describe the backend-neutral SDK. Backend
# capability presets opt into CUDA / TensorRT / ONNX Runtime explicitly.
set(PROJECT_SOURCE_DIR "${KFCORE_SOURCE_DIR}")
include("${KFCORE_SOURCE_DIR}/CMakeOptions.cmake")
if(KFCORE_ENABLE_CUDA)
  message(FATAL_ERROR "KFCORE_ENABLE_CUDA must default OFF")
endif()
if(KFCORE_ENABLE_TENSORRT)
  message(FATAL_ERROR "KFCORE_ENABLE_TENSORRT must default OFF")
endif()
if(KFCORE_ENABLE_ONNXRUNTIME)
  message(FATAL_ERROR "KFCORE_ENABLE_ONNXRUNTIME must default OFF")
endif()

# Generate the install-tree package config exactly from the shipped template.
include(CMakePackageConfigHelpers)
set(_prefix "${KFCORE_CONTRACT_ROOT}/prefix")
set(_kfcore_dir "${_prefix}/lib/cmake/KFCore")
file(MAKE_DIRECTORY "${_kfcore_dir}")
configure_package_config_file(
  "${KFCORE_SOURCE_DIR}/cmake/KFCoreConfig.cmake.in"
  "${_kfcore_dir}/KFCoreConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/KFCore")
file(WRITE "${_kfcore_dir}/KFCoreTargets.cmake" "# focused package-contract stub\n")

# Dependencies are discoverable only through the standard package prefix. The
# legacy KFCore-specific environment roots are deliberately absent.
foreach(_dep IN ITEMS Salts SaltsUtils)
  file(MAKE_DIRECTORY "${_prefix}/lib/cmake/${_dep}")
  file(WRITE "${_prefix}/lib/cmake/${_dep}/${_dep}Config.cmake"
       "set(${_dep}_FOUND TRUE)\n")
endforeach()

set(ENV{SALTS_ROOT} "")
set(ENV{SALTS_UTILS_ROOT} "")

# Consume the generated config from a real CMake project so FindThreads runs
# with C/CXX languages enabled, exactly as it does for normal SDK consumers.
set(_consumer_dir "${KFCORE_CONTRACT_ROOT}/consumer")
set(_consumer_build "${KFCORE_CONTRACT_ROOT}/consumer-build")
file(MAKE_DIRECTORY "${_consumer_dir}")
file(WRITE "${_consumer_dir}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.20)\n"
"project(KFCorePackageConsumer LANGUAGES C CXX)\n"
"find_package(KFCore CONFIG REQUIRED)\n"
"if(NOT KFCore_PLUGIN_DIR STREQUAL \"${_prefix}/plugins\")\n"
"  message(FATAL_ERROR \"unexpected relocatable plugin dir: '\${KFCore_PLUGIN_DIR}'\")\n"
"endif()\n")

execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -S "${_consumer_dir}"
          -B "${_consumer_build}"
          "-DCMAKE_PREFIX_PATH=${_prefix}"
  RESULT_VARIABLE _consumer_result)
if(NOT _consumer_result EQUAL 0)
  message(FATAL_ERROR "KFCore install-tree consumer configure failed")
endif()

# Keep the package-manager identity aligned with the repository/product.
file(READ "${KFCORE_SOURCE_DIR}/vcpkg.json" _manifest)
string(JSON _manifest_name GET "${_manifest}" name)
if(NOT _manifest_name STREQUAL "kfcore")
  message(FATAL_ERROR "vcpkg manifest name must be 'kfcore', got '${_manifest_name}'")
endif()

message(STATUS "KFCore package contract passed")
