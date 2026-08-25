cmake_minimum_required(VERSION 3.20)

project(FindTensorRTFixture NONE)

if(NOT DEFINED FIND_TENSORRT_MODULE_DIR)
  message(FATAL_ERROR "FIND_TENSORRT_MODULE_DIR is required")
endif()
if(NOT DEFINED FIND_TENSORRT_EXPECTED_VERSION)
  message(FATAL_ERROR "FIND_TENSORRT_EXPECTED_VERSION is required")
endif()
if(NOT DEFINED FIND_TENSORRT_SCENARIO)
  set(FIND_TENSORRT_SCENARIO discover)
endif()

list(PREPEND CMAKE_MODULE_PATH "${FIND_TENSORRT_MODULE_DIR}")

function(_fixture_add_tensorrt_target target library_name)
  if(NOT DEFINED FIND_TENSORRT_PRESEEDED_ROOT)
    message(FATAL_ERROR "FIND_TENSORRT_PRESEEDED_ROOT is required for preseeded targets")
  endif()
  if(WIN32)
    set(_library "${FIND_TENSORRT_PRESEEDED_ROOT}/lib/${library_name}.lib")
  else()
    set(_library "${FIND_TENSORRT_PRESEEDED_ROOT}/lib/lib${library_name}.a")
  endif()
  add_library(${target} UNKNOWN IMPORTED)
  set_target_properties(${target} PROPERTIES
    IMPORTED_LOCATION "${_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${FIND_TENSORRT_PRESEEDED_ROOT}/include")
endfunction()

if(FIND_TENSORRT_SCENARIO MATCHES "^(dependency_first|partial|root_mismatch)$")
  _fixture_add_tensorrt_target(TensorRT::nvinfer nvinfer_11)
  if(NOT FIND_TENSORRT_SCENARIO STREQUAL "partial")
    _fixture_add_tensorrt_target(TensorRT::nvinfer_plugin nvinfer_plugin_11)
    set_property(TARGET TensorRT::nvinfer_plugin PROPERTY
      INTERFACE_LINK_LIBRARIES TensorRT::nvinfer)
  endif()
endif()

find_package(TensorRT REQUIRED)
if(FIND_TENSORRT_SCENARIO STREQUAL "repeated")
  find_package(TensorRT REQUIRED)
endif()

if(NOT TensorRT_VERSION STREQUAL FIND_TENSORRT_EXPECTED_VERSION)
  message(FATAL_ERROR
    "Unexpected TensorRT version ${TensorRT_VERSION}; expected ${FIND_TENSORRT_EXPECTED_VERSION}")
endif()
