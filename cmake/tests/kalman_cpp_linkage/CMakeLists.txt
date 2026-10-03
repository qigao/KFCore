cmake_minimum_required(VERSION 3.20)

project(KFCoreKalmanCppLinkage LANGUAGES C CXX)

if(NOT DEFINED KFCORE_SOURCE_DIR OR KFCORE_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "KFCORE_SOURCE_DIR is required")
endif()

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
list(APPEND CMAKE_MODULE_PATH "${KFCORE_SOURCE_DIR}/cmake")
include(CmakeUtils)

add_subdirectory("${KFCORE_SOURCE_DIR}/vendor/miniblas" vendor-miniblas)
add_subdirectory("${KFCORE_SOURCE_DIR}/kalman" kalman)

add_executable(kalman_cpp_link cpp_link.cpp)
target_compile_features(kalman_cpp_link PRIVATE cxx_std_17)
target_link_libraries(kalman_cpp_link PRIVATE KFCore::kalman)
