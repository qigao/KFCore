include(CMakeDependentOption)

set(CMAKE_COLOR_DIAGNOSTICS ON)

option(KFCORE_BUILD_TESTS "Build KFCore C tests" ON)
option(KFCORE_BUILD_NAVIGATION_TOOLS "Build navigation helper functions" ON)
option(KFCORE_BUILD_YOLO_TRACKING "Build YOLO ByteTrack tracking support" OFF)
option(KFCORE_BUILD_TENSORRT_YOLO "Build TensorRT YOLO detector support" OFF)
option(KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS
       "Build TensorRT YOLO SDK integration tests" OFF)
option(KFCORE_BUILD_APRILTAG "Build the vendored AprilTag detector library" ON)
option(KFCORE_BUILD_APRILTAG_EXAMPLES "Build the vendored AprilTag example programs" OFF)
option(KFCORE_BUILD_APRILTAG_TESTS "Build the vendored AprilTag upstream tests" OFF)
option(KFCORE_BUILD_APRILTAG_PYTHON "Build the vendored AprilTag Python wrapper" OFF)
option(KFCORE_ENABLE_WARNINGS "Enable compiler warnings for KFCore targets" ON)
option(KFCORE_INSTALL "Generate KFCore install and CMake package rules" ON)

set(ENABLE_TESTS "${KFCORE_BUILD_TESTS}" CACHE BOOL "Compatibility alias for KFCORE_BUILD_TESTS" FORCE)
set(BUILD_TESTS "${KFCORE_BUILD_TESTS}" CACHE BOOL "Compatibility alias for KFCORE_BUILD_TESTS" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build example programs" FORCE)
