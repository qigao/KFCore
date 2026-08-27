include(CMakeDependentOption)

set(CMAKE_COLOR_DIAGNOSTICS ON)

option(KFCORE_BUILD_TESTS "Build KFCore C tests" ON)
option(KFCORE_BUILD_NAVIGATION_TOOLS "Build navigation helper functions" ON)
option(KFCORE_BUILD_IMAGE_PROCESSOR "Build the reusable CUDA image processor" OFF)
option(KFCORE_BUILD_SIFT "Build the reusable SIFT extractor contract" OFF)
option(KFCORE_BUILD_SIFT_POPSIFT "Build the maintained CUDA SIFT backend" OFF)
option(KFCORE_BUILD_YOLO_TRACKING "Build YOLO ByteTrack tracking support" OFF)
option(KFCORE_BUILD_TENSORRT_YOLO "Build TensorRT YOLO detector support" OFF)
option(KFCORE_BUILD_TENSORRT_RUNTIME "Build the model-neutral TensorRT runtime" OFF)
option(KFCORE_BUILD_FACE_MODELS "Build strict TensorRT face model adapters" OFF)
option(KFCORE_BUILD_FACE_APPLICATIONS "Build OpenCV TensorRT face applications" OFF)
cmake_dependent_option(
  KFCORE_BUILD_FACE_APPLICATION_EXAMPLES
  "Build TensorRT face application examples"
  OFF
  "KFCORE_BUILD_FACE_APPLICATIONS"
  OFF)
cmake_dependent_option(
  KFCORE_BUILD_FACE_APPLICATION_INTEGRATION_TESTS
  "Build real-engine TensorRT face application integration tests"
  OFF
  "KFCORE_BUILD_FACE_APPLICATIONS"
  OFF)
set(KFCORE_FACE_APPLICATION_TEST_ENGINE_12FACE
    "$ENV{KFCORE_FACE_APPLICATION_TEST_ENGINE_12FACE}" CACHE FILEPATH
    "Trusted YOLOv12-face TensorRT engine used by the face application integration test")
set(KFCORE_FACE_APPLICATION_TEST_MATRIX
    "$ENV{KFCORE_FACE_APPLICATION_TEST_MATRIX}" CACHE FILEPATH
    "Trusted InSwapper matrix used by the face application integration test")
set(KFCORE_FACE_APPLICATION_TEST_SOURCE_IMAGE
    "$ENV{KFCORE_FACE_APPLICATION_TEST_SOURCE_IMAGE}" CACHE FILEPATH
    "Source face image used by the face application integration test")
set(KFCORE_FACE_APPLICATION_TEST_TARGET_IMAGE
    "$ENV{KFCORE_FACE_APPLICATION_TEST_TARGET_IMAGE}" CACHE FILEPATH
    "Target face image used by the face application integration test")
option(KFCORE_BUILD_TENSORRT_RUNTIME_INTEGRATION_TESTS
       "Build real-engine TensorRT runtime and face model integration tests" OFF)
set(KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE
    "$ENV{KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE}" CACHE FILEPATH
    "Trusted ArcFace TensorRT engine used by opt-in runtime integration tests")
set(KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER
    "$ENV{KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER}" CACHE FILEPATH
    "Trusted age/gender TensorRT engine used by opt-in runtime integration tests")
set(KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68
    "$ENV{KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68}" CACHE FILEPATH
    "Trusted Face68 TensorRT engine used by opt-in runtime integration tests")
set(KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER
    "$ENV{KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER}" CACHE FILEPATH
    "Trusted InSwapper TensorRT engine used by opt-in runtime integration tests")
set(KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN
    "$ENV{KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN}" CACHE FILEPATH
    "Trusted GFPGAN TensorRT engine used by opt-in runtime integration tests")
option(KFCORE_BUILD_YOLO_OPENCV "Build the optional OpenCV Lite YOLO adapter" OFF)
option(KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS
       "Build TensorRT YOLO SDK integration tests" OFF)
set(KFCORE_TENSORRT_TEST_ENGINE "$ENV{KFCORE_TENSORRT_TEST_ENGINE}" CACHE FILEPATH
    "Trusted TensorRT engine used by the opt-in integration CTest")
set(KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE
    "$ENV{KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE}" CACHE FILEPATH
    "Optional trusted YOLO11-face TensorRT engine used by the opt-in integration CTest")
option(KFCORE_BUILD_APRILTAG "Build the vendored AprilTag detector library" ON)
option(KFCORE_BUILD_APRILTAG_EXAMPLES "Build the vendored AprilTag example programs" OFF)
option(KFCORE_BUILD_APRILTAG_TESTS "Build the vendored AprilTag upstream tests" OFF)
option(KFCORE_BUILD_APRILTAG_PYTHON "Build the vendored AprilTag Python wrapper" OFF)
option(KFCORE_ENABLE_WARNINGS "Enable compiler warnings for KFCore targets" ON)
option(KFCORE_INSTALL "Generate KFCore install and CMake package rules" ON)

set(ENABLE_TESTS "${KFCORE_BUILD_TESTS}" CACHE BOOL "Compatibility alias for KFCORE_BUILD_TESTS" FORCE)
set(BUILD_TESTS "${KFCORE_BUILD_TESTS}" CACHE BOOL "Compatibility alias for KFCORE_BUILD_TESTS" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build example programs" FORCE)
