cmake_add_test(
  NAME test_yolo_onnx
  SOURCES test_yolo_onnx.cpp
  LIBS KFCore::yolo_onnx TurboUtils::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "yolo_onnx/tests")

set(_yolo_test_drone "./yolo-models/yolov8n-drone.onnx")
set(_yolo_test_football "./yolo-models/yolov8n-football.onnx")
set(_yolo_test_parking "./yolo-models/yolov8n-parking.onnx")
cmake_add_test(
  NAME test_yolo_onnx_integration
  SOURCES
    test_yolo_onnx_integration.cpp
    ../../yolo_core/examples/yolo_domain_profile.cpp
  LIBS KFCore::yolo_onnx TurboUtils::TinyTest
  DEFS
    "KFCORE_YOLO_TEST_DRONE=\"${_yolo_test_drone}\""
    "KFCORE_YOLO_TEST_FOOTBALL=\"${_yolo_test_football}\""
    "KFCORE_YOLO_TEST_PARKING=\"${_yolo_test_parking}\""
  INCLUDES ../../yolo_core/examples
  COMPILE_FEATURES cxx_std_17
  TEST_PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  FOLDER "yolo_onnx/tests")

set(_yolo_test_person "./yolo-models/yolov8s.onnx")
cmake_add_test(
  NAME test_yolov8s_person_integration
  SOURCES test_yolov8s_person_integration.cpp
  LIBS KFCore::yolo_onnx TurboUtils::TinyTest
  DEFS "KFCORE_YOLO_TEST_PERSON=\"${_yolo_test_person}\""
  COMPILE_FEATURES cxx_std_17
  TEST_PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  FOLDER "yolo_onnx/tests")

if(WIN32)
  set_tests_properties(test_yolo_onnx test_yolo_onnx_integration
    test_yolov8s_person_integration PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$ENV{ONNXRUNTIME_ROOT}/lib")
endif()
