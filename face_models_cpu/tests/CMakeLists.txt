set(_face_models_cpu_test_folder "face_models_cpu/tests")

cmake_add_test(
  NAME test_cpu_face_models_api
  SOURCES test_cpu_face_models_api.cpp
  LIBS KFCore::face_models_cpu TurboUtils::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_models_cpu_test_folder}")

cmake_add_test(
  NAME test_face_models_cpu_model_names
  SOURCES test_cpu_model_names.cpp
  LIBS KFCore::face_models_cpu TurboUtils::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_models_cpu_test_folder}")

kfcore_resolve_model_path(
  "$ENV{KFCORE_MODEL_ROOT}"
  "yolov12n-face.onnx"
  _face_cpu_detector)
kfcore_resolve_model_path(
  "$ENV{KFCORE_MODEL_ROOT}"
  "MediaPipeFaceLandmarkDetector.onnx"
  _face_cpu_landmarker)
cmake_add_test(
  NAME test_cpu_facemesh_integration
  SOURCES test_cpu_facemesh_integration.cpp
  LIBS KFCore::face_models_cpu TurboUtils::TinyTest
  DEFS
    "KFCORE_FACE_CPU_TEST_DETECTOR=\"${_face_cpu_detector}\""
    "KFCORE_FACE_CPU_TEST_LANDMARKER=\"${_face_cpu_landmarker}\""
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_models_cpu_test_folder}")

add_executable(test_cpu_face_models_integration test_cpu_face_models_integration.cpp)
target_link_libraries(test_cpu_face_models_integration PRIVATE
  KFCore::face_models_cpu TurboUtils::TinyTest)
target_compile_features(test_cpu_face_models_integration PRIVATE cxx_std_17)
cmake_config_target(test_cpu_face_models_integration
  NO_INSTALL FOLDER "${_face_models_cpu_test_folder}")

set(_face_models_cpu_tests test_cpu_face_models_api test_cpu_facemesh_integration)
foreach(_cpu_face_model_mapping IN ITEMS
    "2dfan4;face68"
    "arcface_w600k_r50;arcface"
    "age-gender;age_gender"
    "inswapper_128;inswapper"
    "gfpgan_1.4;gfpgan")
  list(GET _cpu_face_model_mapping 0 _cpu_face_model_file)
  list(GET _cpu_face_model_mapping 1 _cpu_face_model_kind)
  kfcore_resolve_model_path(
    "$ENV{KFCORE_MODEL_ROOT}"
    "${_cpu_face_model_file}.onnx"
    _cpu_face_model_path)
  set(_cpu_face_test_name test_cpu_face_models_integration_${_cpu_face_model_kind})
  add_test(NAME ${_cpu_face_test_name} COMMAND test_cpu_face_models_integration)
  set_property(TEST ${_cpu_face_test_name} PROPERTY ENVIRONMENT
    "KFCORE_CPU_FACE_MODEL_TEST_KIND=${_cpu_face_model_kind}"
    "KFCORE_CPU_FACE_MODEL_TEST_PATH=${_cpu_face_model_path}")
  list(APPEND _face_models_cpu_tests ${_cpu_face_test_name})
endforeach()

if(WIN32)
  set_tests_properties(${_face_models_cpu_tests} PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$ENV{ONNXRUNTIME_ROOT}/lib")
endif()
