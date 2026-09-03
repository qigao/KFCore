set(_face_models_cuda_test_folder "face_models_cuda/tests")
set(_face_models_cuda_unit_tests
  test_face_model_contracts
  test_face_swap_adapter_api)

cmake_add_test(
  NAME test_face_model_contracts
  SOURCES test_contracts.cpp
  LIBS KFCore::face_models_cuda Salts::TinyTest
  INCLUDES ../src ${CMAKE_SOURCE_DIR}/face_model_core/src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_models_cuda_test_folder}")
cmake_add_test(
  NAME test_face_swap_adapter_api
  SOURCES test_swap_adapter_api.cpp
  LIBS KFCore::face_models_cuda Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_models_cuda_test_folder}")
set(_face_mesh_engine_root "tensorrt/${KFCORE_TENSORRT_ENGINE_PROFILE}")
set(_face_mesh_cuda_detector
  "./yolo-models/${_face_mesh_engine_root}/yolov11n-face.engine")
set(_face_mesh_cuda_landmarker
  "./yolo-models/${_face_mesh_engine_root}/face_landmark.engine")
cmake_add_test(
  NAME test_facemesh_cuda_integration
  SOURCES test_facemesh_integration.cpp
  LIBS KFCore::face_models_cuda Salts::TinyTest
  DEFS
    "KFCORE_FACE_TRT_TEST_DETECTOR=\"${_face_mesh_cuda_detector}\""
    "KFCORE_FACE_TRT_TEST_LANDMARKER=\"${_face_mesh_cuda_landmarker}\""
  COMPILE_FEATURES cxx_std_17
  TEST_PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  FOLDER "${_face_models_cuda_test_folder}")

set(_face_engine_root "tensorrt/${KFCORE_TENSORRT_ENGINE_PROFILE}")
foreach(_face_model_mapping IN ITEMS
    "arcface_w600k_r50;arcface_w600k_r50"
    "age-gender;age_gender"
    "2dfan4;face68"
    "inswapper_128;inswapper_128"
    "gfpgan_1.4;gfpgan_1_4")
  list(GET _face_model_mapping 0 _face_model)
  list(GET _face_model_mapping 1 _face_model_id)
  set(_face_engine_${_face_model_id}
    "./yolo-models/${_face_engine_root}/${_face_model}.engine")
endforeach()

add_executable(test_face_models_integration test_face_models_integration.cpp)
target_link_libraries(test_face_models_integration PRIVATE
  KFCore::face_models_cuda Salts::TinyTest)
target_compile_features(test_face_models_integration PRIVATE cxx_std_17)
cmake_config_target(test_face_models_integration
  NO_INSTALL FOLDER "${_face_models_cuda_test_folder}")

function(_kfcore_add_face_model_integration_test test_suffix model_kind engine_variable)
  set(test_name test_face_models_integration_${test_suffix})
  add_test(NAME ${test_name} COMMAND test_face_models_integration)
  set_property(TEST ${test_name} PROPERTY ENVIRONMENT
    "KFCORE_FACE_MODEL_TEST_KIND=${model_kind}"
    "KFCORE_RUNTIME_TENSORRT_TEST_ENGINE_${engine_variable}=${ARGN}")
  set_property(TEST ${test_name} PROPERTY WORKING_DIRECTORY
    "${PROJECT_SOURCE_DIR}")
  set_property(GLOBAL APPEND PROPERTY KFCORE_FACE_MODEL_INTEGRATION_TESTS ${test_name})
endfunction()

_kfcore_add_face_model_integration_test(
  arcface arcface ARCFACE "${_face_engine_arcface_w600k_r50}")
_kfcore_add_face_model_integration_test(
  age_gender age_gender AGE_GENDER "${_face_engine_age_gender}")
_kfcore_add_face_model_integration_test(
  face68 face68 FACE68 "${_face_engine_face68}")
_kfcore_add_face_model_integration_test(
  inswapper inswapper INSWAPPER "${_face_engine_inswapper_128}")
_kfcore_add_face_model_integration_test(
  gfpgan gfpgan GFPGAN "${_face_engine_gfpgan_1_4}")

if(WIN32)
  get_property(_face_model_integration_tests GLOBAL
    PROPERTY KFCORE_FACE_MODEL_INTEGRATION_TESTS)
  set_tests_properties(
    ${_face_models_cuda_unit_tests}
    test_facemesh_cuda_integration
    ${_face_model_integration_tests}
    PROPERTIES ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
endif()
