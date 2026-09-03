set(_face_model_core_test_folder "face_model_core/tests")

cmake_add_test(
  NAME test_face_model_results
  SOURCES test_results.cpp
  LIBS KFCore::face_model_core Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_model_core_test_folder}")

cmake_add_test(
  NAME test_inswapper_embedding
  SOURCES test_inswapper_embedding.cpp
  LIBS KFCore::face_model_core Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_model_core_test_folder}")

cmake_add_test(
  NAME test_face_pipeline_api
  SOURCES test_face_pipeline_api.cpp
  LIBS KFCore::face_model_core Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_model_core_test_folder}")

cmake_add_test(
  NAME test_face_pipeline
  SOURCES test_face_pipeline.cpp
  LIBS KFCore::face_model_core Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_model_core_test_folder}")

cmake_add_test(
  NAME test_facemesh_decode
  SOURCES test_facemesh_decode.cpp
  LIBS KFCore::face_model_core Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_model_core_test_folder}")
