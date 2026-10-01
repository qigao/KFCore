set(_relation_test_folder "vision/core/relation/tests")

if(BUILD_TESTS)
cmake_add_test(
  NAME test_relation_decode
  SOURCES test_decode.cpp
  LIBS KFCore::relation_core Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_relation_test_folder}")


cmake_add_test(
  NAME test_relation_vocabulary_score
  SOURCES test_vocabulary_score.cpp
  LIBS KFCore::relation_core Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_relation_test_folder}")


cmake_add_test(
  NAME test_relation_predicate_text_api
  SOURCES test_predicate_text_api.cpp
  LIBS KFCore::relation_core Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_relation_test_folder}")


cmake_add_test(
  NAME test_relation_predicate_routing_gate_api
  SOURCES test_predicate_routing_gate_api.cpp
  LIBS KFCore::relation_core Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_relation_test_folder}")
endif()

if(KFCORE_BUILD_RELATION_ORT_CPU_QUALIFICATION)
add_executable(kfcore_relation_ort_cpu_qualification
  ort_cpu_qualification.cpp)
target_include_directories(kfcore_relation_ort_cpu_qualification PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../pipelines/scene_graph/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../yolo/include)
target_compile_features(kfcore_relation_ort_cpu_qualification PRIVATE cxx_std_17)
target_link_libraries(kfcore_relation_ort_cpu_qualification PRIVATE
  KFCore::relation_core)
set_target_properties(kfcore_relation_ort_cpu_qualification PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "${_relation_test_folder}")
endif()
