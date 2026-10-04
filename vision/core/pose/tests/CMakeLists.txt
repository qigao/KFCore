set(_pose_test_folder "vision/core/pose/tests")

if(BUILD_TESTS)
cmake_add_test(
  NAME test_pose_schema
  SOURCES test_schema.cpp
  LIBS KFCore::pose_core Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_pose_test_folder}")

cmake_add_test(
  NAME test_pose_simcc_decode
  SOURCES test_simcc_decode.cpp
  LIBS KFCore::pose_core Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_pose_test_folder}")

cmake_add_test(
  NAME test_pose_heatmap_decode
  SOURCES test_heatmap_decode.cpp
  LIBS KFCore::pose_core Salts::TinyTest
  INCLUDES ../src
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_pose_test_folder}")

cmake_add_test(
  NAME test_pose_rtmw_contract
  SOURCES test_rtmw_contract.cpp
  LIBS KFCore::pose_core Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_pose_test_folder}")
endif()

if(KFCORE_BUILD_POSE_RTMW_ORT_CPU_QUALIFICATION)
add_executable(kfcore_pose_rtmw_ort_cpu_qualification
  rtmw_ort_cpu_qualification.cpp)
target_include_directories(kfcore_pose_rtmw_ort_cpu_qualification PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(kfcore_pose_rtmw_ort_cpu_qualification PRIVATE cxx_std_17)
target_link_libraries(kfcore_pose_rtmw_ort_cpu_qualification PRIVATE
  KFCore::pose_core)
set_target_properties(kfcore_pose_rtmw_ort_cpu_qualification PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "${_pose_test_folder}")
endif()
