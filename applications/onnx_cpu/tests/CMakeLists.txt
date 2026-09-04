set(_cpu_face_tests)
foreach(_cpu_face_test IN ITEMS geometry preprocess composer)
  set(_cpu_face_target test_cpu_face_${_cpu_face_test})
  cmake_add_test(
    NAME ${_cpu_face_target}
    SOURCES ${_cpu_face_target}.cpp
    LIBS KFCore::face_applications_cpu Salts::TinyTest
    INCLUDES ../src
    COMPILE_FEATURES cxx_std_17
    FOLDER "applications/onnx_cpu/tests")
  list(APPEND _cpu_face_tests ${_cpu_face_target})
endforeach()

cmake_add_test(
  NAME test_cpu_face_application_validation
  SOURCES test_cpu_face_application_validation.cpp
  LIBS KFCore::face_applications_cpu Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "applications/onnx_cpu/tests")
list(APPEND _cpu_face_tests test_cpu_face_application_validation)

if(WIN32)
  set_tests_properties(${_cpu_face_tests} PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$ENV{ONNXRUNTIME_ROOT}/lib")
endif()
