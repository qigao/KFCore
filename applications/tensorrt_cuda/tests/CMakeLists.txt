find_package(OpenCVLite REQUIRED COMPONENTS core imgproc)
set(_face_application_test_folder "face_applications_cuda/tests")
set(_face_application_tests
  test_face_swap_cli
  test_face_swap_demo_ui
  test_face_geometry
  test_face_composer
  test_face_application_validation)

cmake_add_test(
  NAME test_face_swap_cli
  SOURCES test_face_swap_cli.cpp ../examples/face_swap_cli.cpp
  LIBS KFCore::face_models_cuda Salts::Core Salts::TinyTest
  INCLUDES ../examples
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_application_test_folder}")
cmake_add_test(
  NAME test_face_swap_demo_ui
  SOURCES
    test_face_swap_demo_ui.cpp
    ../examples/face_swap_demo_metrics.cpp
    ../examples/face_swap_demo_ui.cpp
  LIBS
    KFCore::face_applications_cuda
    OpenCVLite::core
    OpenCVLite::imgproc
    Salts::TinyTest
  INCLUDES ../examples
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_application_test_folder}")
cmake_add_test(
  NAME test_face_geometry
  SOURCES test_geometry.cpp
  LIBS KFCore::face_applications_cuda Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_application_test_folder}")
cmake_add_test(
  NAME test_face_composer
  SOURCES test_composer.cpp
  LIBS KFCore::face_applications_cuda Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_application_test_folder}")
cmake_add_test(
  NAME test_face_application_validation
  SOURCES test_application_validation.cpp
  LIBS KFCore::face_applications_cuda Salts::TinyTest
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_face_application_test_folder}")

if(WIN32)
  set_tests_properties(${_face_application_tests} PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR};PATH=path_list_prepend:$ENV{OPENCV_LITE_ROOT}/bin;PATH=path_list_prepend:$ENV{SALTS_ROOT}/bin")
endif()
