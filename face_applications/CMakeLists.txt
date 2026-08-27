find_package(OpenCVLite REQUIRED COMPONENTS core imgproc)

add_library(kfcore_face_applications SHARED
  src/error.cpp
  src/geometry.cpp
  src/preprocess.cpp)
target_include_directories(kfcore_face_applications
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_face_applications PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_applications
  PUBLIC
    KFCore::face_models
    KFCore::tensorrt_yolo
    KFCore::yolo_opencv
    OpenCVLite::core
    OpenCVLite::imgproc)
set_target_properties(kfcore_face_applications PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "face_applications"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_face_applications
  ALIAS KFCore::face_applications
  EXPORT_NAME face_applications
  EXPORT_SET KFCoreTargets
  FOLDER "face_applications")

if(BUILD_TESTS)
  add_executable(test_face_geometry tests/test_geometry.cpp)
  target_link_libraries(test_face_geometry PRIVATE
    KFCore::face_applications
    TurboUtils::TinyTest)
  set_target_properties(test_face_geometry PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/tests")
  add_test(NAME test_face_geometry COMMAND test_face_geometry)

  add_executable(test_face_preprocess tests/test_preprocess.cpp)
  target_link_libraries(test_face_preprocess PRIVATE
    KFCore::face_applications
    TurboUtils::TinyTest)
  set_target_properties(test_face_preprocess PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    FOLDER "face_applications/tests")
  add_test(NAME test_face_preprocess COMMAND test_face_preprocess)

  if(WIN32)
    set_tests_properties(test_face_geometry test_face_preprocess PROPERTIES
      ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$ENV{TENSORRT_ROOT}/bin;PATH=path_list_prepend:$ENV{OPENCV_LITE_ROOT}/bin")
  endif()
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
