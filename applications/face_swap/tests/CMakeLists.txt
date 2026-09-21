add_executable(test_face_application_api test_face_application_api.cpp)
target_link_libraries(test_face_application_api
  PRIVATE KFCore::face_swap_application Salts::TinyTest)
set_target_properties(test_face_application_api PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "applications/face_swap/tests")
add_test(NAME test_face_application_api COMMAND test_face_application_api)

add_executable(test_face_application_validation test_face_application_validation.cpp)
target_link_libraries(test_face_application_validation
  PRIVATE KFCore::face_swap_application Salts::TinyTest)
set_target_properties(test_face_application_validation PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "applications/face_swap/tests")
add_test(NAME test_face_application_validation COMMAND test_face_application_validation)

foreach(_face_swap_helper_test IN ITEMS
    test_face_preprocess
    test_face_geometry
    test_face_composer)
  add_executable(${_face_swap_helper_test} ${_face_swap_helper_test}.cpp)
  target_include_directories(${_face_swap_helper_test}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  target_link_libraries(${_face_swap_helper_test}
    PRIVATE KFCore::face_swap_application Salts::TinyTest)
  set_target_properties(${_face_swap_helper_test} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "applications/face_swap/tests")
  add_test(NAME ${_face_swap_helper_test} COMMAND ${_face_swap_helper_test})
endforeach()
