add_library(kfcore_mediapipe STATIC src/hand_landmarker.cpp src/gesture_recognizer.cpp)
add_library(KFCore::mediapipe ALIAS kfcore_mediapipe)
target_include_directories(kfcore_mediapipe PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(kfcore_mediapipe PUBLIC KFCore::hand_model_runtime)
set_target_properties(kfcore_mediapipe PROPERTIES
  EXPORT_NAME mediapipe FOLDER "vision/mediapipe")
install(TARGETS kfcore_mediapipe EXPORT KFCoreTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

if(BUILD_TESTS)
  add_executable(test_mediapipe tests/test_mediapipe.cpp)
  target_link_libraries(test_mediapipe PRIVATE KFCore::mediapipe Salts::TinyTest)
  add_test(NAME test_mediapipe COMMAND test_mediapipe)
  add_executable(test_gesture_recognizer tests/test_gesture_recognizer.cpp)
  target_link_libraries(test_gesture_recognizer PRIVATE KFCore::mediapipe Salts::TinyTest)
  add_test(NAME test_gesture_features COMMAND test_gesture_recognizer --filter "features")
  if(DEFINED ENV{KFCORE_GESTURE_TEST_ASSETS} AND DEFINED ENV{KFCORE_MEDIAPIPE_TEST_ASSETS} AND KFCORE_ENABLE_ONNXRUNTIME)
    add_test(NAME test_gesture_recognizer_onnx COMMAND test_gesture_recognizer --filter "runs world")
    set_tests_properties(test_gesture_recognizer_onnx PROPERTIES ENVIRONMENT
      "KFCORE_GESTURE_TEST_ASSETS=$ENV{KFCORE_GESTURE_TEST_ASSETS};KFCORE_MEDIAPIPE_TEST_ASSETS=$ENV{KFCORE_MEDIAPIPE_TEST_ASSETS};KFCORE_MEDIAPIPE_TEST_BACKEND=$<TARGET_FILE:kfcore_backend_onnxruntime>")
  endif()
  if(KFCORE_ENABLE_ONNXRUNTIME)
    add_executable(test_mediapipe_onnx tests/test_mediapipe_onnx.cpp)
    target_link_libraries(test_mediapipe_onnx PRIVATE KFCore::mediapipe Salts::TinyTest)
    # Real weights and the recorded hand image are opt-in, never synthetic substitutes.
    if(DEFINED ENV{KFCORE_MEDIAPIPE_TEST_ASSETS})
      add_test(NAME test_mediapipe_onnx COMMAND test_mediapipe_onnx)
      set_tests_properties(test_mediapipe_onnx PROPERTIES ENVIRONMENT
        "KFCORE_MEDIAPIPE_TEST_ASSETS=$ENV{KFCORE_MEDIAPIPE_TEST_ASSETS};KFCORE_MEDIAPIPE_TEST_BACKEND=$<TARGET_FILE:kfcore_backend_onnxruntime>")
    endif()
  endif()
endif()
