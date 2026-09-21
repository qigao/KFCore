add_library(kfcore_gesture_interaction STATIC src/composition_esn.cpp src/interaction.cpp src/experiment.cpp)
if(WIN32)
  target_sources(kfcore_gesture_interaction PRIVATE src/experiment_file.cpp)
endif()
add_library(KFCore::gesture_interaction ALIAS kfcore_gesture_interaction)
target_include_directories(kfcore_gesture_interaction PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# The public API uses C++17 optional and nested namespaces.
target_compile_features(kfcore_gesture_interaction PUBLIC cxx_std_17)
target_link_libraries(kfcore_gesture_interaction PUBLIC KFCore::mediapipe KFCore::esn)
target_link_libraries(kfcore_gesture_interaction PRIVATE Salts::Crypto)
set_target_properties(kfcore_gesture_interaction PROPERTIES
  EXPORT_NAME gesture_interaction FOLDER "vision/interaction")
install(TARGETS kfcore_gesture_interaction EXPORT KFCoreTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
if(BUILD_TESTS)
  add_executable(test_gesture_interaction tests/test_interaction.cpp)
  target_link_libraries(test_gesture_interaction PRIVATE KFCore::gesture_interaction Salts::TinyTest)
  add_test(NAME test_gesture_interaction COMMAND test_gesture_interaction)
endif()
