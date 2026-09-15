add_executable(kfgesture-smoke main.cpp)
target_compile_features(kfgesture-smoke PRIVATE cxx_std_17)
target_link_libraries(kfgesture-smoke PRIVATE KFCore::hand_gesture)
set_target_properties(kfgesture-smoke PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "tools/kfgesture")
install(TARGETS kfgesture-smoke
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
