add_library(kfcore_yolo_tracking SHARED src/tracking.cpp)
add_library(KFCore::yolo_tracking ALIAS kfcore_yolo_tracking)

target_link_libraries(kfcore_yolo_tracking PUBLIC KFCore::trackers)
target_include_directories(kfcore_yolo_tracking PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(kfcore_yolo_tracking PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  EXPORT_NAME yolo_tracking
  FOLDER "tensorrt_yolo"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(BUILD_TESTS)
  add_executable(test_yolo_tracking tests/test_yolo_tracking.cpp)
  target_link_libraries(test_yolo_tracking PRIVATE
    KFCore::yolo_tracking
    TurboUtils::TinyTest)
  set_target_properties(test_yolo_tracking PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_yolo/tests")
  add_test(NAME test_yolo_tracking COMMAND test_yolo_tracking)
endif()

install(TARGETS kfcore_yolo_tracking
  EXPORT KFCoreTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
