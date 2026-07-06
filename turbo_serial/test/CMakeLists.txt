set(TURBOSERIAL_TEST_SOURCES)
foreach(source IN LISTS TURBOSERIAL_SOURCES)
  if(IS_ABSOLUTE "${source}")
    list(APPEND TURBOSERIAL_TEST_SOURCES "${source}")
  else()
    list(APPEND TURBOSERIAL_TEST_SOURCES "${TURBO_SERIAL_SOURCE_DIR}/${source}")
  endif()
endforeach()

add_executable(test_serialport test_turboserial.c)
target_sources(test_serialport PRIVATE ${TURBOSERIAL_TEST_SOURCES})
target_include_directories(test_serialport
  PRIVATE
    "${TURBO_SERIAL_BINARY_DIR}"
    "${TURBO_SERIAL_SOURCE_DIR}")
target_compile_definitions(test_serialport
  PRIVATE TURBO_SERIAL_EXPORTS ${TURBOSERIAL_PRIVATE_DEFS})
target_link_libraries(test_serialport
  PRIVATE TurboNet::Utils TurboNet::TinyTest ${TURBOSERIAL_PLATFORM_LIBS})
add_test(NAME test_serialport COMMAND test_serialport)
set_target_properties(test_serialport PROPERTIES FOLDER "turbo_serial/tests")

add_executable(test_turbo_serial_public_abi test_turbo_serial_public_abi.c)
target_link_libraries(test_turbo_serial_public_abi
  PRIVATE ${TARGET_NAME} TurboNet::TinyTest)
add_test(NAME test_turbo_serial_public_abi COMMAND test_turbo_serial_public_abi)
set_target_properties(test_turbo_serial_public_abi PROPERTIES FOLDER "turbo_serial/tests")
