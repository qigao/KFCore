cmake_minimum_required(VERSION 3.20)

set(TARGET_NAME turbo_serial)
set(EXAMPLES_TARGET_NAME ${TARGET_NAME}_examples)
set(TURBO_SERIAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TURBO_SERIAL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(TURBOSERIAL_SOURCES
    serialport.c
    timing.c
    turbo_serial.c)

set(TURBOSERIAL_PRIVATE_DEFS TURBOSERIAL_ATBUILD)
set(TURBOSERIAL_PLATFORM_LIBS)
set(TURBOSERIAL_INSTALL_HEADERS turbo_serial.h)

if(WIN32)
  list(APPEND TURBOSERIAL_SOURCES windows.c)
  list(APPEND TURBOSERIAL_PLATFORM_LIBS setupapi cfgmgr32)
elseif(APPLE)
  list(APPEND TURBOSERIAL_SOURCES macosx.c)
  list(APPEND TURBOSERIAL_PLATFORM_LIBS "-framework IOKit" "-framework CoreFoundation")
  list(APPEND TURBOSERIAL_PRIVATE_DEFS HAVE_SYS_FILE_H=1 HAVE_CLOCK_GETTIME=1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include(CheckCSourceCompiles)
  include(CheckFunctionExists)
  include(CheckIncludeFile)
  include(CheckStructHasMember)

  check_include_file(sys/file.h HAVE_SYS_FILE_H)
  check_function_exists(realpath HAVE_REALPATH)
  check_function_exists(flock HAVE_FLOCK)
  check_function_exists(clock_gettime HAVE_CLOCK_GETTIME)
  check_c_source_compiles("#include <linux/serial.h>
int main(void) { struct serial_struct value; (void)value; return 0; }"
                          HAVE_STRUCT_SERIAL_STRUCT)
  check_struct_has_member("struct termios" c_ispeed linux/termios.h
                          HAVE_STRUCT_TERMIOS_C_ISPEED LANGUAGE C)
  check_struct_has_member("struct termios" c_ospeed linux/termios.h
                          HAVE_STRUCT_TERMIOS_C_OSPEED LANGUAGE C)
  check_c_source_compiles("#include <linux/termios.h>
int main(void) { struct termios2 value; (void)value; return 0; }"
                          HAVE_STRUCT_TERMIOS2)
  check_struct_has_member("struct termios2" c_ispeed linux/termios.h
                          HAVE_STRUCT_TERMIOS2_C_ISPEED LANGUAGE C)
  check_struct_has_member("struct termios2" c_ospeed linux/termios.h
                          HAVE_STRUCT_TERMIOS2_C_OSPEED LANGUAGE C)
  check_c_source_compiles("#include <linux/termios.h>
int main(void) { struct termiox value; (void)value; return 0; }"
                          HAVE_STRUCT_TERMIOX)

  list(APPEND TURBOSERIAL_SOURCES linux.c linux_termios.c linux_termios.h)
  list(APPEND TURBOSERIAL_PRIVATE_DEFS
       $<$<BOOL:${HAVE_SYS_FILE_H}>:HAVE_SYS_FILE_H=1>
       $<$<BOOL:${HAVE_REALPATH}>:HAVE_REALPATH=1>
       $<$<BOOL:${HAVE_FLOCK}>:HAVE_FLOCK=1>
       $<$<BOOL:${HAVE_CLOCK_GETTIME}>:HAVE_CLOCK_GETTIME=1>
       $<$<BOOL:${HAVE_STRUCT_SERIAL_STRUCT}>:HAVE_STRUCT_SERIAL_STRUCT=1>
       $<$<BOOL:${HAVE_STRUCT_TERMIOS_C_ISPEED}>:HAVE_STRUCT_TERMIOS_C_ISPEED=1>
       $<$<BOOL:${HAVE_STRUCT_TERMIOS_C_OSPEED}>:HAVE_STRUCT_TERMIOS_C_OSPEED=1>
       $<$<BOOL:${HAVE_STRUCT_TERMIOS2}>:HAVE_STRUCT_TERMIOS2=1>
       $<$<BOOL:${HAVE_STRUCT_TERMIOS2_C_ISPEED}>:HAVE_STRUCT_TERMIOS2_C_ISPEED=1>
       $<$<BOOL:${HAVE_STRUCT_TERMIOS2_C_OSPEED}>:HAVE_STRUCT_TERMIOS2_C_OSPEED=1>
       $<$<BOOL:${HAVE_STRUCT_TERMIOX}>:HAVE_STRUCT_TERMIOX=1>)
elseif(CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
  list(APPEND TURBOSERIAL_SOURCES freebsd.c)
  list(APPEND TURBOSERIAL_PLATFORM_LIBS usb)
  list(APPEND TURBOSERIAL_PRIVATE_DEFS HAVE_SYS_FILE_H=1 HAVE_FLOCK=1 HAVE_CLOCK_GETTIME=1)
else()
  list(APPEND TURBOSERIAL_PRIVATE_DEFS NO_ENUMERATION=1 NO_PORT_METADATA=1)
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/config.h)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/turbo_serial.h
  ${CMAKE_CURRENT_BINARY_DIR}/include/turbo_serial.h
  COPYONLY)

add_library(${TARGET_NAME} SHARED
  ${TURBOSERIAL_SOURCES}
  ${TURBOSERIAL_INSTALL_HEADERS})
add_library(TurboScript::turbo_serial ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(${TARGET_NAME}
  PRIVATE ${TURBOSERIAL_PRIVATE_DEFS})

target_link_libraries(${TARGET_NAME}
  PUBLIC TurboNet::Utils
  PRIVATE ${TURBOSERIAL_PLATFORM_LIBS})

set_target_properties(${TARGET_NAME} PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  DEFINE_SYMBOL TURBO_SERIAL_EXPORTS
  OUTPUT_NAME turbo_serial
  C_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE /wd4996)
else()
  target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra)
endif()

cmake_config_target(
  ${TARGET_NAME}
  FOLDER "turbo_serial"
  EXPORT_NAME turbo_serial)

install(FILES ${TURBOSERIAL_INSTALL_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
 
if(BUILD_TESTS AND ENABLE_TESTS)
  add_subdirectory(test)
endif()

add_subdirectory(examples)
